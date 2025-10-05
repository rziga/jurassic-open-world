from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from ..common_blocks.bbox_embedding import (
    BBoxPositionEmbedding,
    BBoxUpdateEmbedding,
)
from ..common_blocks.class_embedding import (
    ContrastiveBiasClassEmbedding,
    ContrastiveClassEmbedding,
)
from ..common_blocks.skip_blocks import (
    FFNBlock,
    PositionModulatedCrossAttentionBlock,
    PositionModulatedDeformableAttentionBlock,
    PositionModulatedSelfAttentionBlock,
)
from ....utils.config import BaseConfig
from ....utils.types import (
    ImageFeatures,
    OutputFeatures,
    QueryFeatures,
    TextFeatures,
)


@dataclass
class DecoderLayerConfig(BaseConfig["DecoderLayer"]):
    emb_dim: int
    ffn_dim: int
    num_heads: int
    num_points: int
    num_levels: int
    dropout: float
    attention_dropout: float
    use_fusion: bool
    cls_emb_type: Literal["grounding_dino", "mm_grounding_dino"]
    use_legacy_pos: bool


class DecoderLayer(nn.Module):
    """Decoder layer.

    1) query self attention
    2) query-txt cross attention (if using fusion)
    3) query-img cross attention
    4) query ffn
    5) predict bbox update + cls
    """

    def __init__(self, cfg: DecoderLayerConfig):
        super().__init__()
        self.cfg = cfg

        self.self_att = PositionModulatedSelfAttentionBlock(
            cfg.emb_dim, cfg.num_heads, cfg.dropout, cfg.attention_dropout
        )

        self.txt_cross_att = (
            PositionModulatedCrossAttentionBlock(
                cfg.emb_dim, cfg.num_heads, cfg.dropout, cfg.attention_dropout
            )
            if cfg.use_fusion
            else None
        )

        self.img_cross_att = PositionModulatedDeformableAttentionBlock(
            cfg.emb_dim,
            cfg.num_levels,
            cfg.num_heads,
            cfg.num_points,
            cfg.dropout,
        )

        self.ffn = FFNBlock(cfg.emb_dim, cfg.ffn_dim, cfg.dropout)

        self.bbox_pos_embed = BBoxPositionEmbedding(cfg.emb_dim)
        self.bbox_update_embed = BBoxUpdateEmbedding(
            cfg.emb_dim, zero_init=True
        )

        if cfg.cls_emb_type == "grounding_dino":
            self.cls_embed = ContrastiveClassEmbedding()
        elif cfg.cls_emb_type == "mm_grounding_dino":
            self.cls_embed = ContrastiveBiasClassEmbedding()

        self.norm = nn.LayerNorm(cfg.emb_dim)

    def forward(
        self, query: QueryFeatures, img: ImageFeatures, txt: TextFeatures
    ) -> tuple[OutputFeatures, QueryFeatures]:
        """
        Decoder layer forward pass.

        Args:
            query (QueryFeatures): Query features
            img (ImageFeatures): Image features
            txt (TextFeatures): Text features

        Returns:
            out (tuple[OutputFeatures, QueryFeatures]): Ouput features and updated query features.
        """
        # copy so we don't modify out of scope
        query, img, txt = query.copy(), img.copy(), txt.copy()

        # detach queries from previous layer and
        # save undetached version for look forward twice update trick
        query_bbox_logit_undetached = query["bbox"]
        query_bbox_logit_detached = query["bbox"].detach()
        query_bbox_detached = torch.sigmoid(query_bbox_logit_detached)

        # get sine embedding based on top level bbox queries and embed them
        query_pos = self.bbox_pos_embed(
            query_bbox_detached[..., [1, 0, 2, 3]]
            if self.cfg.use_legacy_pos
            else query_bbox_detached
        )

        # --- update query features --- #

        # query self attention
        query["feat"] = self.self_att(
            query["feat"], query_pos, query["mask"], query["att_mask"]
        )

        # query txt cross attention
        if self.txt_cross_att is not None:
            query["feat"] = self.txt_cross_att(
                query["feat"], query_pos, txt["feat"], txt["mask"]
            )

        # query img cross attention
        query["feat"] = self.img_cross_att(
            img["feat"],
            img["mask"],
            img["shapes"],
            img["valid_ratios"],
            query["feat"],
            query_pos,
            query_bbox_detached,
        )

        # ffn
        query["feat"] = self.ffn(query["feat"])
        normed_query_feat = self.norm(query["feat"])

        # --- update query bbox - look forward twice trick --- #

        # update detached bbox for next layer
        bbox_update = self.bbox_update_embed(normed_query_feat)
        query["bbox"] = query_bbox_logit_detached + bbox_update

        # update undetached bbox for output
        out_bbox = torch.sigmoid(query_bbox_logit_undetached + bbox_update)
        out_cls = self.cls_embed(
            normed_query_feat, query["mask"], txt["feat"], txt["mask"]
        )

        out: OutputFeatures = {
            "cls": out_cls,
            "bbox": out_bbox,
            "mask": query["mask"],
            "cap_ids": txt["cap_ids"],
        }

        return out, query


DecoderLayerConfig._target_class = DecoderLayer
