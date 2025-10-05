from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from ..common_blocks.bbox_embedding import BBoxUpdateEmbedding
from ..common_blocks.class_embedding import (
    ContrastiveBiasClassEmbedding,
    ContrastiveClassEmbedding,
)
from ....utils.config import BaseConfig
from ....utils.types import (
    ImageFeatures,
    OutputFeatures,
    QueryFeatures,
    TextFeatures,
)


@dataclass
class QuerySelectorConfig(BaseConfig["QuerySelector"]):
    num_queries: int
    emb_dim: int
    cls_emb_type: Literal["grounding_dino", "mm_grounding_dino"]


class QuerySelector(nn.Module):
    def __init__(self, cfg: QuerySelectorConfig):
        super().__init__()
        self.num_queries = cfg.num_queries

        # input projection of encoder outputs
        self.input_proj = nn.Sequential(
            nn.Linear(cfg.emb_dim, cfg.emb_dim), nn.LayerNorm(cfg.emb_dim)
        )

        # learned initial class embedding
        self.cls_init = nn.Parameter(torch.randn(cfg.num_queries, cfg.emb_dim))

        # cls embedding for outputs and to determine top-K locations
        if cfg.cls_emb_type == "grounding_dino":
            self.cls_embed = ContrastiveClassEmbedding()
        elif cfg.cls_emb_type == "mm_grounding_dino":
            self.cls_embed = ContrastiveBiasClassEmbedding()
        self.bbox_update_embed = BBoxUpdateEmbedding(
            cfg.emb_dim, zero_init=True
        )

    def forward(
        self, img: ImageFeatures, txt: TextFeatures
    ) -> tuple[OutputFeatures, QueryFeatures]:
        B, N_txt, _ = txt["feat"].shape
        device = txt["feat"].device

        # project the input features
        img_feat = self.input_proj(img["feat"])

        # get idxs of img features with highest similarity to txt features
        enc_cls = self.cls_embed(
            img_feat, img["mask"], txt["feat"], txt["mask"]
        )  # [B, N_img, N_txt]
        max_sim, _ = enc_cls.max(-1)  # [B, N_img]
        _, topk_idx = torch.topk(
            max_sim, k=self.num_queries, dim=-1
        )  # [B, N_q, N_txt], [B, N_q]

        # init bbox queries for the decoder and update them based on encoder output
        query_bbox_init = self.generate_initial_bbox_proposals(
            img["coor"], img["shapes"]
        )  # [B, N_img, 4]
        query_bbox_update = self.bbox_update_embed(img_feat)  # [B, N_img, 4]
        query_bbox = query_bbox_init + query_bbox_update  # [B, N_img, 4]

        # select encoder outputs
        enc_cls = torch.gather(
            enc_cls,
            dim=1,
            index=topk_idx[:, :, None].expand(
                -1, -1, N_txt
            ),  # manual broadcast to [B, N_q, N_txt]
        )  # [B, N_q, N_txt]

        # select bbox queries
        query_bbox = torch.gather(
            query_bbox,
            dim=1,
            index=topk_idx[:, :, None].expand(
                -1, -1, 4
            ),  # manual broadcast to [B, N_q, 4]
        )  # [B, N_q, 4]

        # init cls queries
        query_cls = self.cls_init.expand(B, -1, -1)  # [B, N_q, C]

        return (
            {
                "cls": enc_cls,  # [B, N_q, N_txt]
                "bbox": torch.sigmoid(
                    query_bbox
                ),  # [B, N_q, 4] in cxcywh sigmoid coordinates
                "mask": torch.zeros(
                    B, self.num_queries, device=device, dtype=torch.bool
                ),  # [B, N_q]
                "cap_ids": txt["cap_ids"],  # [B, N_txt]
            },
            {
                "feat": query_cls,  # [B, N_q, C]
                "bbox": query_bbox.detach(),  # [B, N_q, 4] in cxcywh logit coordinates
                "mask": torch.zeros(
                    B, self.num_queries, device=device, dtype=torch.bool
                ),  # [B, N_q]
                "att_mask": torch.zeros(
                    B,
                    self.num_queries,
                    self.num_queries,
                    device=device,
                    dtype=torch.bool,
                ),  # [B, N_q, N_q]
            },
        )

    def generate_initial_bbox_proposals(
        self, flat_coor: torch.Tensor, hw_shapes: torch.Tensor
    ) -> torch.Tensor:
        # get xy coordinates
        xy = flat_coor  # [B, N_img, 2]

        # generate initial hws for each level
        level_sizes = hw_shapes.prod(dim=-1)  # [L]
        level_inits = 0.05 * 2 ** torch.arange(
            level_sizes.shape[0], device=xy.device
        )  # [L]
        wh = level_inits.repeat_interleave(level_sizes)[None, :, None].expand(
            xy.shape[0], -1, 2
        )  # [N_img] -> [B, N_img, 2]

        # concat them
        bbox_proposals = torch.cat([xy, wh], dim=-1)  # [B, N_img, 4]

        # clamp degenerate initializations
        # this should only affect images that have fewer usable pixels than N_queries
        bbox_proposals.clamp_(0.01, 0.99)

        return torch.logit(bbox_proposals)  # [B, N_img, 4] in logit cxcywh


QuerySelectorConfig._target_class = QuerySelector
