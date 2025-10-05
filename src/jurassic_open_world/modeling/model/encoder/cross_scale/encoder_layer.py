from dataclasses import dataclass

from torch import nn

from ...common_blocks.fusion_layer import FusionLayer
from ...common_blocks.skip_blocks import (
    FFNBlock,
    PositionModulatedDeformableAttentionBlock,
    PositionModulatedSelfAttentionBlock,
)
from .....utils.config import BaseConfig
from .....utils.types import ImageFeatures, TextFeatures


@dataclass
class CrossScaleEncoderLayerConfig(BaseConfig["CrossScaleEncoderLayer"]):
    emb_dim: int
    emb_dim_fusion: int
    ffn_dim: int
    ffn_dim_txt: int
    num_heads: int
    num_heads_fusion: int
    num_heads_txt: int
    num_points: int
    num_levels: int
    dropout: float
    attention_dropout: float
    droppath: float
    use_fusion: bool
    use_txt_self_att: bool


class CrossScaleEncoderLayer(nn.Module):
    """Normal Grounding DINO encoder layer.

    1) txt<->img fusion
    2) img self att
    2) txt self att
    3) img ffn
    3) txt ffn
    """

    def __init__(self, cfg: CrossScaleEncoderLayerConfig):
        super().__init__()

        self.fusion = (
            FusionLayer(
                cfg.emb_dim,
                cfg.emb_dim_fusion,
                cfg.num_heads_fusion,
                cfg.attention_dropout,
                cfg.droppath,
            )
            if cfg.use_fusion
            else None
        )

        self.img_self_att = PositionModulatedDeformableAttentionBlock(
            cfg.emb_dim,
            cfg.num_levels,
            cfg.num_heads,
            cfg.num_points,
            cfg.dropout,
        )
        self.img_ffn = FFNBlock(cfg.emb_dim, cfg.ffn_dim, cfg.dropout)

        self.txt_self_att = (
            PositionModulatedSelfAttentionBlock(
                cfg.emb_dim,
                cfg.num_heads_txt,
                cfg.dropout,
                cfg.attention_dropout,
            )
            if cfg.use_txt_self_att
            else None
        )
        self.txt_ffn = FFNBlock(cfg.emb_dim, cfg.ffn_dim_txt, cfg.dropout)

    def forward(
        self, img: ImageFeatures, txt: TextFeatures
    ) -> tuple[ImageFeatures, TextFeatures]:
        # copy so we modify in scope
        img, txt = img.copy(), txt.copy()

        # txt2img and img2txt cross attentions
        if self.fusion is not None:
            img["feat"], txt["feat"] = self.fusion(
                img["feat"],
                img["mask"],
                txt["feat"],
                txt["mask"],
            )

        # text self attention
        # NOTE: key padding mask is None, since self att can produce NaNs on padding elements
        if self.txt_self_att is not None:
            txt["feat"] = self.txt_self_att(
                txt["feat"], txt["pos"], None, txt["att_mask"]
            )
        txt["feat"] = self.txt_ffn(txt["feat"])

        # image self attention
        img["feat"] = self.img_self_att(
            img["feat"],
            img["mask"],
            img["shapes"],
            img["valid_ratios"],
            img["feat"],
            img["pos"],
            img["coor"],
        )
        img["feat"] = self.img_ffn(img["feat"])

        return img, txt


CrossScaleEncoderLayerConfig._target_class = CrossScaleEncoderLayer
