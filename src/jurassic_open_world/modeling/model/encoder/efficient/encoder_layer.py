from dataclasses import dataclass

from torch import nn

from ...common_blocks.fusion_layer import FusionLayer
from ...common_blocks.skip_blocks import (
    FFNBlock,
    PositionModulatedSelfAttentionBlock,
)
from .....utils.config import BaseConfig
from .....utils.types import ImageLevelFeatures, TextFeatures


@dataclass
class EfficientEncoderLayerConfig(BaseConfig["EfficientEncoderLayer"]):
    emb_dim: int
    emb_dim_fusion: int
    ffn_dim: int
    num_heads: int
    num_heads_fusion: int
    dropout: float
    attention_dropout: float
    droppath: float
    use_fusion: bool


class EfficientEncoderLayer(nn.Module):
    """Efficient encoder layer.

    Essentially a RT detr encoder layer with added img<->txt fusion from Grounding DINO.
    """

    def __init__(self, cfg: EfficientEncoderLayerConfig):
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

        self.img_self_att = PositionModulatedSelfAttentionBlock(
            cfg.emb_dim, cfg.num_heads, cfg.dropout, cfg.attention_dropout
        )
        self.img_ffn = FFNBlock(cfg.emb_dim, cfg.ffn_dim, cfg.dropout)

    def forward(
        self, img: ImageLevelFeatures, txt: TextFeatures
    ) -> tuple[ImageLevelFeatures, TextFeatures]:
        # flatten image features, extract txt features
        B, _, H, W = img["feat"].shape
        img_feat = img["feat"].flatten(-2).mT
        img_pos = img["pos"].flatten(-2).mT
        mask_pos = img["mask"].flatten(-2)
        txt_feat = txt["feat"]

        # img2text and text2img fusion with cross attention
        if self.fusion is not None:
            img_feat, txt_feat = self.fusion(
                img_feat, mask_pos, txt_feat, txt["mask"]
            )

        # img self attention
        img_feat = self.img_self_att(img_feat, img_pos, mask_pos, None)
        img_feat = self.img_ffn(img_feat)

        # unflatten image features
        img_feat = img_feat.mT.reshape(B, -1, H, W)

        # pack the outputs again
        img, txt = img.copy(), txt.copy()
        img["feat"] = img_feat
        txt["feat"] = txt_feat
        return img, txt


EfficientEncoderLayerConfig._target_class = EfficientEncoderLayer
