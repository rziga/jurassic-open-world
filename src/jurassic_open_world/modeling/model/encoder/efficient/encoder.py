from dataclasses import dataclass

from torch import nn

from .cross_scale_fuser import CrosScaleFuser
from .encoder_layer import EfficientEncoderLayerConfig
from .....utils.config import BaseConfig
from .....utils.image import merge_fpn_levels, split_fpn_levels
from .....utils.types import ImageFeatures, TextFeatures


@dataclass
class EfficientEncoderConfig(BaseConfig["EfficientEncoder"]):
    layer_cfg: EfficientEncoderLayerConfig
    num_layers: int
    emb_dim: int
    num_levels: int


class EfficientEncoder(nn.Module):
    """
    Efficient encoder.

    Essentially a RT-DETR encoder with added img->txt and txt->img fusion.

    Based on Grounding DINO 1.5 edge technical report.
    """

    def __init__(self, cfg: EfficientEncoderConfig):
        super().__init__()

        self.layers = nn.ModuleList(
            [cfg.layer_cfg.build() for _ in range(cfg.num_layers)]
        )

        self.cross_scale_fuser = CrosScaleFuser(cfg.emb_dim, cfg.num_levels)

    def forward(
        self, img_feat: ImageFeatures, txt_feat: TextFeatures
    ) -> tuple[ImageFeatures, TextFeatures]:
        # split fpn levels
        img_fpn = split_fpn_levels(img_feat)

        # send last feature level through encoder
        for layer in self.layers:
            img_fpn[-1], txt_feat = layer(img_fpn[-1], txt_feat)

        # extract image features, fuse them and update them
        level_feats = [level["feat"] for level in img_fpn]
        level_feats = self.cross_scale_fuser(level_feats)
        for level, feat in zip(img_fpn, level_feats):
            level["feat"] = feat

        # flatten back image features
        img_feat = merge_fpn_levels(img_fpn)

        return img_feat, txt_feat


EfficientEncoderConfig._target_class = EfficientEncoder
