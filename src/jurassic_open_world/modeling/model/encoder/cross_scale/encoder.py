from dataclasses import dataclass

from torch import nn

from .encoder_layer import CrossScaleEncoderLayerConfig
from .....utils.config import BaseConfig
from .....utils.types import ImageFeatures, TextFeatures


@dataclass
class CrossScaleEncoderConfig(BaseConfig["CrossScaleEncoder"]):
    layer_cfg: CrossScaleEncoderLayerConfig
    num_layers: int


class CrossScaleEncoder(nn.Module):
    """Stack of encoder layers."""

    def __init__(self, cfg: CrossScaleEncoderConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [cfg.layer_cfg.build() for _ in range(cfg.num_layers)]
        )

    def forward(
        self, img_feat: ImageFeatures, txt_feat: TextFeatures
    ) -> tuple[ImageFeatures, TextFeatures]:
        for layer in self.layers:
            img_feat, txt_feat = layer(img_feat, txt_feat)
        return img_feat, txt_feat


CrossScaleEncoderConfig._target_class = CrossScaleEncoder
