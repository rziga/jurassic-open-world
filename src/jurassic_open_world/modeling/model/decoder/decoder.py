from dataclasses import dataclass

from torch import nn

from .decoder_layer import DecoderLayerConfig
from ....utils.config import BaseConfig
from ....utils.types import (
    ImageFeatures,
    OutputFeatures,
    QueryFeatures,
    TextFeatures,
)


@dataclass
class DecoderConfig(BaseConfig["Decoder"]):
    layer_cfg: DecoderLayerConfig
    num_layers: int
    share_cls_head: bool
    share_bbox_head: bool
    share_norm: bool
    share_pos_head: bool


class Decoder(nn.Module):
    """Stack of decoder layers."""

    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [cfg.layer_cfg.build() for _ in range(cfg.num_layers)]
        )

        if cfg.share_cls_head:
            for layer in self.layers:
                layer.cls_embed = self.layers[0].cls_embed

        if cfg.share_bbox_head:
            for layer in self.layers:
                layer.bbox_update_embed = self.layers[0].bbox_update_embed

        if cfg.share_norm:
            for layer in self.layers:
                layer.norm = self.layers[0].norm

        if cfg.share_pos_head:
            for layer in self.layers:
                layer.bbox_pos_embed = self.layers[0].bbox_pos_embed

    def forward(
        self, query: QueryFeatures, img: ImageFeatures, txt: TextFeatures
    ) -> list[OutputFeatures]:
        # run the decoder layers with look forward twice
        outputs = []
        for layer in self.layers:
            output, query = layer.forward(query, img, txt)
            outputs.append(output)
        return outputs


DecoderConfig._target_class = Decoder
