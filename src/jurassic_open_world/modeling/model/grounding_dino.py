from dataclasses import dataclass
from pathlib import Path
from typing import Union

import huggingface_hub
import torch
from safetensors.torch import load_model, save_model
from torch import nn

from .decoder.decoder import DecoderConfig
from .encoder.cross_scale.encoder import CrossScaleEncoderConfig
from .encoder.efficient.encoder import EfficientEncoderConfig
from .image_backbone.image_backbone import ImageBackboneConfig
from .query_selector.query_selector import QuerySelectorConfig
from .text_backbone.legacy_text_backbone import LegacyTextBackboneConfig
from .text_backbone.text_backbone import TextBackboneConfig
from ...utils.config import BaseConfig
from ...utils.types import ModelOutput


@dataclass
class GroundingDINOConfig(BaseConfig["GroundingDINO"]):
    img_backbone_cfg: ImageBackboneConfig
    txt_backbone_cfg: Union[TextBackboneConfig, LegacyTextBackboneConfig]
    encoder_cfg: Union[CrossScaleEncoderConfig, EfficientEncoderConfig]
    query_selector_cfg: QuerySelectorConfig
    decoder_cfg: DecoderConfig
    img_mean: tuple[float, float, float]
    img_std: tuple[float, float, float]


class GroundingDINO(nn.Module):
    def __init__(self, cfg: GroundingDINOConfig):
        super().__init__()
        self.cfg = cfg
        self.img_backbone = cfg.img_backbone_cfg.build()
        self.txt_backbone = cfg.txt_backbone_cfg.build()
        self.encoder = cfg.encoder_cfg.build()
        self.query_selector = cfg.query_selector_cfg.build()
        self.decoder = cfg.decoder_cfg.build()

    def forward(
        self,
        img: torch.Tensor,
        img_mask: torch.Tensor,
        captions: list[list[str]],
    ) -> ModelOutput:
        img_feat = self.img_backbone(img, img_mask)
        txt_feat = self.txt_backbone(captions)
        fused_img_feat, fused_txt_feat = self.encoder(img_feat, txt_feat)
        enc_output, query_feat = self.query_selector(
            fused_img_feat, fused_txt_feat
        )
        dec_outputs = self.decoder(query_feat, fused_img_feat, fused_txt_feat)

        return {
            "outputs": dec_outputs[-1],
            "decoder_outputs": dec_outputs,
            "encoder_output": enc_output,
            "img_features": fused_img_feat,
            "txt_features": fused_txt_feat,
            "query_features": query_feat,
        }

    def image_backbone_parameters(self):
        return self.img_backbone.backbone_parameters()

    def text_backbone_parameters(self):
        return self.txt_backbone.backbone_parameters()

    def non_backbone_parameters(self):
        backbone_param_ids = {
            id(p)
            for p in list(self.image_backbone_parameters())
            + list(self.text_backbone_parameters())
        }
        return (
            p for p in self.parameters() if id(p) not in backbone_param_ids
        )

    def freeze_image_backbone(self):
        self.img_backbone.freeze_backbone()

    def freeze_text_backbone(self):
        self.txt_backbone.freeze_backbone()

    def save_pretrained(self, save_dir: str | Path):
        if not isinstance(save_dir, Path):
            save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

        config_path = save_dir / "config.yaml"
        self.cfg.to_yaml(config_path)

        model_path = save_dir / "model.safetensors"
        save_model(self, str(model_path))

        readme_path = save_dir / "README.md"
        with open(readme_path, "w") as f:
            f.write(
                r"Model created using [Jurassic Open World](https://github.com/rziga/jurassic-open-world)."
            )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | Path
    ) -> "GroundingDINO":
        if (dir := Path(pretrained_model_name_or_path)).exists():
            config_path = dir / "config.yaml"
            model_path = dir / "model.safetensors"
        else:
            config_path = huggingface_hub.hf_hub_download(
                str(pretrained_model_name_or_path), "config.yaml"
            )
            model_path = huggingface_hub.hf_hub_download(
                str(pretrained_model_name_or_path), "model.safetensors"
            )

        config = GroundingDINOConfig.from_yaml(config_path)
        model = GroundingDINO(config)
        load_model(model, model_path)
        return model


GroundingDINOConfig._target_class = GroundingDINO
