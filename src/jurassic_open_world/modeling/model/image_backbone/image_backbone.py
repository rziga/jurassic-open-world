from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from .feature_extractor import get_feature_extractor
from ..common_blocks.position_embedding import SineEmbedding
from ....utils.config import BaseConfig
from ....utils.image import generate_xy_coordinates, get_valid_ratios
from ....utils.types import ImageFeatures


@dataclass
class ImageBackboneConfig(BaseConfig["ImageBackbone"]):
    provider: Literal["transformers_auto", "transformers_clip", "timm"]
    cfg_str: str
    use_pretrained: bool
    backbone_kwargs: dict
    emb_dim: int
    num_extra_up_projs: int
    num_extra_down_projs: int
    use_legacy_pos: bool


class ImageBackbone(nn.Module):
    def __init__(self, cfg: ImageBackboneConfig):
        super().__init__()
        self.cfg = cfg

        # load feature extractor
        self.model = get_feature_extractor(
            cfg.provider, cfg.cfg_str, cfg.use_pretrained, cfg.backbone_kwargs
        )

        # input projection layers
        input_dims = self.model.get_channels()
        # fmt: off
        self.input_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_dim, cfg.emb_dim, 1),
                nn.GroupNorm(32, cfg.emb_dim),
            )
            for input_dim in input_dims
        ])

        # extra upsample projection layers
        if cfg.num_extra_up_projs > 0:
            extra_up_dims = [input_dims[0]] + [cfg.emb_dim] * (cfg.num_extra_up_projs-1)
        else:
            extra_up_dims = []
        self.extra_input_up_projs = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(dim, cfg.emb_dim, 3, 1, 1),
                nn.GroupNorm(32, cfg.emb_dim),
            )
            for dim in extra_up_dims
        ])

        # extra downsample projection layers
        if cfg.num_extra_down_projs:
            extra_down_dims = [input_dims[-1]] + [cfg.emb_dim] * (cfg.num_extra_down_projs-1)
        else:
            extra_down_dims = []
        self.extra_input_down_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, cfg.emb_dim, 3, 2, 1),
                nn.GroupNorm(32, cfg.emb_dim),
            )
            for dim in extra_down_dims
        ])
        # fmt: on

        # postion and level encoding
        self.pos_emb = SineEmbedding(cfg.emb_dim, temperature=20)
        num_levels = (
            len(input_dims) + cfg.num_extra_down_projs + cfg.num_extra_up_projs
        )
        self.level_emb = nn.Parameter(torch.randn(num_levels, cfg.emb_dim))

    def forward(
        self, img: torch.Tensor, img_mask: torch.Tensor
    ) -> ImageFeatures:
        # get backbone features
        backbone_features = self.model(
            img
        )  # List[B, C_i, H_i, W_i], len(List) - num levels of backbone

        # project them to emb_dim
        features = [
            proj(feat)
            for proj, feat in zip(self.input_projs, backbone_features)
        ]  # List[B, C, H_i, W_i], len(List) - num levels of backbone

        # upsample for additional feature levels
        prev = backbone_features[0]
        for proj in self.extra_input_up_projs:
            prev = proj(prev)
            features = [prev] + features

        # downsample for additonal feature levels
        prev = backbone_features[-1]
        for proj in self.extra_input_down_projs:
            prev = proj(prev)
            features = features + [prev]
        # features - List[B, C, H_i, W_i], len(List) - num levels

        # downsample the input mask
        masks = [
            F.interpolate(
                img_mask.unsqueeze(1).float(),
                feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(1)
            .bool()
            for feature in features
        ]  # List[B, H_i, W_i], len(List) - num levels

        # generate xy coordinate grids
        # NOTE: for some reason the original has centers different centers
        # for coordinates and embeddings; I am sad because it is a bit more cluttered
        coor_grids = [
            generate_xy_coordinates(mask, center_offset=0.5) for mask in masks
        ]  # List[B, 2, H_i, W_i], len(List) - num levels
        coor_grids_emb = [
            generate_xy_coordinates(
                mask, center_offset=(1.0 if self.cfg.use_legacy_pos else 0.5)
            )
            for mask in masks
        ]  # List[B, 2, H_i, W_i], len(List) - num levels

        # save shapes and valid ratios
        hw_shapes = torch.tensor(
            [f.shape[-2:] for f in features], device=img.device
        )  # [L, 2]
        hw_valid_ratios = torch.stack(
            [get_valid_ratios(m) for m in masks], dim=1
        )  # [B, L, 2]

        # flatten features, masks and grids
        features = self._flat_concat(features).mT  # [B, N_img, C]
        masks = self._flat_concat(masks)  # [B, N_img]
        coor_grids = self._flat_concat(coor_grids).mT  # [B, N_img, 2]
        coor_grids_emb = self._flat_concat(coor_grids_emb).mT  # [B, N_img, 2]

        # generate position embeddings and add level embeddings to them
        level_sizes = hw_shapes.prod(dim=1)  # [L]
        level_embeds = self.level_emb.repeat_interleave(
            level_sizes, dim=0
        )  # [N_img, C]
        pos_embeds = self.pos_emb(
            coor_grids_emb.flip(-1)
            if self.cfg.use_legacy_pos
            else coor_grids_emb
        )  # [B, N_img, C], NOTE: flip to swap xy so it matches original
        pos_level_embeds = pos_embeds + level_embeds

        return {
            "feat": features,
            "pos": pos_level_embeds,
            "mask": masks,
            "coor": coor_grids,
            "shapes": hw_shapes,
            "valid_ratios": hw_valid_ratios,
        }

    @staticmethod
    def _flat_concat(xs):
        return torch.cat([x.flatten(-2) for x in xs], dim=-1)

    def backbone_parameters(self):
        return self.model.backbone_parameters()

    def non_backbone_parameters(self):
        backbone_param_ids = {id(p) for p in self.backbone_parameters()}
        return (
            p for p in self.parameters() if id(p) not in backbone_param_ids
        )

    def freeze_backbone(self):
        self.model.freeze_backbone()


ImageBackboneConfig._target_class = ImageBackbone
