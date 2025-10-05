from dataclasses import dataclass
from typing import Literal

import torch
from torchvision.transforms import v2

from .custom_transforms import NormalizeBoundingBoxes
from ..utils.config import BaseConfig


def _get_norm(norm_mean, norm_std):
    return v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(norm_mean, norm_std),
            v2.ClampBoundingBoxes(),
            v2.SanitizeBoundingBoxes(labels_getter=lambda tup: tup[2]),
            v2.ConvertBoundingBoxFormat("CXCYWH"),
            NormalizeBoundingBoxes(),
        ]
    )


@dataclass
class COCOTransformConfig(BaseConfig["COCOTransform"]):
    img_mean: tuple[float, float, float]
    img_std: tuple[float, float, float]
    max_size: int
    resize_sizes: tuple[int, ...]
    crop_sizes: tuple[int, ...]
    mode: Literal["train", "test"]


class COCOTransform(v2.Compose):
    def __init__(self, cfg: COCOTransformConfig):
        if cfg.mode == "train":
            transforms = self._get_train_transforms(cfg)
        elif cfg.mode == "test":
            transforms = self._get_test_transforms(cfg)
        else:
            raise NotImplementedError(f"{cfg.mode=} not supported")
        super().__init__(transforms)

    def _get_train_transforms(
        self, cfg: COCOTransformConfig
    ) -> list[v2.Transform]:
        # fmt: off
        return [
            v2.RandomHorizontalFlip(),
            v2.RandomChoice([
                v2.RandomShortestSize(
                    list(cfg.resize_sizes), max_size=cfg.max_size
                ),
                v2.Compose([
                    v2.RandomChoice([
                        v2.RandomCrop(crop_size, pad_if_needed=True)
                        for crop_size in cfg.crop_sizes
                    ]),
                    v2.RandomShortestSize(
                        list(cfg.resize_sizes), max_size=cfg.max_size
                    ),
                ])
            ]),
            _get_norm(cfg.img_mean, cfg.img_std),
        ]
        # fmt: on

    def _get_test_transforms(
        self, cfg: COCOTransformConfig
    ) -> list[v2.Transform]:
        return [
            v2.Resize(max(cfg.resize_sizes), max_size=cfg.max_size),
            _get_norm(cfg.img_mean, cfg.img_std),
        ]


COCOTransformConfig._target_class = COCOTransform


@dataclass
class SimpleTransformConfig(BaseConfig["SimpleTransform"]):
    img_mean: tuple[float, float, float]
    img_std: tuple[float, float, float]
    max_size: int
    mode: Literal["train", "test"]


class SimpleTransform(v2.Compose):
    def __init__(self, cfg: SimpleTransformConfig):
        super().__init__(
            [
                v2.Resize(size=None, max_size=cfg.max_size),
                _get_norm(cfg.img_mean, cfg.img_std),
            ]
        )


SimpleTransformConfig._target_class = SimpleTransform
