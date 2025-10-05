from typing import Iterator

import torch

from .types import ImageFeatures, ImageLevelFeatures


def merge_fpn_levels(levels: list[ImageLevelFeatures]) -> ImageFeatures:
    device = levels[0]["feat"].device
    return {
        "feat": torch.cat(
            [level["feat"].flatten(-2).mT for level in levels], dim=1
        ),
        "pos": torch.cat(
            [level["pos"].flatten(-2).mT for level in levels], dim=1
        ),
        "mask": torch.cat(
            [level["mask"].flatten(-2) for level in levels], dim=1
        ),
        "coor": torch.cat(
            [level["coor"].flatten(-2).mT for level in levels], dim=1
        ),
        "shapes": torch.tensor(
            [level["feat"].shape[-2:] for level in levels], device=device
        ),
        "valid_ratios": torch.stack(
            [level["valid_ratios"] for level in levels], dim=1
        ),
    }


def split_fpn_levels(pyramid: ImageFeatures) -> list[ImageLevelFeatures]:
    return [
        {
            "feat": feat,
            "pos": pos,
            "mask": mask,
            "coor": coor,
            "valid_ratios": valid,
        }
        for (feat, pos, mask, coor, valid) in zip(
            iter_fpn_levels(pyramid["feat"], pyramid["shapes"]),
            iter_fpn_levels(pyramid["pos"], pyramid["shapes"]),
            iter_fpn_levels(pyramid["mask"], pyramid["shapes"]),
            iter_fpn_levels(pyramid["coor"], pyramid["shapes"]),
            pyramid["valid_ratios"].unbind(dim=1),
        )
    ]


def iter_fpn_levels(
    flat_data: torch.Tensor, hw_shapes: torch.Tensor
) -> Iterator[torch.Tensor]:
    level_sizes = hw_shapes.prod(dim=1).cumsum(dim=0).tolist()
    for (H, W), start, stop in zip(
        hw_shapes.tolist(), [0] + level_sizes, level_sizes
    ):
        if flat_data.ndim == 3:
            B, _, C = flat_data.shape
            yield flat_data[:, start:stop].mT.reshape(B, C, H, W)
        if flat_data.ndim == 2:
            B, _ = flat_data.shape
            yield flat_data[:, start:stop].reshape(B, H, W)


def generate_xy_coordinates(
    masks: torch.Tensor, center_offset: float = 0.5
) -> torch.Tensor:
    B, H, W = masks.shape

    # generate xy grid
    xy = torch.stack(
        torch.meshgrid(
            torch.arange(0, W, device=masks.device),
            torch.arange(0, H, device=masks.device),
            indexing="xy",
        ),
        dim=0,
    )  # [2, H, W]

    # offset so centers are in the middle of pixels
    xy = xy + center_offset

    # get valid ratios
    wh = torch.tensor([W, H], device=masks.device)
    wh_valid = get_valid_ratios(masks).flip(1) * wh  # [B, 2]

    # normalize
    xy = (
        xy / wh_valid[:, :, None, None]
    )  # [2, H, W] / [B, 2, 1, 1] -> [B, 2, H, W]

    return xy


def get_valid_ratios(mask: torch.Tensor) -> torch.Tensor:
    _, H, W = mask.shape
    mask = ~mask
    return torch.stack(
        [
            mask[:, :, 0].sum(dim=1) / H,
            mask[:, 0, :].sum(dim=1) / W,
        ],
        dim=-1,
    )  # [B, 2]
