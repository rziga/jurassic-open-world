import torch
from torchvision.ops import box_convert


# Similar to torhvision box ops, but functions here are in cxcywh
# and independent of batch dims so you can easily do outer operations and stuff


def box_area(bbox_cxcywh: torch.Tensor) -> torch.Tensor:
    return bbox_cxcywh[..., 2] * bbox_cxcywh[..., 3]


def box_intersection_union(
    bbox1_cxcywh: torch.Tensor,
    bbox2_cxcywh: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # calculate intersection area
    # TODO: there is probably a way to do this without format conversion
    xyxy_bbox1 = box_convert(bbox1_cxcywh, "cxcywh", "xyxy")
    xyxy_bbox2 = box_convert(bbox2_cxcywh, "cxcywh", "xyxy")
    lt = torch.max(xyxy_bbox1[..., :2], xyxy_bbox2[..., :2])  # [..., 2]
    rb = torch.min(xyxy_bbox1[..., 2:], xyxy_bbox2[..., 2:])  # [..., 2]
    wh = (rb - lt).clamp(min=0)  # [...,2]
    intersection_area = wh[..., 0] * wh[..., 1]  # [...]

    # calculate union area
    area1 = box_area(bbox1_cxcywh)
    area2 = box_area(bbox2_cxcywh)
    union_area = area1 + area2 - intersection_area

    return intersection_area, union_area


def generalized_iou(
    bbox1_cxcywh: torch.Tensor, bbox2_cxcywh: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    intersection, union = box_intersection_union(bbox1_cxcywh, bbox2_cxcywh)
    iou = intersection / (union + eps)

    xyxy_bbox1 = box_convert(bbox1_cxcywh, "cxcywh", "xyxy")
    xyxy_bbox2 = box_convert(bbox2_cxcywh, "cxcywh", "xyxy")
    lt = torch.min(xyxy_bbox1[..., :2], xyxy_bbox2[..., :2])  # [..., 2]
    rb = torch.max(xyxy_bbox1[..., 2:], xyxy_bbox2[..., 2:])  # [..., 2]
    wh = (rb - lt).clamp(min=0)  # [...,2]
    total_area = wh[..., 0] * wh[..., 1]  # [...]

    return iou - (total_area - union) / (total_area + eps)
