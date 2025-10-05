import torch
from torchvision.ops import box_convert

from .original_impl import generalized_box_iou as gt_generalized_box_iou
from jurassic_open_world.utils.box import generalized_iou


def test_giou_gt():
    bboxes1 = torch.rand(12, 4)
    bboxes2 = torch.rand(7, 4)

    a = gt_generalized_box_iou(
        box_convert(bboxes1, "cxcywh", "xyxy"),
        box_convert(bboxes2, "cxcywh", "xyxy"),
    )
    b = generalized_iou(
        bboxes1[:, None, :],
        bboxes2[None, :, :]
    )
    assert torch.allclose(a, b, atol=1e-6)
