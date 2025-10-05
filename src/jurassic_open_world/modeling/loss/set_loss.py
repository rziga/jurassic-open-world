from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import sigmoid_focal_loss

from .matcher import HungarianMatcherConfig
from ...utils.box import generalized_iou
from ...utils.config import BaseConfig


@dataclass
class SetLossConfig(BaseConfig["SetLoss"]):
    matcher_cfg: HungarianMatcherConfig
    weight_cls: float
    weight_l1: float
    weight_giou: float
    focal_alpha: float
    focal_gamma: float


class SetLoss(nn.Module):
    """One-to-one loss.
    Consists of sigmoid focal loss for class and l1 and giou losses for bounding boxes.
    """

    def __init__(self, cfg: SetLossConfig):
        super().__init__()

        self.matcher = cfg.matcher_cfg.build()

        self.weight_cls = cfg.weight_cls
        self.weight_l1 = cfg.weight_l1
        self.weight_giou = cfg.weight_giou
        self.focal_alpha = cfg.focal_alpha
        self.focal_gamma = cfg.focal_gamma

    def forward(
        self,
        pred_cls: torch.Tensor,
        pred_bbox: torch.Tensor,
        pred_mask: torch.Tensor,
        txt_cap_ids: torch.Tensor,
        target_bbox: torch.Tensor,
        target_cap_ids: torch.Tensor,
        target_mask: torch.Tensor,
        pred_idxs: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        target_idxs: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[float, float, float]]:
        # generate the target label map
        target_label_map = self.generate_label_map(
            txt_cap_ids, target_cap_ids, pred_cls.dtype
        )  # [B, N_target, N_txt]

        # match pred bboxes with target bboxes if they are not already passed
        if pred_idxs is None or target_idxs is None:
            pred_idxs, target_idxs = self.matcher(
                pred_cls,
                pred_bbox,
                pred_mask,
                target_label_map,
                target_bbox,
                target_mask,
            )  # magic, magic, where magic is tuple[batch_idx, el_idx]

        # select matched pred and target bboxes
        pred_bbox = pred_bbox[pred_idxs]  # [B*N_matches, 4]
        target_bbox = target_bbox[target_idxs]  # [B*N_matches, 4]

        # generate target cls
        target_cls = torch.zeros_like(pred_cls)  # [B, N_query, N_txt]
        target_cls[pred_idxs] = target_label_map[
            target_idxs
        ]  # [B, N_query, N_txt]

        # select valid
        # fmt: off
        cls_mask = (
            (txt_cap_ids == -1)[:, None, :]
            |         pred_mask[:, :, None]
        )  # [B, 1, N_txt] | [B, N_query, 1] -> [B, N_query, N_txt], True -> padded
        # fmt: on
        pred_cls = pred_cls.masked_select(~cls_mask)  # [B*N_matches]
        target_cls = target_cls.masked_select(~cls_mask)  # [B*N_matches]

        # calculate loss components
        # fmt: off
        num_boxes = target_bbox.shape[0]
        loss_cls = sigmoid_focal_loss(
            pred_cls, target_cls, self.focal_alpha, self.focal_gamma, reduction="sum"
        ) / num_boxes
        loss_l1 = F.l1_loss(
            pred_bbox, target_bbox, reduction="sum"
        ) / num_boxes
        loss_giou = torch.sum(
            1 - generalized_iou(pred_bbox, target_bbox)
        ) / num_boxes
        # fmt: on

        # calculate loss
        # fmt: off
        loss = (
              self.weight_cls  * loss_cls
            + self.weight_l1   * loss_l1
            + self.weight_giou * loss_giou
        )
        # fmt: on

        return loss, (loss_cls.item(), loss_l1.item(), loss_giou.item())

    def generate_label_map(self, txt_cap_ids, target_cap_ids, dtype):
        txt_cap_ids = txt_cap_ids[:, None, :]  # [B, 1, N_txt]
        target_cap_ids = target_cap_ids[:, :, None]  # [B, N_target, 1]

        label_map = txt_cap_ids == target_cap_ids  # [B, N_target, N_txt]
        label_map &= txt_cap_ids != -1
        label_map &= target_cap_ids != -1
        return label_map.to(dtype)


SetLossConfig._target_class = SetLoss
