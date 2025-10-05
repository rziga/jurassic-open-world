from dataclasses import dataclass

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from ...utils.box import generalized_iou
from ...utils.config import BaseConfig


@dataclass
class HungarianMatcherConfig(BaseConfig["HungarianMatcher"]):
    coef_cls: float
    coef_l1: float
    coef_giou: float
    focal_alpha: float
    focal_gamma: float
    eps: float = 1e-5


class HungarianMatcher(nn.Module):
    """Hungarian matcher.
    One-to-one matcher between targets and model outputs.
    """

    def __init__(self, cfg: HungarianMatcherConfig):
        super().__init__()
        self.coef_cls = cfg.coef_cls
        self.coef_l1 = cfg.coef_l1
        self.coef_giou = cfg.coef_giou

        self.focal_alpha = cfg.focal_alpha
        self.focal_gamma = cfg.focal_gamma

        self.eps = cfg.eps

    def _get_cls_cost_matrix(self, pred_cls, target_label_map):
        # fmt: off
        # normalize label map
        target_label_map = target_label_map / (target_label_map.sum(axis=-1, keepdim=True) + self.eps) # [B, N_target, N_txt]

        # generate focal prob matrices
        p = pred_cls.sigmoid() # [B, N_query, N_txt]
        cost = (
              (1 - self.focal_alpha) * (p) ** self.focal_gamma * torch.log(1 - p + self.eps)
            - (self.focal_alpha) * (1 - p) ** self.focal_gamma * torch.log(p + self.eps)
        )
        # fmt: on
        return cost @ target_label_map.mT  # [B, N_query, N_target]

    def _get_l1_cost_matrix(self, pred_bbox, target_bbox):
        return torch.cdist(
            pred_bbox, target_bbox, p=1
        )  # [B, N_query, N_target]

    def _get_giou_cost_matrix(self, pred_bbox, target_bbox):
        # fmt: off
        return -generalized_iou(
              pred_bbox[:, :, None, :], # [B, N_query, 1, 4]
            target_bbox[:, None, :, :], # [B, 1, N_target, 4]
        ) # [B, N_query, N_target]
        # fmt: on

    @torch.no_grad()
    def forward(
        self,
        pred_cls: torch.Tensor,
        pred_bbox: torch.Tensor,
        pred_mask: torch.Tensor,
        target_cls: torch.Tensor,
        target_bbox: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
    ]:
        device = pred_cls.device

        # construct cost matrix
        cost_cls = self._get_cls_cost_matrix(
            pred_cls, target_cls
        )  # [B, N_query, N_target]
        cost_l1 = self._get_l1_cost_matrix(
            pred_bbox, target_bbox
        )  # [B, N_query, N_target]
        cost_giou = self._get_giou_cost_matrix(
            pred_bbox, target_bbox
        )  # [B, N_query, N_target]
        cost = (
            self.coef_cls * cost_cls
            + self.coef_l1 * cost_l1
            + self.coef_giou * cost_giou
        )  # [B, N_query, N_target]

        # mask out the padded targets
        # NOTE: fill with 1_000_000 because scipy throws error if inf is present
        cost.masked_fill_(pred_mask[:, :, None], 1_000_000)
        cost.masked_fill_(target_mask[:, None, :], 1_000_000)

        # run the linear sum assignment optimization
        pred_idxs, target_idxs = [], []
        pred_batch_idxs, target_batch_idxs = [], []
        for i, cost_mtx in enumerate(cost.cpu()):
            pred_idx, target_idx = linear_sum_assignment(cost_mtx)

            pred_idx = torch.tensor(pred_idx, device=device)  # [N_matches]
            target_idx = torch.tensor(target_idx, device=device)  # [N_matches]

            # check if any padding got matched
            # fmt: off
            mask_match = (
                    pred_mask[i, pred_idx]
                | target_mask[i, target_idx]
            )  # [N_matches]
            # fmt: on
            pred_idx = pred_idx[~mask_match]
            target_idx = target_idx[~mask_match]

            pred_idxs.append(pred_idx)
            target_idxs.append(target_idx)
            pred_batch_idxs.append(torch.full_like(pred_idx, i))
            target_batch_idxs.append(torch.full_like(target_idx, i))

        # ([B*N_matches], [B*N_matches])
        return (
            (torch.cat(pred_batch_idxs), torch.cat(pred_idxs)),
            (torch.cat(target_batch_idxs), torch.cat(target_idxs)),
        )


HungarianMatcherConfig._target_class = HungarianMatcher
