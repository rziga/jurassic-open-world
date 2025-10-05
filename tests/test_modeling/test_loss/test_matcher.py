import torch
import pytest

from jurassic_open_world.modeling.loss.matcher import HungarianMatcher, HungarianMatcherConfig
from .original_impl import HungarianMatcher as gt_HungarianMatcher


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_forward(device):
    B, N_pred, N_tgt, N_txt = 4, 16, 3, 12 
    pred_cls = torch.rand(B, N_pred, N_txt, device=device)
    pred_bbox = torch.rand(B, N_pred, 4, device=device)
    pred_mask = torch.zeros(B, N_pred, dtype=torch.bool, device=device)
    target_label_map = torch.eye(N_tgt, N_txt, device=device).expand(B, N_tgt, N_txt)
    target_bbox = torch.rand(B, N_tgt, 4, device=device)
    target_mask = torch.zeros(B, N_tgt, dtype=torch.bool, device=device)

    matcher = HungarianMatcherConfig(1, 1, 1, 0.25, 2).build()
    outputs = matcher(pred_cls, pred_bbox, pred_mask, target_label_map, target_bbox, target_mask)
    assert outputs is not None


def test_gt():
    B, N_pred, N_tgt, N_txt = 3, 16, 3, 12 
    pred_cls = torch.rand(B, N_pred, N_txt)
    pred_bbox = torch.rand(B, N_pred, 4)
    pred_mask = torch.zeros(B, N_pred, dtype=torch.bool)
    target_label_map = torch.eye(N_tgt, N_txt).expand(B, N_tgt, N_txt)
    target_bbox = torch.rand(B, N_tgt, 4)
    target_mask = torch.zeros(B, N_tgt, dtype=torch.bool)

    gt_matcher = gt_HungarianMatcher()
    matcher = HungarianMatcher(HungarianMatcherConfig(1, 1, 1, 0.25, 2))

    # run gt and get the same output format
    gt_pred_idxs, gt_tgt_idxs = [], []
    for i in range(B):
        idxs = gt_matcher.forward(
            {
                "pred_logits": pred_cls[[i]],
                "pred_boxes": pred_bbox[[i]],
            },
            [{
                "boxes": target_bbox[i],
                "labels": torch.arange(N_tgt)
            }],
            target_label_map[i]
        )
        gt_pred_idxs.append(idxs[0][0])
        gt_tgt_idxs.append(idxs[0][1])

    pred_idxs, tgt_idxs = matcher.forward(pred_cls, pred_bbox, pred_mask, target_label_map, target_bbox, target_mask)

    for i in range(B):
        pred_idxs_b = pred_idxs[1][pred_idxs[0] == i]
        tgt_idxs_b = tgt_idxs[1][tgt_idxs[0] == i]

        assert torch.allclose(gt_pred_idxs[i], pred_idxs_b)
        assert torch.allclose(gt_tgt_idxs[i], tgt_idxs_b)

