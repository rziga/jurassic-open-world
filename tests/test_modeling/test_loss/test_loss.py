# type: ignore
import torch
import pytest

from jurassic_open_world.modeling.loss.set_loss import SetLoss, SetLossConfig, HungarianMatcherConfig
from .original_impl import SetCriterion as gt_GroundingDINOLoss, HungarianMatcher as gt_Matcher


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_forward(device):
    B, N_pred, N_tgt, N_txt = 4, 16, 3, 12 
    pred_cls = torch.rand(B, N_pred, N_txt, device=device)
    pred_bbox = torch.rand(B, N_pred, 4, device=device)
    pred_mask = torch.zeros(B, N_pred, dtype=torch.bool, device=device)
    pred_cap_ids = torch.arange(N_tgt, device=device).repeat_interleave(N_txt//N_tgt+1)[:N_txt].expand(B, N_txt)
    target_bbox = torch.rand(B, N_tgt, 4, device=device)
    target_cap_ids = torch.arange(N_tgt, device=device).expand(B, N_tgt)
    target_mask = torch.zeros(B, N_tgt, dtype=torch.bool, device=device)

    loss_fn = SetLoss(SetLossConfig(HungarianMatcherConfig(1, 1, 1, 0.25, 2), 1, 1, 1, 0.25, 2)).to(device)
    output, *_ = loss_fn.forward(pred_cls, pred_bbox, pred_mask, pred_cap_ids, target_bbox, target_cap_ids, target_mask)
    assert output is not None
    assert not output.isnan() and not output.isinf()


def test_gt():
    B, N_pred, N_tgt, N_txt = 4, 16, 3, 12 
    pred_cls = torch.rand(B, N_pred, N_txt)
    pred_bbox = torch.rand(B, N_pred, 4)
    pred_mask = torch.zeros(B, N_pred, dtype=torch.bool)
    pred_cap_ids = torch.arange(N_tgt).repeat_interleave(N_txt//N_tgt+1)[:N_txt].expand(B, N_txt)
    target_bbox = torch.rand(B, N_tgt, 4)
    target_cap_ids = torch.arange(N_tgt).expand(B, N_tgt)
    target_mask = torch.zeros(B, N_tgt, dtype=torch.bool)

    gt_matcher = gt_Matcher(1, 1, 1, 0.25)
    gt_loss_fn = gt_GroundingDINOLoss(gt_matcher, {}, 0.25, 2, ["labels", "boxes"])
    loss_fn = SetLoss(SetLossConfig(HungarianMatcherConfig(1, 1, 1, 0.25, 2), 1, 1, 1, 0.25, 2))

    gt_out = gt_loss_fn.forward(
        {
            "pred_logits": pred_cls,
            "pred_boxes": pred_bbox,
            "text_mask": torch.ones(B, N_txt, dtype=torch.bool),
        },
        [
            {
                "boxes": b,
                "labels": torch.arange(len(b)),
            }
            for b in target_bbox
        ],
        pred_cap_ids,
        target_cap_ids,
    )
    gt_loss = gt_out["loss_ce"] + gt_out["loss_bbox"] + gt_out["loss_giou"]

    loss, (cls, l1, giou) = loss_fn.forward(pred_cls, pred_bbox, pred_mask, pred_cap_ids, target_bbox, target_cap_ids, target_mask)

    assert torch.allclose(gt_loss, loss, atol=1e-6)
    assert torch.allclose(gt_out["loss_ce"], torch.tensor(cls, device=loss.device), atol=1e-6)
    assert torch.allclose(gt_out["loss_bbox"], torch.tensor(l1, device=loss.device), atol=1e-6)
    assert torch.allclose(gt_out["loss_giou"], torch.tensor(giou, device=loss.device), atol=1e-6)


def test_gt_mask():
    B, N_pred, N_tgt, N_txt = 3, 16, 3, 12 
    pred_cls = torch.rand(B, N_pred, N_txt)
    pred_bbox = torch.rand(B, N_pred, 4)
    pred_mask = torch.zeros(B, N_pred, dtype=torch.bool)
    pred_cap_ids = torch.arange(N_tgt).repeat_interleave(N_txt//N_tgt+1)[:N_txt].expand(B, N_txt).clone()
    pred_cap_ids[:, -2:] = -1

    target_bbox = torch.rand(B, N_tgt, 4)
    target_cap_ids = torch.arange(N_tgt).expand(B, N_tgt).clone()
    target_cap_ids[:, -1:] = -1

    text_mask = pred_cap_ids != -1
    target_mask = target_cap_ids == -1
    target_bbox[target_mask] = 0

    gt_matcher = gt_Matcher(1, 1, 1, 0.25)
    gt_loss_fn = gt_GroundingDINOLoss(gt_matcher, {}, 0.25, 2, ["labels", "boxes"])
    loss_fn = SetLoss(SetLossConfig(HungarianMatcherConfig(1, 1, 1, 0.25, 2), 1, 1, 1, 0.25, 2))

    gt_out = gt_loss_fn.forward(
        {
            "pred_logits": pred_cls,
            "pred_boxes": pred_bbox,
            "text_mask": text_mask,
        },
        [
            {
                "boxes": b[m],
                "labels": l[m],
            }
            for b, l, m in zip(target_bbox, target_cap_ids, ~target_mask)  # noqa: E741
        ],
        pred_cap_ids,
        target_cap_ids,
    )
    gt_loss = gt_out["loss_ce"] + gt_out["loss_bbox"] + gt_out["loss_giou"]

    loss, (cls, l1, giou) = loss_fn.forward(pred_cls, pred_bbox, pred_mask, pred_cap_ids, target_bbox, target_cap_ids, target_mask)

    assert torch.allclose(gt_loss, loss, atol=1e-6)
    assert torch.allclose(gt_out["loss_ce"], torch.tensor(cls, device=loss.device), atol=1e-6)
    assert torch.allclose(gt_out["loss_bbox"], torch.tensor(l1, device=loss.device), atol=1e-6)
    assert torch.allclose(gt_out["loss_giou"], torch.tensor(giou, device=loss.device), atol=1e-6)
