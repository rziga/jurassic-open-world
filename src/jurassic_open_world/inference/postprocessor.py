from copy import deepcopy
from typing import Optional

import torch
from torch import nn
from torchvision.ops import box_convert

from ..utils.text import get_output_caption_probs
from ..utils.types import OutputFeatures


def process_outputs(
    outputs: list[dict[str, torch.Tensor]],
    hws: Optional[torch.Tensor | list[tuple[int, int]]],
    label_map: Optional[dict[int, int]],
    box_format: Optional[str],
) -> list[dict[str, torch.Tensor]]:
    processed = deepcopy(outputs)
    for i, out in enumerate(processed):
        if label_map is not None:
            labels = out["labels"]
            out["labels"] = torch.tensor(
                [label_map[int(lab.item())] for lab in labels],
                device=labels.device,
                dtype=labels.dtype,
            )
        if hws is not None:
            h, w = hws[i]
            out["boxes"][..., 0::2] *= w
            out["boxes"][..., 1::2] *= h
        if box_format is not None:
            out["boxes"] = box_convert(out["boxes"], "cxcywh", box_format)

    return processed


class PostProcessorInclusive(nn.Module):
    """
    Postprocessor for calculating the mAP metric.

    Can output multiple classes for same bounding box.
    """

    def __init__(self, num_select: int):
        super().__init__()
        self.num_select = num_select

    @torch.no_grad()
    def forward(
        self,
        outputs: OutputFeatures,
        hws: Optional[torch.Tensor | list[tuple[int, int]]] = None,
        label_map: Optional[dict[int, int]] = None,
        box_format: Optional[str] = "cxcywh",
    ) -> list[dict[str, torch.Tensor]]:
        # pick num_select queries with highest confidence
        # NOTE: same bbox can be assigned to multiple classes
        #       AFAIK this is only done to calculate the mAP metric :/
        out_probs = get_output_caption_probs(outputs)
        top_probs, top_idx = torch.topk(
            out_probs.flatten(-2), k=self.num_select, dim=1
        )  # [B, N_s]
        top_cap_ids = top_idx % out_probs.shape[2]
        top_box_ids = top_idx // out_probs.shape[2]
        top_bboxes = torch.gather(
            outputs["bbox"],
            dim=1,
            index=top_box_ids[:, :, None].expand(-1, -1, 4),
        )  # [B, N_s, 4]
        top_masks = torch.gather(
            outputs["mask"], dim=1, index=top_box_ids
        )  # [B, N_s]

        # return in the standard format
        processed = [
            {
                "scores": p[~m],  # [N_s]
                "labels": c[~m],  # [N_s]
                "boxes": b[~m],  # [N_s, 4]
            }
            for p, c, b, m in zip(
                top_probs, top_cap_ids, top_bboxes, top_masks
            )
        ]
        processed = process_outputs(processed, hws, label_map, box_format)
        return processed


class PostProcessorExclusive(nn.Module):
    """
    Postprocessor for inference.

    One class per bounding box.
    """

    def __init__(self, confidence_threshold: float):
        super().__init__()
        self.confidence_threshold = confidence_threshold

    @torch.no_grad()
    def forward(
        self,
        outputs: OutputFeatures,
        hws: Optional[torch.Tensor | list[tuple[int, int]]] = None,
        label_map: Optional[dict[int, int]] = None,
        box_format: Optional[str] = "cxcywh",
    ) -> list[dict[str, torch.Tensor]]:
        # iterate over batch dim since each output has
        # different number of out_bboxes after thresholding
        out_probs = get_output_caption_probs(outputs)
        processed = []
        for cap_prob, bbox, mask in zip(
            out_probs, outputs["bbox"], outputs["mask"]
        ):
            # filter out low prob bboxes
            cap_score, cap_idx = cap_prob.max(dim=-1)
            score_mask = (cap_score > self.confidence_threshold) & (
                ~mask
            )  # [N_q], True -> valid
            processed.append(
                {
                    "scores": cap_score[score_mask],  # [N_s]
                    "labels": cap_idx[score_mask],  # [N_s]
                    "boxes": bbox[score_mask],  # [N_s, 4]
                }
            )
        processed = process_outputs(processed, hws, label_map, box_format)
        return processed
