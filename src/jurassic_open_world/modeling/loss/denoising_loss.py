from dataclasses import dataclass

import torch
from torch import nn

from .contrastive_denoiser import ContrastiveDenoiserConfig
from .set_loss import SetLossConfig
from ..model.grounding_dino import GroundingDINO
from ...utils.config import BaseConfig
from ...utils.types import ModelOutput, ModelTarget


@dataclass
class GroundingDINODenoisingLossConfig(
    BaseConfig["GroundingDINODenoisingLoss"]
):
    set_loss_cfg: SetLossConfig
    denoiser_cfg: ContrastiveDenoiserConfig


class GroundingDINODenoisingLoss(nn.Module):
    """Denoising loss.
    It computes one-to-one set loss for denoising queries in each decoder layer.
    """

    def __init__(self, cfg: GroundingDINODenoisingLossConfig):
        super().__init__()
        self.loss_fn = cfg.set_loss_cfg.build()
        self.denoiser = cfg.denoiser_cfg.build()

    def forward(
        self,
        outputs: ModelOutput,
        targets: ModelTarget,
        model: GroundingDINO,
    ) -> tuple[torch.Tensor, tuple[float, float, float]]:
        loss = 0.0

        cdn_outputs, cdn_pred_idxs, cdn_target_idxs = self.denoiser(
            outputs["img_features"], outputs["txt_features"], targets, model
        )

        for cdn_output in cdn_outputs:
            cdn_layer_loss, components = self.loss_fn(
                cdn_output["cls"],
                cdn_output["bbox"],
                cdn_output["mask"],
                cdn_output["cap_ids"],
                targets["bbox"],
                targets["cap_ids"],
                targets["mask"],
                cdn_pred_idxs,
                cdn_target_idxs,
            )

            assert not cdn_layer_loss.isnan() and not cdn_layer_loss.isinf(), (
                "Anomaly in denoising loss!"
            )

            loss += cdn_layer_loss

        assert isinstance(loss, torch.Tensor)
        return loss, components


GroundingDINODenoisingLossConfig._target_class = GroundingDINODenoisingLoss
