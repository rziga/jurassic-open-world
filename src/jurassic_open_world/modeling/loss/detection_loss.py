from dataclasses import dataclass

import torch
from torch import nn

from .set_loss import SetLossConfig
from ...utils.config import BaseConfig
from ...utils.types import ModelOutput, ModelTarget


@dataclass
class GroundingDINODetectionLossConfig(
    BaseConfig["GroundingDINODetectionLoss"]
):
    set_loss_cfg: SetLossConfig


class GroundingDINODetectionLoss(nn.Module):
    """Detection loss.
    It computes one-to-one set loss for the final encoder layer and each decoder layer.
    """

    def __init__(self, cfg: GroundingDINODetectionLossConfig):
        super().__init__()
        self.loss_fn = cfg.set_loss_cfg.build()

    def forward(
        self,
        outputs: ModelOutput,
        targets: ModelTarget,
    ) -> tuple[torch.Tensor, tuple[float, float, float]]:
        loss = 0.0

        # one-to-one set loss for each layer
        for output in [outputs["encoder_output"]] + outputs["decoder_outputs"]:
            # fmt: off
            layer_loss, components = self.loss_fn(
                output["cls"], output["bbox"], output["mask"], output["cap_ids"],
                targets["bbox"], targets["cap_ids"], targets["mask"],
            )
            # fmt: on

            loss += layer_loss

        return loss, components  # type: ignore


GroundingDINODetectionLossConfig._target_class = GroundingDINODetectionLoss
