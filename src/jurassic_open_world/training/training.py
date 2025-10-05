from dataclasses import dataclass

import lightning as L
import torch

from ..inference.postprocessor import (
    PostProcessorExclusive,
    PostProcessorInclusive,
)
from ..modeling.loss.denoising_loss import GroundingDINODenoisingLossConfig
from ..modeling.loss.detection_loss import GroundingDINODetectionLossConfig
from ..modeling.model.grounding_dino import GroundingDINO
from ..utils.config import BaseConfig
from ..utils.types import Batch, ImageMetadata, ModelInput, ModelTarget


@dataclass
class GroundingDINOTrainingConfig(BaseConfig["GroundingDINOTraining"]):
    detection_loss_cfg: GroundingDINODetectionLossConfig
    denoising_loss_cfg: GroundingDINODenoisingLossConfig

    lr: float
    image_backbone_lr: float
    text_backbone_lr: float
    weight_decay: float
    lr_drop_step: list[int]
    num_epochs: int
    grad_clip_val: float


class GroundingDINOTraining(L.LightningModule):
    def __init__(self, cfg: GroundingDINOTrainingConfig, model: GroundingDINO):
        super().__init__()
        self.save_hyperparameters(
            {"training_cfg": cfg.to_dict(), "model_cfg": model.cfg.to_dict()}
        )

        self.lr = cfg.lr
        self.img_lr = cfg.image_backbone_lr
        self.txt_lr = cfg.text_backbone_lr
        self.weight_decay = cfg.weight_decay
        self.lr_drop_step = cfg.lr_drop_step

        # set up model
        self.model = model

        # freeze backbones if needed
        self.img_backbone_frozen = self.img_lr == 0
        self.txt_backbone_frozen = self.txt_lr == 0
        if self.img_backbone_frozen:
            self.model.freeze_image_backbone()
        if self.txt_backbone_frozen:
            self.model.freeze_text_backbone()

        # set up loss function
        self.detection_loss_fn = cfg.detection_loss_cfg.build()
        self.denoising_loss_fn = cfg.denoising_loss_cfg.build()

        # set up postprocessors
        self.postprocessor_map = PostProcessorInclusive(num_select=100)
        self.postprocessor_inference = PostProcessorExclusive(
            confidence_threshold=0.3
        )

    def forward(
        self,
        inputs: ModelInput,
        targets: ModelTarget,
        meta: list[ImageMetadata],
        denoise: bool,
    ) -> dict:
        # forward through model
        outputs = self.model(**inputs)

        loss = 0.0

        # detection loss
        detection_loss, (det_cls, det_l1, det_giou) = self.detection_loss_fn(
            outputs, targets
        )
        loss += detection_loss

        # denoising loss
        if denoise:
            denoising_loss, (cdn_cls, cdn_l1, cdn_giou) = (
                self.denoising_loss_fn(outputs, targets, self.model)
            )
            loss += denoising_loss

        # postprocess outputs from last layer
        map_outputs = self.postprocessor_map(outputs["outputs"])
        inference_outputs = self.postprocessor_inference(outputs["outputs"])

        return {
            "loss": loss,
            "loss_components": {
                "det": detection_loss.item(),
                "det_cls": det_cls,
                "det_l1": det_l1,
                "det_giou": det_giou,
            }
            | (
                {
                    "cdn": denoising_loss.item(),
                    "cdn_cls": cdn_cls,
                    "cdn_l1": cdn_l1,
                    "cdn_giou": cdn_giou,
                }
                if denoise
                else {}
            ),
            "outputs": outputs["outputs"],
            "map_outputs": map_outputs,
            "inference_outputs": inference_outputs,
        }

    def training_step(self, batch: Batch) -> dict:
        return self(**batch, denoise=True)

    def validation_step(
        self, batch: Batch, batch_idx: int, dataloader_idx: int = 0
    ) -> dict:
        return self(**batch, denoise=False)

    def configure_optimizers(self) -> dict:
        # fmt: off
        param_groups = param_groups = [
            {"params": self.model.non_backbone_parameters(), "lr": self.lr},
            {"params": self.denoising_loss_fn.parameters(), "lr": self.lr},
        ]
        if not self.img_backbone_frozen:
            param_groups += [{
                "params": self.model.image_backbone_parameters(),
                "lr": self.img_lr,
            }]
        if not self.txt_backbone_frozen:
            param_groups += [{
                "params": self.model.text_backbone_parameters(),
                "lr": self.txt_lr,
            }]
        # fmt: on

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.weight_decay,
        )
        scheduler = scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, self.lr_drop_step, gamma=0.1
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }


GroundingDINOTrainingConfig._target_class = GroundingDINOTraining
