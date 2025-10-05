import warnings
from typing import Literal

import lightning as L
import torch
import torch.distributed
from lightning.pytorch.utilities import move_data_to_device
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
)
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes

from .types import Batch, ImageMetadata, ModelOutput, ModelTarget


class LossLogger(L.Callback):
    def __init__(self, train_kwargs: dict, val_kwargs: dict):
        super().__init__()
        self.train_kwargs = train_kwargs
        self.val_kwargs = val_kwargs

    def _log_losses(self, stage: Literal["train", "val"], pl_module, outputs):
        kwargs = self.train_kwargs if stage == "train" else self.val_kwargs
        batch_size = outputs["outputs"]["cls"].shape[0]
        pl_module.log(
            f"{stage}/loss", outputs["loss"], batch_size=batch_size, **kwargs
        )
        for component_name, component_value in outputs[
            "loss_components"
        ].items():
            pl_module.log(
                f"{stage}/loss_{component_name}",
                component_value,
                batch_size=batch_size,
                **kwargs,
            )

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        self._log_losses("train", pl_module, outputs)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self._log_losses("val", pl_module, outputs)


class VisualizationLogger(L.Callback):
    def __init__(self, logging_interval: int):
        super().__init__()
        self.logging_interval = logging_interval
        self.should_log = True

    def setup(
        self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str
    ):
        if not hasattr(pl_module.logger, "log_image"):
            warnings.warn(
                "Logger does not support image logging. Skipping image logging."
            )
            self.should_log = False

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: dict,
        batch: Batch,
        batch_idx: int,
    ):
        if (
            self.should_log
            and trainer.global_step % self.logging_interval == 0
        ):
            self._log_image(pl_module, batch, outputs["inference_outputs"])  # type: ignore

    def _log_image(
        self,
        pl_module: L.LightningModule,
        batch: Batch,
        processed_outputs: dict,
    ):
        # move to cpu
        batch = move_data_to_device(batch, device="cpu")
        processed_outputs = move_data_to_device(processed_outputs, "cpu")

        # select only the first element in the batch for logging
        inputs, targets, meta = batch
        img = batch["inputs"]["img"][0]
        mask = batch["inputs"]["img_mask"][0]
        captions = batch["inputs"]["captions"][0]
        target_bboxes = batch["targets"]["bbox"][0]
        target_cap_ids = batch["targets"]["cap_ids"][0]
        processed_outputs = processed_outputs[0]

        # unmask image
        H_valid = (~mask[:, 0]).sum()
        W_valid = (~mask[0, :]).sum()
        img = img[:, :H_valid, :W_valid]

        # normalize img to 0-1
        img = (img - img.min()) / (img.max() - img.min())

        # plot bboxes
        def _draw(bboxes, cap_ids):
            bboxes = box_convert(bboxes, "cxcywh", "xyxy")
            bboxes[:, 0::2] *= W_valid
            bboxes[:, 1::2] *= H_valid
            return draw_bounding_boxes(
                img,
                bboxes,
                labels=[captions[cap_id] for cap_id in cap_ids],
                width=3,
            )

        img1 = _draw(processed_outputs["boxes"], processed_outputs["labels"])
        img2 = _draw(target_bboxes, target_cap_ids)

        # log
        pl_module.logger.log_image(  # type: ignore
            key="train/image",
            images=[img1, img2],
            caption=["predicted", "target"],
        )


class MeanAveragePrecisionLogger(L.Callback):
    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: dict,
        batch: Batch,
        batch_idx: Batch,
    ):
        targets, metas = batch["targets"], batch["meta"]
        batch_size = outputs["outputs"]["cls"].shape[0]
        results = self._calculate_map(
            self._process_outputs(outputs, metas),
            self._process_targets(targets, metas),
        )
        pl_module.log_dict(
            {"train/" + k: v.cuda() for k, v in results.items()},
            batch_size=batch_size,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
        )

    def on_validation_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ):
        self.epoch_outputs: list[dict] = []
        self.epoch_targets: list[dict] = []

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningDataModule,
        outputs: dict,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        targets, metas = batch["targets"], batch["meta"]
        self.epoch_outputs += move_data_to_device(
            self._process_outputs(outputs, metas), "cpu"
        )
        self.epoch_targets += move_data_to_device(
            self._process_targets(targets, metas), "cpu"
        )

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ):
        results = self._calculate_map(self.epoch_outputs, self.epoch_targets)
        pl_module.log_dict({"val/" + k: v for k, v in results.items()})

    def _process_outputs(
        self, outputs: dict, metas: list[ImageMetadata]
    ) -> list[dict]:
        return [
            {
                "boxes": self._unnorm_boxes(output["boxes"], meta["hw"]),
                "scores": output["scores"],
                "labels": output["labels"],
            }
            for output, meta in zip(outputs["map_outputs"], metas)
        ]

    def _process_targets(
        self, targets: ModelTarget, metas: list[ImageMetadata]
    ) -> list[dict]:
        return [
            {
                "boxes": self._unnorm_boxes(bboxes[ids != -1], meta["hw"]),
                "labels": ids[ids != -1],
            }
            for bboxes, ids, meta in zip(
                targets["bbox"], targets["cap_ids"], metas
            )
        ]

    def _unnorm_boxes(self, boxes: torch.Tensor, hw: tuple) -> torch.Tensor:
        hw = torch.tensor(hw, device=boxes.device)  # [2] # type: ignore
        whwh = hw.flip(-1).repeat(2)  # [4] # type: ignore
        return boxes * whwh

    def _calculate_map(self, outputs: list[dict], targets: list[dict]) -> dict:
        results = MeanAveragePrecision("cxcywh")(outputs, targets)
        results.pop("classes")
        return {k: v for k, v in results.items()}


class DistributedMeanAveragePrecisionLogger(MeanAveragePrecisionLogger):
    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ):
        # prepare storage
        if trainer.is_global_zero:
            outputs = [None for _ in range(trainer.world_size)]
            targets = [None for _ in range(trainer.world_size)]
        else:
            outputs = targets = None

        # gather all outputs to rank 0
        torch.distributed.gather_object(self.epoch_outputs, outputs)
        torch.distributed.gather_object(self.epoch_targets, targets)

        # calculate and log mAP
        if trainer.strategy.is_global_zero:
            # flatten lists
            outputs = [el for list in outputs for el in list]  # type: ignore
            targets = [el for list in targets for el in list]  # type: ignore

            results = self._calculate_map(
                self.epoch_outputs, self.epoch_targets
            )
            pl_module.log_dict({"val/" + k: v for k, v in results.items()})


class CountingMetricsLogger(L.Callback):
    def __init__(
        self,
        metric: Literal["mae", "rmse", "mape"],
        confidence_threshold: float,
        autotune_threshold: bool,
    ):
        super().__init__()

        self.metric = metric
        if metric == "mae":
            self.train_metric = MeanAbsoluteError()
            self.val_metric = MeanAbsoluteError()
            self.tuning_metric = MeanAbsoluteError()
        elif metric == "rmse":
            self.train_metric = MeanSquaredError(squared=False)
            self.val_metric = MeanSquaredError(squared=False)
            self.tuning_metric = MeanSquaredError(squared=False)
        elif metric == "mape":
            self.train_metric = MeanAbsolutePercentageError()
            self.val_metric = MeanAbsolutePercentageError()
            self.tuning_metric = MeanAbsolutePercentageError()
        else:
            raise ValueError(f"{metric=} not supported!")

        self.thresh = confidence_threshold
        self.autotune = autotune_threshold

        # for autotuning, populated during validation
        self.outputs: list[ModelOutput] = []
        self.targets: list[ModelTarget] = []

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: dict,
        batch: Batch,
        batch_idx: int,
    ):
        outputs = move_data_to_device(outputs, "cpu")
        targets = move_data_to_device(batch["targets"], "cpu")
        pred_num, target_num = self._count_boxes(outputs, targets, self.thresh)  # type: ignore
        pl_module.log(
            f"train/{self.metric}_step",
            self.train_metric(pred_num, target_num),
        )

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.log(
            f"train/{self.metric}_epoch", self.train_metric.compute()
        )
        self.train_metric.reset()

    def on_validation_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ):
        self.outputs = []
        self.targets = []

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: dict,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        outputs = move_data_to_device(outputs, "cpu")
        targets = move_data_to_device(batch["targets"], "cpu")
        pred_num, target_num = self._count_boxes(outputs, targets, self.thresh)  # type: ignore
        self.val_metric(pred_num, target_num)

        # save outputs and targets for autotuning threshold
        if self.autotune:
            self.outputs.append(outputs)  # type: ignore
            self.targets.append(targets)

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ):
        # autotune threshold if needed
        if self.autotune and not trainer.sanity_checking:
            best_score, best_thresh = float("inf"), 0
            for thresh in torch.linspace(0, 1, 20):
                for output, target in zip(self.outputs, self.targets):
                    pred_num, target_num = self._count_boxes(
                        output, target, float(thresh)
                    )
                    self.tuning_metric(pred_num, target_num)
                score = self.tuning_metric.compute()
                self.tuning_metric.reset()
                if score <= best_score:
                    best_score, best_thresh = score, thresh
            print(
                f"Autotuned {self.metric} threshold based on validation: {best_thresh}."
            )
            self.thresh = float(best_thresh)

        # log epoch level metric
        pl_module.log(f"val/{self.metric}", self.val_metric.compute())
        self.val_metric.reset()

        # clear results
        self.outputs = []
        self.targets = []

    @staticmethod
    def _count_boxes(
        outputs: ModelOutput, targets: ModelTarget, thresh: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_num = (~targets["mask"]).sum(-1)  # [B]
        valid_mask = ~outputs["outputs"]["mask"]  # [B, N]
        score_mask = (
            outputs["outputs"]["cls"].sigmoid().max(-1)[0] >= thresh
        )  # [B, N]
        pred_num = (score_mask & valid_mask).sum(-1)  # [B]
        return pred_num, target_num
