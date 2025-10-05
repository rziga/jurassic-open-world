from dataclasses import dataclass

import lightning as L
from torch.utils.data import ConcatDataset, DataLoader

from .dataset import TextBasedDetectionDataset, TextBasedDetectionDatasetConfig
from ..utils.config import BaseConfig, _ConfigT


@dataclass
class TextBasedDetectionDataModuleConfig(
    BaseConfig["TextBasedDetectionDataModule"]
):
    train_dataset_cfgs: list[TextBasedDetectionDatasetConfig]
    val_dataset_cfgs: list[TextBasedDetectionDatasetConfig]
    batch_size: int
    num_workers: int

    def to_dict(self) -> dict:
        train_dataset_dicts = [
            cfg.to_dict() for cfg in self.train_dataset_cfgs
        ]
        val_dataset_dicts = [cfg.to_dict() for cfg in self.val_dataset_cfgs]
        out = super().to_dict()
        out["train_dataset_cfgs"] = train_dataset_dicts
        out["val_dataset_cfgs"] = val_dataset_dicts
        return out

    @classmethod
    def from_dict(cls: type[_ConfigT], data: dict) -> _ConfigT:
        obj = super().from_dict(data)
        obj.train_dataset_cfgs = [
            TextBasedDetectionDatasetConfig.from_dict(dict)  # type: ignore
            for dict in obj.train_dataset_cfgs
        ]
        obj.val_dataset_cfgs = [
            TextBasedDetectionDatasetConfig.from_dict(dict)  # type: ignore
            for dict in obj.val_dataset_cfgs
        ]
        return obj


class TextBasedDetectionDataModule(L.LightningDataModule):
    def __init__(self, cfg: TextBasedDetectionDataModuleConfig):
        super().__init__()
        self.save_hyperparameters(cfg.to_dict())

        self.cfg = cfg

        # assigned in setup() which is called internally in lightning.Trainer
        self.train_dataset = None
        self.val_datasets = None
        self.test_datasets = None

    def setup(self, stage):
        if stage == "fit":
            self.train_datasets = [
                dataset_cfg.build()
                for dataset_cfg in self.cfg.train_dataset_cfgs
            ]
        if stage == "fit" or stage == "validate":
            self.val_datasets = [
                dataset_cfg.build()
                for dataset_cfg in self.cfg.val_dataset_cfgs
            ]

    def train_dataloader(self):
        # concat train datasets
        return DataLoader(
            ConcatDataset(self.train_datasets),
            batch_size=self.cfg.batch_size,
            drop_last=False,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            collate_fn=TextBasedDetectionDataset.collate_fn,
        )

    def val_dataloader(self):
        # a list of loaders for individual val_datasets
        if self.val_datasets is None:
            raise RuntimeError(f"{self.val_datasets=}. Call .setup first.")
        return [
            DataLoader(
                val_dataset,
                batch_size=1,
                drop_last=False,
                num_workers=self.cfg.num_workers,
                collate_fn=TextBasedDetectionDataset.collate_fn,
            )
            for val_dataset in self.val_datasets
        ]


TextBasedDetectionDataModuleConfig._target_class = TextBasedDetectionDataModule
