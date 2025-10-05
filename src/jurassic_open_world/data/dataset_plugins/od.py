import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .base import BaseDatasetPlugin, DatasetPluginOutput
from ...utils.config import BaseConfig


@dataclass
class ObjectDetectionDatasetPluginConfig(
    BaseConfig["ObjectDetectionDatasetPlugin"]
):
    root: str
    annotation_fpath: str
    label_map_fpath: str
    mode: Literal["train", "test"]


class ObjectDetectionDatasetPlugin(BaseDatasetPlugin):
    def __init__(self, cfg: ObjectDetectionDatasetPluginConfig):
        super().__init__()
        self.cfg = cfg
        self.root = Path(cfg.root)
        self.mode = cfg.mode

        with open(cfg.annotation_fpath, "r") as f:
            gen = (json.loads(line) for line in f)
            self.annotations = list(
                filter(lambda ann: ann["detection"]["instances"], gen)
            )  # TODO: remove filtering of empty annos

        with open(cfg.label_map_fpath, "r") as f:
            self.label_map = json.load(f)

    def __getitem__(self, index: int) -> DatasetPluginOutput:
        ann = self.annotations[index]

        # load image path
        img_path = self.root / ann["filename"]

        # load captions -- just category names in object detection
        # take all possible classes for testing and validation
        if self.mode == "train":
            captions = list(
                set(
                    instance["category"]
                    for instance in ann["detection"]["instances"]
                )
            )
        else:
            captions = list(self.label_map.values())

        bboxes: list[tuple[int, int, int, int]] = []
        bbox_cap_ids: list[int] = []
        for instance in ann["detection"]["instances"]:
            bbox_cap_ids.append(captions.index(instance["category"]))
            bboxes.append(instance["bbox"])

        return {
            "img_fpath": img_path,
            "captions": captions,
            "bboxes": bboxes,
            "bbox_cap_ids": bbox_cap_ids,
            "bbox_format": "xyxy",
        }

    def __len__(self) -> int:
        return len(self.annotations)


ObjectDetectionDatasetPluginConfig._target_class = ObjectDetectionDatasetPlugin
