import json
from pathlib import Path
from typing import Literal

from .base import BaseDatasetPlugin, DatasetPluginOutput
from ...utils.config import BaseConfig


class VisualGroundingDatasetPluginConfig(
    BaseConfig["VisualGroundingDatasetPlugin"]
):
    root: str
    mode: Literal["train", "test"]
    annotation_fpath: str
    label_map_fpath: str


class VisualGroundingDatasetPlugin(BaseDatasetPlugin):
    def __init__(self, cfg: VisualGroundingDatasetPluginConfig):
        super().__init__()
        self.root = Path(cfg.root)

        with open(cfg.annotation_fpath, "r") as f:
            self.annotations = [json.loads(line) for line in f]

    def __getitem__(self, index: int) -> DatasetPluginOutput:
        ann = self.annotations[index]

        # load image path
        img_path = self.root / ann["filename"]

        # load grounding boxes and captions
        bboxes, captions, bbox_cap_ids = [], [], []
        for instance in ann["grounding"]["regions"]:
            # if there are multiple phrases, select random one
            caption = instance["phrase"]
            if not isinstance(caption, list):
                caption = [caption]

            # if there is a single bbox, make it a list of 1
            bbox = instance["bbox"]
            if not isinstance(bbox[0], list):
                bbox = [instance["bbox"]]

            for cap in caption:
                cap = cap.lower()
                bboxes += bbox
                if cap not in captions:
                    captions.append(cap.lower())
                bbox_cap_ids += [captions.index(cap)] * len(bbox)

        return {
            "img_fpath": img_path,
            "captions": captions,
            "bboxes": bboxes,
            "bbox_cap_ids": bbox_cap_ids,
            "bbox_format": "xyxy",
        }

    def __len__(self) -> int:
        return len(self.annotations)


VisualGroundingDatasetPluginConfig._target_class = VisualGroundingDatasetPlugin
