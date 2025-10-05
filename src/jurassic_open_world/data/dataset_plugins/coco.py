from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pycocotools.coco import COCO

from .base import BaseDatasetPlugin, DatasetPluginOutput
from ...utils.config import BaseConfig


@dataclass
class COCODatasetPluginConfig(BaseConfig["COCODatasetPlugin"]):
    root: str
    annotation_fpath: str
    mode: Literal["train", "test"]


class COCODatasetPlugin(BaseDatasetPlugin):
    def __init__(self, cfg: COCODatasetPluginConfig):
        super().__init__()
        self.root = Path(cfg.root)
        self.coco = COCO(cfg.annotation_fpath)
        # filter out empty ids
        self.ids = list(
            filter(
                lambda img_id: self.coco.getAnnIds(img_id),
                self.coco.imgs.keys(),
            )
        )
        self.ids.sort()
        self.mode = cfg.mode

    def __getitem__(self, index: int) -> DatasetPluginOutput:
        # get img path and target
        id = self.ids[index]
        img_fname = self.coco.loadImgs(id)[0]["file_name"]
        img_fpath = self.root / img_fname
        targets = self.coco.loadAnns(self.coco.getAnnIds(id))

        # get captions
        if self.mode == "train":
            coco_cat_ids = list(
                set(target["category_id"] for target in targets)
            )
        else:
            coco_cat_ids = list(self.coco.cats.keys())
        captions = [self.coco.cats[id]["name"] for id in coco_cat_ids]

        # get bboxes and bbox caption ids
        bboxes = []
        bbox_cap_ids = []
        for target in targets:
            bboxes.append(target["bbox"])
            bbox_cap_ids.append(coco_cat_ids.index(target["category_id"]))

        return {
            "img_fpath": img_fpath,
            "captions": captions,
            "bboxes": bboxes,
            "bbox_cap_ids": bbox_cap_ids,
            "bbox_format": "xywh",
        }

    def __len__(self) -> int:
        return len(self.ids)


COCODatasetPluginConfig._target_class = COCODatasetPlugin
