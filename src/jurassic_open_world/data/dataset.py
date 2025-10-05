from dataclasses import dataclass
from typing import Optional, Union

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

from .dataset_plugins.coco import COCODatasetPluginConfig
from .dataset_plugins.od import ObjectDetectionDatasetPluginConfig
from .dataset_plugins.vg import VisualGroundingDatasetPluginConfig
from .transforms import COCOTransformConfig, SimpleTransformConfig
from ..utils.config import BaseConfig
from ..utils.misc import stack_pad
from ..utils.types import (
    Batch,
    ModelInput,
    ModelTarget,
    TextBasedDetectionDatasetOutput,
)


@dataclass
class TextBasedDetectionDatasetConfig(BaseConfig["TextBasedDetectionDataset"]):
    plugin_cfg: Union[
        ObjectDetectionDatasetPluginConfig,
        VisualGroundingDatasetPluginConfig,
        COCODatasetPluginConfig,
    ]
    transform_cfg: Optional[Union[COCOTransformConfig, SimpleTransformConfig]]
    num_false_captions: Optional[int] = None
    max_captions: Optional[int] = None


class TextBasedDetectionDataset(Dataset):
    def __init__(
        self,
        cfg: TextBasedDetectionDatasetConfig,
        transforms: Optional[v2.Transform] = None,
    ):
        super().__init__()
        self.cfg = cfg

        if cfg.plugin_cfg.mode != "train" and cfg.num_false_captions:
            raise ValueError(
                f"Trying to add {cfg.num_false_captions=} when {cfg.plugin_cfg=} != 'train'!"
            )
        if cfg.plugin_cfg.mode != "train" and cfg.max_captions:
            raise ValueError(
                f"Trying to crop {cfg.max_captions=} when {cfg.plugin_cfg=} != 'train'!"
            )

        self.dataset_plugin = cfg.plugin_cfg.build()
        if cfg.transform_cfg is not None:
            self.transforms = cfg.transform_cfg.build()
        else:
            self.transforms = transforms

    def __getitem__(self, index: int) -> TextBasedDetectionDatasetOutput:
        sample = self.dataset_plugin[index]

        # load img and bboxes
        img = tv_tensors.Image(Image.open(sample["img_fpath"]))
        h, w = img.shape[-2:]
        bboxes = tv_tensors.BoundingBoxes(
            sample["bboxes"], format=sample["bbox_format"], canvas_size=(h, w)
        )  # type: ignore
        bbox_cap_ids = torch.tensor(sample["bbox_cap_ids"])

        # caption
        captions = sample["captions"]

        # transform imgs and bboxes
        if self.transforms is not None:
            img, bboxes, bbox_cap_ids = self.transforms(
                img, bboxes, bbox_cap_ids
            )

        # add additional captions if needed
        if self.cfg.num_false_captions is not None:
            num_added = offset = 0
            while (
                num_added < self.cfg.num_false_captions
                and offset < self.cfg.num_false_captions + 1_000
            ):
                false_captions = self.dataset_plugin[
                    (index + offset) % len(self)
                ]["captions"]
                for cap in false_captions:
                    if cap not in captions:
                        captions += [cap]
                        num_added += 1
                offset += 1

        return {
            "img": img,
            "captions": captions,
            "bboxes": bboxes,
            "bbox_cap_ids": bbox_cap_ids,
            "meta": {
                "hw": (h, w),
                "fpath": sample["img_fpath"],
            },
        }

    def __len__(self) -> int:
        return len(self.dataset_plugin)

    def get_label_map(self) -> dict[str, int] | None:
        return getattr(self.dataset_plugin, "label_map", None)

    @staticmethod
    def collate_fn(samples: list[TextBasedDetectionDatasetOutput]) -> Batch:
        # stack pad images
        img_list = [sample["img"] for sample in samples]
        # [B, 3, H, W], [B, 3, H, W]
        img, img_mask = stack_pad(img_list)
        # [B, H, W]
        img_mask = img_mask[:, 0, :, :]

        # stack pad bboxes and bbox caption ids
        bbox_list = [sample["bboxes"] for sample in samples]
        bbox_caption_id_list = [sample["bbox_cap_ids"] for sample in samples]
        # [B, N, 4]
        bbox, target_mask = stack_pad(bbox_list)
        # [B, N]
        target_mask = target_mask[:, :, 0]
        # [B, N]
        bbox_cap_ids, _ = stack_pad(bbox_caption_id_list)
        bbox_cap_ids[target_mask] = -1

        # extract captions
        captions_list = [sample["captions"] for sample in samples]

        # extract metas
        meta_list = [sample["meta"] for sample in samples]

        # convert to format
        inputs: ModelInput = {
            "img": img,
            "img_mask": img_mask,
            "captions": captions_list,
        }
        targets: ModelTarget = {
            "bbox": bbox,
            "cap_ids": bbox_cap_ids,
            "mask": target_mask,
        }

        return {"inputs": inputs, "targets": targets, "meta": meta_list}


TextBasedDetectionDatasetConfig._target_class = TextBasedDetectionDataset
