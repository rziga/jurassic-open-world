import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from torchvision.datasets import CocoDetection
from torchvision.transforms import v2
from tqdm import tqdm

from jurassic_open_world.inference.predictor import GroundingDINOPredictor
from jurassic_open_world.modeling.model.grounding_dino import (
    GroundingDINOConfig,
)


class COCODetectionPatch(CocoDetection):
    """
    A small wrapper that also returns image id and shape even if there are no targets.
    """

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        return (
            *super().__getitem__(index),
            img_id,
            img_info["height"],
            img_info["width"],
        )


def parse_args():
    parser = ArgumentParser("Efficiency evaluation script.")
    parser.add_argument(
        "--root", type=Path, required=True, help="Path to COCO root."
    )
    parser.add_argument(
        "--anno",
        type=Path,
        required=True,
        help="Path to COCO annotation file.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        required=True,
        help="Path to model config file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device on which to run the inference.",
    )
    parser.add_argument(
        "--cache-text",
        action="store_true",
        help="Whether to cache text features or not.",
    )
    return parser.parse_args()


def main(args):
    print("Loading dataset...")
    dataset = COCODetectionPatch(
        args.coco_root, args.coco_anno, transform=v2.Resize((800, 1333))
    )
    coco_gt = COCO(args.coco_anno)

    print("Loading model...")
    model_config = GroundingDINOConfig.from_yaml(args.model_config)
    model = model_config.build()
    predictor = GroundingDINOPredictor(
        model, "inclusive", num_select=300, device=args.device
    )

    # captions will be just all category names
    captions = [
        cat["name"].lower().replace("_", " ") for cat in coco_gt.cats.values()
    ]
    ids = [cat["id"] for cat in coco_gt.cats.values()]

    # cache if needed
    if args.cache_text:
        predictor.set_text(captions)

    # warmup
    print("Warmup...")
    for i in tqdm(range(10)):
        image, _, _, image_height, image_width = dataset[i]

        if args.cache_text:
            _ = predictor.set_image(image).predict(
                target_hw=(image_height, image_width),
                label_map=ids,
                box_format="xywh",  # type: ignore
            )
        else:
            _ = (
                predictor.set_text(captions)
                .set_image(image)
                .predict(
                    target_hw=(image_height, image_width),
                    label_map=ids,
                    box_format="xywh",  # type: ignore
                )
            )

    # run the speed eval
    print("Testing speed and memory consumption...")
    times = []
    mem_usages = []
    for i in tqdm(range(100)):
        image, _, _, image_height, image_width = dataset[i]

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()
        start = time.time()

        if args.cache_text:
            _ = predictor.set_image(image).predict(
                target_hw=(image_height, image_width),
                label_map=ids,
                box_format="xywh",  # type: ignore
            )
        else:
            _ = (
                predictor.set_text(captions)
                .set_image(image)
                .predict(
                    target_hw=(image_height, image_width),
                    label_map=ids,
                    box_format="xywh",  # type: ignore
                )
            )

        stop = time.time()
        torch.cuda.synchronize()
        max_mem = torch.cuda.max_memory_allocated()
        mem = (max_mem - start_mem) / 1e6  # convert bytes to MB

        mem_usages.append(mem)
        times.append(stop - start)

    res = {
        "time (ms)": [1000 * t for t in times],
        "fps": [1 / t for t in times],
        "mem usage (MB)": mem_usages,
    }

    # print results
    for k, v in res.items():
        vals = np.array(v)
        print(f"{k}: {vals.mean():.2f} +/- {vals.std():.2f}")

    return res


if __name__ == "__main__":
    main(parse_args())
