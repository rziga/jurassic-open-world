import contextlib
import io
from argparse import ArgumentParser
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.datasets import CocoDetection
from torchvision.transforms import v2
from tqdm import tqdm

from jurassic_open_world.inference.predictor import GroundingDINOPredictor
from jurassic_open_world.modeling.model.grounding_dino import GroundingDINO


class COCODetectionPatch(CocoDetection):
    """A small wrapper that also returns image id and shape even if there are no targets."""

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        return (
            *super().__getitem__(index),
            img_id,
            img_info["height"],
            img_info["width"],
        )


class COCOEvaluator:
    """A small wrapper for cocoeval.CocoEval"""

    def __init__(self, coco_gt: COCO, coco_dt: COCO):
        self.coco_gt = coco_gt
        self.coco_dt = coco_dt

    def eval(self):
        # original stats
        stats = self.get_stats(self.coco_gt, self.coco_dt)

        # stat names
        stat_templates = [
            "AP",
            "AP50",
            "AP75",
            "APs",
            "APm",
            "APl",
            "AR@1",
            "AR@10",
            "AR@100",
            "ARs@100",
            "ARm@100",
            "ARl@100",
        ]

        for template, stat in zip(stat_templates, stats):
            print(f"{template}: {stat}")

    def get_stats(self, coco_gt: COCO, coco_dt: COCO):
        with contextlib.redirect_stdout(io.StringIO()):
            coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        return coco_eval.stats


def parse_args():
    parser = ArgumentParser("COCO evaluation script.")
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
        "--model-path",
        type=Path,
        required=True,
        help="Path to dir with model.safetensors and config.yaml.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device on which to run the inference.",
    )
    parser.add_argument(
        "--max-img-size",
        type=int,
        default=1333,
        help="Max image size (longer side) in px.",
    )
    return parser.parse_args()


def main(args):
    print("Loading dataset...")
    dataset = COCODetectionPatch(
        args.root,
        args.anno,
        transform=v2.Resize(800, max_size=args.max_img_size),
    )
    coco_gt = COCO(args.anno)

    print("Loading model...")
    model = GroundingDINO.from_pretrained(args.model_path)
    predictor = GroundingDINOPredictor(
        model, "inclusive", num_select=300, device=args.device
    )

    # captions will be just all category names
    captions = [
        cat["name"].lower().replace("_", " ") for cat in coco_gt.cats.values()
    ]
    ids = [cat["id"] for cat in coco_gt.cats.values()]

    # cache text
    predictor.set_text(captions)

    # run the eval
    print("Predicting...")
    coco_detections = []
    for i in tqdm(range(len(dataset))):
        image, _, image_id, image_height, image_width = dataset[i]

        outputs, _ = predictor.set_image(image).predict(
            target_hw=(image_height, image_width),
            label_map=ids,  # type: ignore
            box_format="xywh",
        )

        for score, box, label in zip(
            outputs["scores"], outputs["boxes"], outputs["labels"]
        ):
            coco_detections.append(
                {
                    "image_id": image_id,
                    "category_id": label.item(),
                    "bbox": box.tolist(),
                    "score": score.item(),
                }
            )

    # eval
    coco_dt = coco_gt.loadRes(coco_detections)  # type: ignore
    evaluator = COCOEvaluator(coco_gt, coco_dt)
    evaluator.eval()


if __name__ == "__main__":
    main(parse_args())
