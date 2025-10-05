from argparse import ArgumentParser
from pathlib import Path

from lvis import LVIS, LVISEval, LVISResults
from torchvision.datasets import CocoDetection
from torchvision.transforms import v2
from tqdm import tqdm

from jurassic_open_world.inference.predictor import GroundingDINOPredictor
from jurassic_open_world.modeling.model.grounding_dino import GroundingDINO


class LVISDetectionPatch(CocoDetection):
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


class LVISEvaluator:
    def __init__(self, lvis_gt: LVIS, lvis_dt: LVISResults):
        self.lvis_gt = lvis_gt
        self.lvis_dt = lvis_dt

    def eval(self):
        stats = self.get_stats(self.lvis_gt, self.lvis_dt)
        for name, stat in stats.items():
            print(f"{name}: {stat}")

    def get_stats(self, lvis_gt: LVIS, lvis_dt: LVISResults):
        lvis_eval = LVISEval(lvis_gt, lvis_dt, iou_type="bbox")
        lvis_eval.run()
        return lvis_eval.get_results()


def parse_args():
    parser = ArgumentParser("LVIS evaluation script.")
    parser.add_argument(
        "--root", type=Path, required=True, help="Path to LVIS root."
    )
    parser.add_argument(
        "--anno",
        type=Path,
        required=True,
        help="Path to LVIS annotation file.",
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
        default="cuda",
        help="Device on which to run the inference.",
    )
    parser.add_argument(
        "--max-img-size",
        type=int,
        default=1333,
        help="Max image size (longer side) in px",
    )
    return parser.parse_args()


def main(args):
    print("Loading dataset...")
    dataset = LVISDetectionPatch(
        args.root,
        args.anno,
        transform=v2.Resize(800, max_size=args.max_img_size),
    )
    lvis_gt = LVIS(args.anno)

    print("Loading model...")
    model = GroundingDINO.from_pretrained(args.model_path)
    predictor = GroundingDINOPredictor(
        model, "inclusive", num_select=300, device=args.device
    )

    # captions will be just all category names
    captions = [
        cat["name"].lower().replace("_", " ") for cat in lvis_gt.cats.values()
    ]
    ids = [cat["id"] for cat in lvis_gt.cats.values()]

    # cache text
    predictor.set_text(captions)

    # run the prediction
    print("Predicting...")
    lvis_detections = []
    for i in tqdm(range(len(dataset))):
        image, _, image_id, image_height, image_width = dataset[i]

        outputs, _ = predictor.set_image(image).predict(
            target_hw=(image_height, image_width),
            label_map=ids,
            box_format="xywh",  # type: ignore
        )

        for score, box, label in zip(
            outputs["scores"], outputs["boxes"], outputs["labels"]
        ):  # type: ignore
            lvis_detections.append(
                {
                    "image_id": image_id,
                    "category_id": label.item(),
                    "bbox": box.tolist(),
                    "score": score.item(),
                }
            )

    # run the eval
    lvis_dt = LVISResults(lvis_gt, lvis_detections)
    evaluator = LVISEvaluator(lvis_gt, lvis_dt)
    evaluator.eval()


if __name__ == "__main__":
    main(parse_args())
