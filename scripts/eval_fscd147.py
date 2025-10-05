import json
from argparse import ArgumentParser

import torch
from torchmetrics.functional.detection import mean_average_precision
from torchmetrics.functional.regression import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from torchvision.ops import box_convert
from torchvision.transforms import v2
from tqdm import tqdm

from jurassic_open_world.data.dataset import (
    TextBasedDetectionDataset,
    TextBasedDetectionDatasetConfig,
)
from jurassic_open_world.data.dataset_plugins.od import (
    ObjectDetectionDatasetPluginConfig,
)
from jurassic_open_world.inference.predictor import GroundingDINOPredictor
from jurassic_open_world.modeling.model.grounding_dino import GroundingDINO


def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def load_jsonl(path):
    with open(path, "r") as f:
        data = [json.loads(line) for line in f.read().splitlines()]
    return data


class FSC147Evaluator:
    def __init__(
        self,
        odvg_anno_path: str,
        fsc_anno_path: str,
        preds: list,
        thresh: float,
    ):
        odvg_anno = load_jsonl(odvg_anno_path)
        self.odvg_anno = {
            el["filename"]: el["detection"]["instances"] for el in odvg_anno
        }
        self.fsc_anno = load_json(fsc_anno_path)
        self.pred = preds
        self.thresh = thresh

    def eval(self):
        stats = self.get_stats(self.odvg_anno, self.fsc_anno, self.pred)

        for name, stat in stats.items():
            print(f"{name}: {stat}")

    def get_stats(self, odvg_anno, fsc_anno, pred) -> dict[str, float]:
        img_names = [el["img_name"] for el in pred]

        # counting metrics
        pred_scores = [torch.tensor(el["scores"]) for el in pred]
        pred_counts = [(score > self.thresh).sum() for score in pred_scores]
        target_counts = [
            len(fsc_anno[img_name]["points"]) for img_name in img_names
        ]
        mae = mean_absolute_error(
            torch.tensor(pred_counts), torch.tensor(target_counts)
        )
        rmse = mean_squared_error(
            torch.tensor(pred_counts),
            torch.tensor(target_counts),
            squared=False,
        )
        mape = mean_absolute_percentage_error(
            torch.tensor(pred_counts), torch.tensor(target_counts)
        )

        # detection metrics
        preds = [
            {
                "boxes": torch.tensor(el["boxes"]),
                "scores": torch.tensor(el["scores"]),
                "labels": torch.tensor(el["labels"]),
            }
            for el in pred
        ]
        for el in preds:
            mask = el["scores"] > self.thresh
            el["boxes"] = el["boxes"][mask]
            el["scores"] = el["scores"][mask]
            el["labels"] = el["labels"][mask]

        targets = [
            {
                "boxes": box_convert(
                    torch.tensor([el["bbox"] for el in odvg_anno[img_name]]),
                    "xyxy",
                    "xywh",
                ),
                "labels": torch.tensor([0 for _ in odvg_anno[img_name]]),
            }
            for img_name in img_names
        ]
        map = mean_average_precision(
            preds,
            targets,
            box_format="xywh",
            backend="pycocotools",
            max_detection_thresholds=[10000, 10000, 10000],
        )

        return {
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "AP": map["map"],
            "AP@50": map["map_50"],
        }  # type: ignore


def parse_args():
    parser = ArgumentParser("FSCD-147 evaluation script.")
    parser.add_argument(
        "--root", type=str, required=True, help="Path to FSCD-147 root."
    )
    parser.add_argument(
        "--odvg-anno",
        type=str,
        required=True,
        help="Path to odvg annotation file.",
    )
    parser.add_argument(
        "--odvg-label-map",
        type=str,
        required=True,
        help="Path to odvg label map annotation",
    )
    parser.add_argument(
        "--fsc-anno",
        type=str,
        required=True,
        help="Path to FSC147 annotation file.",
    )
    parser.add_argument(
        "--thresh",
        type=float,
        required=False,
        default=0.3,
        help="Detection Threshold",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to dir with model.safetensors and config.yaml.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device on which to run the inference.",
    )
    return parser.parse_args()


def main(args):
    print("Loading dataset...")
    dataset = TextBasedDetectionDataset(
        TextBasedDetectionDatasetConfig(
            ObjectDetectionDatasetPluginConfig(
                root=args.root,
                annotation_fpath=args.odvg_anno,
                label_map_fpath=args.odvg_label_map,
                mode="train",
            ),
            transform_cfg=None,
            num_false_captions=None,
            max_captions=None,
        ),
        v2.Resize(size=None, max_size=1333),
    )

    print("Loading model...")
    model = GroundingDINO.from_pretrained(args.model_path)
    predictor = GroundingDINOPredictor(
        model,
        "inclusive",
        num_select=model.cfg.query_selector_cfg.num_queries,
        device=args.device,
    )

    # run the prediction
    print("Predicting...")
    detections = []
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]

        outputs, _ = (
            predictor.set_text(sample["captions"])
            .set_image(sample["img"])
            .predict(target_hw=sample["meta"]["hw"], box_format="xywh")
        )

        detections.append(
            {
                "boxes": outputs["boxes"].tolist(),  # type: ignore
                "scores": outputs["scores"].tolist(),  # type: ignore
                "labels": outputs["labels"].tolist(),  # type: ignore
                "img_name": sample["meta"]["fpath"].name,
            }
        )

    # run the eval
    evaluator = FSC147Evaluator(
        args.odvg_anno, args.fsc_anno, detections, args.thresh
    )
    evaluator.eval()


if __name__ == "__main__":
    main(parse_args())
