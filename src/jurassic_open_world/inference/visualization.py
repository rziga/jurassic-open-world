import numpy as np
import PIL.Image
import torch
from torchvision.ops import box_convert
from torchvision.transforms.v2.functional import to_image, to_pil_image
from torchvision.utils import draw_bounding_boxes


def draw_results(
    img: torch.Tensor | PIL.Image.Image | np.ndarray,
    txt: list[str],
    results: dict[str, torch.Tensor],
    **draw_kwargs,
) -> PIL.Image.Image:
    """
    Draw bounding boxes onto the image.

    Expects:
        * bounding boxes in results to be scaled to image height and width
        * bounding boxes to bo in cxcywh format
        * labels to map to indexes in txt list

    Args:
        img (torch.Tensor | PIL.Image.Image | np.ndarray): Image to draw to.
        txt (list[str]): Text labels.
        results (dict[str, torch.Tensor]): Result in torchmetrics standard format:
            dict with "boxes", "labels", "scores" keys.
        draw_kwargs (dict): kwargs fro `torchvision.utils.draw_bounding_boxes`

    Returns:
        out (PIL.Image.Image): Image with drawn bounding boxes.
    """

    return to_pil_image(
        draw_bounding_boxes(
            image=to_image(img),
            boxes=box_convert(results["boxes"], "cxcywh", "xyxy"),
            labels=[
                f"{txt[lab]}: {score:.2f}"
                for lab, score in zip(results["labels"], results["scores"])
            ],
            **draw_kwargs,
        )
    )
