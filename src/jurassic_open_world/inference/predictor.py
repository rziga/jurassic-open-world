from pathlib import Path
from typing import Literal, Mapping, Optional

import numpy as np
import PIL.Image
import torch
from torch import nn
from torchvision.transforms import v2
from torchvision.transforms.v2.functional import to_image

from .postprocessor import PostProcessorExclusive, PostProcessorInclusive
from .visualization import draw_results
from ..modeling.model.grounding_dino import GroundingDINO


class GroundingDINOPredictor(nn.Module):
    """
    Predictor wrapper for easier inference. Support only batchless elements for simplicity.

    It is meant to be used statefully, so you need to call set_text() and set_image() first and then predict().
    Alternatively, the forward method is stateless.

    Args:
        model (GroundingDINO): Trained model to perform inference.
        postprocess_type (Literal["exclusive", "inclusive"]): Postprocessor type for inference.
            "exclusive" for normal inference, "inclusive" for calculating metrics.
            Check `PostProcessorExclusive` and `PostProcessorInclusive` for more details.
        confidence_threshold (float): Confidence threshold for exclusive postprocessor. Defaults to 0.3.
        num_select (int): Number of boxes to select for inclusive postprocessor. Defaults to 300.
        device (str): Device on which to perform inference. Defaults to "cuda".
    """

    def __init__(
        self,
        model: GroundingDINO,
        postprocess_type: Literal["exclusive", "inclusive"] = "exclusive",
        confidence_threshold: float = 0.3,
        num_select: int = 300,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.model = model.eval().to(device)
        self.image_preprocessor = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(model.cfg.img_mean, model.cfg.img_std),
            ]
        ).to(device)

        if postprocess_type == "exclusive":
            self.postprocessor = PostProcessorExclusive(
                confidence_threshold
            ).to(device)
        elif postprocess_type == "inclusive":
            self.postprocessor = PostProcessorInclusive(num_select).to(device)
        else:
            raise ValueError(f"{postprocess_type=} is not supported")

        self.img_cache = None
        self.txt_cache = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        postprocess_type: Literal["exclusive", "inclusive"] = "exclusive",
        confidence_threshold: float = 0.3,
        num_select: int = 300,
        device: str = "cuda",
    ) -> "GroundingDINOPredictor":
        """
        Inits directly from pretrained.

        Args:
            cfg (str | GroundingDINOConfig): path to config .yaml file or GroundingDINOConfig instance.
            state (str | dict): path to state dict .pt file or state dict.
            postprocess_type (Literal["exclusive", "inclusive"]): Postprocessor type for inference.
                "exclusive" for normal inference, "inclusive" for calculating metrics.
                Check `PostProcessorExclusive` and `PostProcessorInclusive` for more details.
            confidence_threshold (float): Confidence threshold for exclusive postprocessor. Defaults to 0.3.
            num_select (int): Number of boxes to select for inclusive postprocessor. Defaults to 300.
            device (str): Device on which to perform inference. Defaults to "cuda".
        """
        return cls(
            GroundingDINO.from_pretrained(pretrained_model_name_or_path),
            postprocess_type,
            confidence_threshold,
            num_select,
            device,
        )

    @torch.no_grad()
    def set_text(self, txt: list[str]) -> "GroundingDINOPredictor":
        """
        Calculate text embeddings and cache them.

        Args:
            txt (list[str]): Text prompts for detection. Usually category names.

        Returns:
            out (GroundingDINOPredictor): self
        """
        self.txt_cache = {
            "input": txt,
            "feat": self.model.txt_backbone([txt]),
        }
        return self

    def clear_text(self) -> "GroundingDINOPredictor":
        """
        Clear text cache.
        """
        self.txt_cache = None
        return self

    @torch.no_grad()
    def set_image(
        self, img: torch.Tensor | PIL.Image.Image | np.ndarray
    ) -> "GroundingDINOPredictor":
        """
        Calculate image embeddings and cache them.

        Args:
            img (torch.Tensor | PIL.Image.Image | np.ndarray): Image to detect objects on.
                If torch.Tensor it should have shape [3, H, W], if np.ndarray it should have shape [H, W, 3], if PIL.Image it should be in RGB.

        Returns:
            out (GroundingDINOPredictor): self
        """
        img_processed = (
            self.image_preprocessor(img).to(self.device).unsqueeze(0)
        )
        img_mask = torch.zeros_like(
            img_processed[:, 0, :, :], dtype=torch.bool
        )
        img_feat = self.model.img_backbone(img_processed, img_mask)
        self.img_cache = {
            "input": img,
            "feat": img_feat,
        }
        return self

    def clear_image(self) -> "GroundingDINOPredictor":
        """
        Clear image cache.
        """
        self.img_cache = None
        return self

    @torch.no_grad()
    def predict(
        self,
        target_hw: Optional[tuple[int, int]] = None,
        label_map: Optional[Mapping[int, int]] = None,
        box_format: Literal["cxcywh", "xyxy", "xywh"] = "cxcywh",
        draw: bool = False,
        draw_kwargs: Optional[dict] = None,
    ) -> tuple[dict[str, torch.Tensor], Optional[PIL.Image.Image]]:
        """
        Stateful inference.

        Compute detection from cached image and text embeddings.

        Args:
            target_hw (Optional[tuple[int, int]], optional): Target height and width to which to scale bounding boxes to.
                If None, the boxes are scaled to height and width of input image. Defaults to None.
            label_map (Optional[Mapping[int, int]], optional): Label mapping for label conversion.
                If None the boxes will have ids based on list index of the associated text prompt.
                Label map then maps these ids to custom ids. Useful for COCO. Defaults to None.
            box_format (Literal["cxcywh", "xyxy", "xywh"], optional): Output bounding box format. Defaults to "cxcywh".
            draw (bool): Whether or not to draw the results on the image.
            draw_kwargs (dict, optional): kwargs for

        Returns:
            out (dict[str, torch.Tensor]): Torchmetrics standard output, dict with keys "boxes", "labels", "scores".
        """
        # forward through the model with cached data
        assert self.img_cache is not None, (
            "You need to `set_image(img)` before predicting."
        )
        img_feat = self.img_cache["feat"]
        assert self.txt_cache is not None, (
            "You need to `set_text(txt)` before predicting."
        )
        txt_feat = self.txt_cache["feat"]
        fused_img_feat, fused_txt_feat = self.model.encoder(img_feat, txt_feat)
        _, query_feat = self.model.query_selector(
            fused_img_feat, fused_txt_feat
        )
        dec_outputs = self.model.decoder(
            query_feat, fused_img_feat, fused_txt_feat
        )
        outputs = dec_outputs[-1]

        # load input image and text from cache
        img, txt = self.img_cache["input"], self.txt_cache["input"]
        img = to_image(img)  # convert so we can get shapes consistently
        img_h, img_w = img.shape[-2:]

        # postprocessing
        if target_hw is None:
            target_hw = (img_h, img_w)
        res = self.postprocessor(
            outputs,
            hws=[target_hw],
            label_map=label_map,
            box_format=box_format,
        )[0]

        # drawing
        if draw:
            if draw_kwargs is None:
                draw_kwargs = {}
            res_draw = self.postprocessor(outputs, hws=[(img_h, img_w)])[0]
            draw_img = draw_results(img, txt, res_draw, **draw_kwargs)
        else:
            draw_img = None

        return res, draw_img

    @torch.no_grad()
    def forward(
        self,
        img: torch.Tensor | PIL.Image.Image | np.ndarray,
        txt: list[str],
        target_hw: Optional[tuple[int, int]] = None,
        label_map: Optional[Mapping[int, int]] = None,
        box_format: Literal["cxcywh", "xyxy", "xywh"] = "cxcywh",
        draw: bool = False,
        draw_kwargs: Optional[dict] = None,
    ) -> tuple[dict[str, torch.Tensor], Optional[PIL.Image.Image]]:
        """
        Stateless inference.

        Recompute image and text embeddings each time.

        Args:
            img (torch.Tensor | PIL.Image.Image | np.ndarray): Image to detect objects on.
                If torch.Tensor it should have shape [3, H, W], if np.ndarray it should have shape [H, W, 3], if PIL.Image it should be in RGB.
            txt (list[str]): Text prompts for detection. Usually category names.
            draw (bool): Whether or not to draw the results on the image.
            img_hw (Optional[tuple[int, int]], optional): Target height and width to which to scale bounding boxes to.
                If None, the boxes are scaled to height and width of input image. Defaults to None.
            label_map (Optional[Mapping[int, int]], optional): Label mapping for label conversion.
                If None the boxes will have ids based on list index of the associated text prompt.
                Label map then maps these ids to custom ids. Useful for COCO. Defaults to None.
            box_format (Literal["cxcywh", "xyxy", "xywh"], optional): Output bounding box format. Defaults to "cxcywh".

        Returns:
            out (dict[str, torch.Tensor]): Torchmetrics standard output, dict with keys "boxes", "labels", "scores".
        """
        output = (
            self.set_image(img)
            .set_text(txt)
            .predict(target_hw, label_map, box_format, draw, draw_kwargs)
        )
        self.clear_image()
        self.clear_text()
        return output
