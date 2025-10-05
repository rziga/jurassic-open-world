from torch import nn
from torchvision import tv_tensors


class NormalizeBoundingBoxes(nn.Module):
    """Normalize bounding box coordinates to 0-1.
    **WARNING:** Should be used as the last transform as it will otherwise mess up downstream bounding box augmentations.
    """

    def forward(
        self, img: tv_tensors.Image, bbox: tv_tensors.BoundingBoxes, *args
    ) -> tuple[tv_tensors.Image, tv_tensors.BoundingBoxes]:
        format = bbox.format
        H, W = bbox.canvas_size
        inpt = bbox.float()
        inpt[..., 0::2] /= W
        inpt[..., 1::2] /= H
        bbox = tv_tensors.BoundingBoxes(
            inpt, format=format, canvas_size=(1, 1)
        )  # type: ignore

        return img, bbox, *args
