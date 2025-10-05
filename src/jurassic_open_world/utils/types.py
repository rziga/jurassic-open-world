from pathlib import Path
from typing import Annotated, Literal, TypedDict

import torch
from torchvision import tv_tensors


# Shared among all of the features:
#   * masks are always in format True -> padded and False -> unpadded
#   * number of channels C is the same in all features
#   * bboxes are in mask-relative cxcywh coordinates: they go from (0, 0) in top left corner to (1, 1) in bottom right corner of valid image pixels. Valid image pixels are pixels where mask = False.
#   * cap_ids are indices of caption within a batch for each token, -1 if padding or if token does not belong to any caption (e.g. "." in grounding DINO)

# B - batch_size
# N_img - number of feature pyramid pixels sum(H_i*W_i for H_i, W_i in fpn_level_shapes]
# C - number of channels
# L - number of feature pyramid levels
# fmt: off


ImageFeaturesFeat = Annotated[torch.Tensor, "[B, N_img, C]"]
ImageFeaturesPos  = Annotated[torch.Tensor, "[B, N_img, C]"]
ImageFeaturesMask = Annotated[torch.Tensor, "[B, N_img]"]
ImageFeaturesCoor = Annotated[torch.Tensor, "[B, N_img, 2]"]
ImageFeaturesShapes = Annotated[torch.Tensor, "[L, 2]"]
ImageFeaturesValidRatios = Annotated[torch.Tensor, "[B, L, 2]"]

class ImageFeatures(TypedDict):
    """
    Image pyramid features.

    Attributes:
        feat (torch.Tensor): Flattened image pyramid features
        pos (torch.Tensor): Flattened position embeddings
        mask (torch.Tensor): Flattened image padding mask
        coor (torch.Tensor): Flattened relative xy coordinates
        shapes (torch.Tensor): Spatial (H, W) shapes of levels
        valid_ratios (torch.Tensor): (H, W) valid ratios
    """
    feat: ImageFeaturesCoor
    pos:  ImageFeaturesPos
    mask: ImageFeaturesMask
    coor: ImageFeaturesCoor
    shapes: ImageFeaturesShapes
    valid_ratios: ImageFeaturesValidRatios


ImageLevelFeaturesFeat = Annotated[torch.Tensor, "[B, C, H, W]"]
ImageLevelFeaturesPos  = Annotated[torch.Tensor, "[B, C, H, W]"]
ImageLevelFeaturesMask = Annotated[torch.Tensor, "[B, H, W]"]
ImageLevelFeaturesCoor = Annotated[torch.Tensor, "[B, 2, H, W]"]
ImageLevelFeaturesShapes = Annotated[torch.Tensor, "[L, 2]"]

class ImageLevelFeatures(TypedDict):
    """
    Image level features.

    Unflattened features for a single level in a pyramid.

    Attributes:
        feat (torch.Tensor): Unflattened image features
        pos (torch.Tensor): Unflattened position embeddings
        mask (torch.Tensor): Unflattened image masks
        coor (torch.Tensor): Unflattened xy coordinates
        valid_ratios (torch.Tensor): (H, W) valid ratios
    """
    feat: ImageLevelFeaturesFeat
    pos:  ImageLevelFeaturesPos
    mask: ImageLevelFeaturesMask
    coor: ImageLevelFeaturesCoor
    valid_ratios: ImageLevelFeaturesShapes


TextFeaturesFeat = Annotated[torch.Tensor, "[B, N_txt, C]"]
TextFeaturesPos  = Annotated[torch.Tensor, "[B, N_txt, C]"]
TextFeaturesMask = Annotated[torch.Tensor, "[B, N_txt]"]
TextFeaturesAttMask = Annotated[torch.Tensor, "[B, N_txt, N_txt]"]
TextFeaturesCapIds = Annotated[torch.Tensor, "[B, N_txt]"]

class TextFeatures(TypedDict):
    """
    Text features.

    Attributes:
        feat (torch.Tensor): Text features
        pos (torch.Tensor): Text position embeddings
        mask (torch.Tensor): Text padding mask
        att_mask (torch.Tensor): Text self attention mask
        cap_ids (torch.Tensor): Text caption ids
    """
    feat: torch.Tensor
    pos:  torch.Tensor
    mask: torch.Tensor
    att_mask: torch.Tensor
    cap_ids:  torch.Tensor




class QueryFeatures(TypedDict):
    """
    Query features.

    Attributes:
        feat (torch.Tensor): Query features:        [B, N_q, C]
        bbox (torch.Tensor): Query bboxes:          [B, N_q, 4], in relative logit cxcywh coordinates (bbox.sigmoid() to get relative cxcywh)
        mask (torch.Tensor): Query padding mask:    [B, N_q]
        att_mask (torch.Tensor): Query self attention mask: [B, N_q, N_q]
    """
    feat: torch.Tensor
    bbox: torch.Tensor
    mask: torch.Tensor
    att_mask: torch.Tensor


class OutputFeatures(TypedDict):
    """
    Output features.

    Attributes:
        cls (torch.Tensor): Output cls logits:      [B, N_q, N_txt], similarities to each txt token
        bbox (torch.Tensor): Output bbox:           [B, N_q, 4], in relative cxcywh coordinates
        mask (torch.Tensor): Output padding mask:   [B, N_q]
        cap_ids (torch.Tensor): Output caption ids: [B, N_txt]
    """
    cls: torch.Tensor
    bbox: torch.Tensor
    mask: torch.Tensor
    cap_ids: torch.Tensor


class ModelOutput(TypedDict):
    """Model outputs.

    Attributes:
        outputs (OutputFeatures): Outputs from the last decoder layer-
        decoder_outputs (list[OutputFeatures]): Outputs from all decoder layers, last one is the same as `outputs`.
        encoder_output (OutputFeatures): Outputs from the encoder after the query selection.
        img_features (ImageFeaturePyramid): Image features from the encoder before the query selection.
        txt_features (TextFeatures): Text features from the encoder before the query selection.
        query_featuers (QueryFeatures): Query features from the query selector.
    """
    outputs: OutputFeatures
    decoder_outputs: list[OutputFeatures]
    encoder_output: OutputFeatures
    img_features: ImageFeatures
    txt_features: TextFeatures
    query_features: QueryFeatures


class ModelTarget(TypedDict):
    """Model targets.

    Attributes:
        bbox (torch.Tensor): Target bboxes: [B, N_tgt, 4], in relative cxcywh
        mask (torch.Tensor): Target mask:   [B, N_tgt], True -> padded
        cap_ids (torch.Tensor): caption ids of each box (idx of caption to which the box belongs to): [B, N_tgt]
    """
    bbox: torch.Tensor
    mask: torch.Tensor
    cap_ids: torch.Tensor


class ModelInput(TypedDict):
    """Model inputs.

    Attributes:
        img (torch.Tensor): [B, H, W, 3]
        img_mask (torch.Tensor): [B, H, W], True -> padded
        captions (list[list[str]]): where len(captions) is B
    """
    img: torch.Tensor
    img_mask: torch.Tensor
    captions: list[list[str]]


class DatasetPluginOutput(TypedDict):
    """Output of a dataset plugin.

    Attributes:
        img_fpath (Path): path to the image
        captions (list[str]): descriptions of objects on the image
        bboxes (list[tuple[int, int, int, int]]): bounding boxes
        bbox_cap_ids (list[int]): idx of corresponding caption to which the bbox belongs to, the same len as bboxes
        bbox_format (Literal["xywh", "cxcywh", "xyxy"]): format of the bounding boxes
    """
    img_fpath: Path
    captions: list[str]
    bboxes: list[tuple[int, int, int, int]]
    bbox_cap_ids: list[int]
    bbox_format: Literal["xywh", "cxcywh", "xyxy"]


class ImageMetadata(TypedDict):
    """Image metadata.

    Attributes:
        hw (tuple[int, int]): original height and width of the image
        fpath (Path): original file path of the image
    """""""""
    hw: tuple[int, int]
    fpath: Path


class TextBasedDetectionDatasetOutput(TypedDict):
    """Output of the dataset.

    Attributes:
        img (tv_tensors.Image): image array: [3, H, W]
        captions (list[str]): descriptions of objects on the image
        bboxes (tv_tensors.BoundingBoxes): bounding boxes: [N_tgt, 4]
        bbox_cap_ids (torch.tensors): idx of corresponding caption to which the bbox belongs to: [N_tgt]
        meta (ImageMetadata): image metadata
    """
    img: tv_tensors.Image
    captions: list[str]
    bboxes: tv_tensors.BoundingBoxes
    bbox_cap_ids: torch.Tensor
    meta: ImageMetadata


class Batch(TypedDict):
    """Batch.

    Attributes:
        inputs (ModelInput): model inputs
        targets (ModelTarget): model targets
    """
    inputs: ModelInput
    targets: ModelTarget
    meta: list[ImageMetadata]
# fmt: on
