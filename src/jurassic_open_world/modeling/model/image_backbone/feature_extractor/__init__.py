from typing import Literal

from .base import ImageFeatureExtractor
from .transformers import (
    AutoBackboneImageFeatureExtractor,
    CLIPVisionFeatureExtractor,
    TimmBackboneFeatureExtractor,
)


def get_feature_extractor(
    provider: Literal["transformers_auto", "transformers_clip", "timm"],
    cfg_str: str,
    use_pretrained: bool,
    kwargs: dict,
) -> ImageFeatureExtractor:
    if provider == "transformers_auto":
        extractor_type = AutoBackboneImageFeatureExtractor
    elif provider == "transformers_clip":
        extractor_type = CLIPVisionFeatureExtractor
    elif provider == "timm":
        extractor_type = TimmBackboneFeatureExtractor
    else:
        raise ValueError(f"{provider=} is not supported.")
    return extractor_type.build_from_config(cfg_str, use_pretrained, kwargs)
