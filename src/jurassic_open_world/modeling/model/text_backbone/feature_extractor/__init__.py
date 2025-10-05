from typing import Literal

from .base import TextFeatureExtractor
from .openclip import OpenCLIPTextFeatureExtractor
from .transformers import (
    AutoModelTextFeatureExtractor,
    CLIPTextFeatureExtractor,
)


def get_feature_extractor(
    provider: Literal["transformers_auto", "transformers_clip", "openclip"],
    cfg_str: str,
    use_pretrained: bool,
    kwargs: dict,
) -> TextFeatureExtractor:
    if provider == "transformers_auto":
        extractor_type = AutoModelTextFeatureExtractor
    elif provider == "transformers_clip":
        extractor_type = CLIPTextFeatureExtractor
    elif provider == "openclip":
        extractor_type = OpenCLIPTextFeatureExtractor
    else:
        raise ValueError(f"{provider=} is not supported.")

    return extractor_type.build_from_config(cfg_str, use_pretrained, kwargs)
