import torch
from torch import nn


class ImageFeatureExtractor(nn.Module):
    """
    Base for custom extractor
    """

    @classmethod
    def build_from_config(
        cls, cfg_str: str, pretrained: bool, kwargs
    ) -> "ImageFeatureExtractor":
        """
        Builds the FeatureExtractor based on a config string.
        """
        raise NotImplementedError(
            f"{cls.__name__} should implement a `build` method."
        )

    def forward(self, img: torch.Tensor) -> list[torch.Tensor]:
        """
        Returns outputs of each FPN level.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement a `forward` method."
        )

    def get_channels(self) -> list[int]:
        """
        Returns number of channels of each FPN level output.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement a `get_channels` method."
        )

    def backbone_parameters(self) -> list[nn.Parameter]:
        """
        Returns parameters of the backbone.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement a `get_backbone_params` method."
        )

    def freeze_backbone(self):
        """
        Freezes backbone. Freezes only pretrained parameters.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement a `freeze_backbone` method."
        )
