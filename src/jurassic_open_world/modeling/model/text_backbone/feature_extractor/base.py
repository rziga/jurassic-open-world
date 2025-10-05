import torch
from torch import nn


class TextFeatureExtractor(nn.Module):
    """Base for custom text feature extractors."""

    @classmethod
    def build_from_config(
        cls, cfg_str: str, pretrained: bool, kwargs
    ) -> "TextFeatureExtractor":
        """
        Builds the FeatureExtractor based on a config string.
        """
        raise NotImplementedError(
            f"{cls.__name__} should implement a `build` method."
        )

    def forward_embed(
        self, txt: list[str]
    ) -> tuple[torch.Tensor, torch.BoolTensor]:
        """
        Returns text features([batch, embed_dim]) and mask([batch]). Mask: False -> padding.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement a `forward` method."
        )

    def forward_pool(
        self, txt: list[str]
    ) -> tuple[torch.Tensor, torch.BoolTensor]:
        """
        Returns text features([batch, num_tokens, embed_dim]) and mask([batch, num_tokens]). Mask: False -> padding.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement a `forward` method."
        )

    def get_channels(self) -> int:
        """
        Returns number of output channels.
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
