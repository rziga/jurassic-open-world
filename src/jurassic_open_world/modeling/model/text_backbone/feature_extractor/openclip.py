import torch
from open_clip import create_model, get_tokenizer

from .base import TextFeatureExtractor


class OpenCLIPTextFeatureExtractor(TextFeatureExtractor):
    def __init__(self, tokenizer, model):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model

    @classmethod
    def build_from_config(cls, cfg_str, pretrained, kwargs):
        tokenizer = get_tokenizer(cfg_str)
        if pretrained:
            model = create_model(cfg_str, **kwargs)
            del model.visual  # A bit hacky
        else:
            raise NotImplementedError(
                f"{cls.__name__} does not support random weights currently."
            )

        return cls(tokenizer, model)

    def forward_embed(self, txt):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support non pooling embedding currently."
        )

    def forward_pool(self, txt):
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(txt).to(device)
        pooled = self.model.encode_text(inputs)  # [B, C]
        mask = torch.ones_like(pooled[:, 0], dtype=torch.bool)  # [B]
        return pooled[:, None, :], mask[:, None]  # [B, 1]

    def get_channels(self):
        if getattr(self.model, "text", None) is not None:
            return self.model.text.text_projection.shape[-1]
        if getattr(self.model, "transformer", None) is not None:
            return self.model.text_projection.shape[-1]
        raise NotImplementedError("Model not supported.")

    def backbone_parameters(self):
        return self.model.parameters()

    def freeze_backbone(self):
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
