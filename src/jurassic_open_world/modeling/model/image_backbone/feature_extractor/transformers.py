from transformers import (
    AutoBackbone,
    AutoConfig,
    CLIPVisionConfig,
    CLIPVisionModel,
    TimmBackbone,
    TimmBackboneConfig,
)

from .base import ImageFeatureExtractor


class AutoBackboneImageFeatureExtractor(ImageFeatureExtractor):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @classmethod
    def build_from_config(cls, cfg_str: str, pretrained: bool, kwargs):
        if pretrained:
            return cls(AutoBackbone.from_pretrained(cfg_str, **kwargs).train())
        return cls(
            AutoBackbone.from_config(
                AutoConfig.from_pretrained(cfg_str, **kwargs)
            ).train()
        )

    def forward(self, img):
        return self.model(img).feature_maps

    def get_channels(self):
        return self.model.channels

    def backbone_parameters(self):
        return self.model.parameters()

    def freeze_backbone(self):
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)


class TimmBackboneFeatureExtractor(ImageFeatureExtractor):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @classmethod
    def build_from_config(cls, cfg_str: str, pretrained: bool, kwargs):
        if pretrained:
            return cls(TimmBackbone.from_pretrained(cfg_str, **kwargs).train())
        cfg = TimmBackboneConfig.from_pretrained(cfg_str, **kwargs)
        return cls(TimmBackbone(cfg).train())

    def forward(self, img):
        return self.model(img).feature_maps

    def get_channels(self):
        return self.model.channels

    def backbone_parameters(self):
        return self.model.parameters()

    def freeze_backbone(self):
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)


class CLIPVisionFeatureExtractor(ImageFeatureExtractor):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @classmethod
    def build_from_config(cls, cfg_str, pretrained, kwargs):
        if pretrained:
            return cls(
                CLIPVisionModel.from_pretrained(cfg_str, **kwargs).train()
            )
        cfg = CLIPVisionConfig.from_pretrained(cfg_str, **kwargs)
        return cls(CLIPVisionModel(cfg).train())

    def forward(self, img):
        stride, channels = (
            self.model.config.patch_size,
            self.model.config.hidden_size,
        )
        img_h, img_w = img.shape[-2:]
        out_h, out_w = img_h // stride, img_w // stride

        out = self.model(img, interpolate_pos_encoding=True)
        out = out.last_hidden_state[:, 1:]
        out = out.mT.reshape(-1, channels, out_h, out_w)

        return [out]

    def get_channels(self):
        return [self.model.config.hidden_size]

    def backbone_parameters(self):
        return self.model.parameters()

    def freeze_backbone(self):
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
