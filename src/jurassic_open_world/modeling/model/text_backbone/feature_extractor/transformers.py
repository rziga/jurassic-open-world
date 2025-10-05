import torch
from transformers import (
    AutoConfig,
    AutoModelForTextEncoding,
    AutoTokenizer,
    CLIPTextConfig,
)

from .base import TextFeatureExtractor


class AutoModelTextFeatureExtractor(TextFeatureExtractor):
    def __init__(self, tokenizer, model):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model

    @classmethod
    def build_from_config(cls, cfg_str, pretrained, kwargs):
        cfg = AutoConfig.from_pretrained(cfg_str)
        tokenizer = AutoTokenizer.from_pretrained(
            cfg_str, clean_up_tokenization_spaces=True, use_fast=True
        )
        if pretrained:
            model = AutoModelForTextEncoding.from_pretrained(
                cfg_str, config=cfg, **kwargs
            ).train()
        else:
            model = AutoModelForTextEncoding.from_config(cfg, **kwargs).train()

        return cls(tokenizer, model)

    def forward_embed(self, txt: list[str]):
        inputs = self.tokenizer(
            txt,
            add_special_tokens=False,
            return_tensors="pt",
            padding="longest",
        ).to(self.model.device)
        embeddings = self.model(**inputs)["last_hidden_state"]  # [B, N, C]
        mask = inputs["attention_mask"].bool()  # [B, N]
        return embeddings, mask  # [B, N, C], [B, N]

    def forward_pool(self, txt):
        inputs = self.tokenizer(
            txt,
            add_special_tokens=True,
            return_tensors="pt",
            padding="longest",
        ).to(self.model.device)
        pooled = self.model(**inputs)["pooler_output"]  # [B, C]
        mask = torch.ones_like(pooled[:, 0], dtype=torch.bool)  # [B]
        return pooled[:, None, :], mask[:, None]  # [B, 1, C], [B, 1]

    def get_channels(self):
        return self.model.config.hidden_size  # TODO: this is kinda brittle

    def backbone_parameters(self):
        return self.model.parameters()

    def freeze_backbone(self):
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)


class CLIPTextFeatureExtractor(AutoModelTextFeatureExtractor):
    @classmethod
    def build_from_config(cls, cfg_str, pretrained, kwargs):
        cfg = CLIPTextConfig.from_pretrained(cfg_str)
        tokenizer = AutoTokenizer.from_pretrained(
            cfg_str, clean_up_tokenization_spaces=True, use_fast=True
        )
        if pretrained:
            model = AutoModelForTextEncoding.from_pretrained(
                cfg_str, config=cfg, **kwargs
            ).train()
        else:
            model = AutoModelForTextEncoding(cfg_str, config=cfg).train()  # type: ignore

        return cls(tokenizer, model)
