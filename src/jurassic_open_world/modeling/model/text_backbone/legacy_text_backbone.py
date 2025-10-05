from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch import nn

from .feature_extractor import get_feature_extractor
from ..common_blocks.position_embedding import SineEmbedding
from ....utils.config import BaseConfig
from ....utils.types import TextFeatures


@dataclass
class LegacyTextBackboneConfig(BaseConfig["LegacyTextBackbone"]):
    provider: Literal["transformers_auto", "transformers_clip", "openclip"]
    cfg_str: str
    use_pretrained: bool
    backbone_kwargs: dict
    emb_dim: int
    use_pooling: bool
    max_txt_len: Optional[int]


class LegacyTextBackbone(nn.Module):
    """
    Text Backbone for compat with original Grounding DINO weights.

    Downsides:
        - Adds a bunch of unused tokens for "." which makes downstream processing more expenisve
    """

    def __init__(self, cfg: LegacyTextBackboneConfig):
        super().__init__()
        self.cfg = cfg

        # load pretrained tokenizer and model
        self.model = get_feature_extractor(
            cfg.provider, cfg.cfg_str, cfg.use_pretrained, cfg.backbone_kwargs
        )

        # input projection layer
        model_dim = self.model.get_channels()
        self.input_proj = nn.Linear(model_dim, cfg.emb_dim)

        # position embedding for text tokens
        self.pos_emb = SineEmbedding(cfg.emb_dim, temperature=10_000)

        # separator handling
        self.sep_token = "."
        sep_id = self.model.tokenizer.convert_tokens_to_ids(self.sep_token)  # type: ignore
        self.special_ids = [sep_id] + self.model.tokenizer.all_special_ids  # type: ignore

    def forward(self, captions: list[list[str]]) -> TextFeatures:
        device = self.model.model.device

        # merge the captions for each image with "." inbetween
        sep = self.sep_token
        merged_captions = [sep.join(caps) + sep for caps in captions]

        # tokenize the merged captions
        inputs = self.model.tokenizer(
            merged_captions,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=True,
        )  # type: ignore
        inputs = inputs.to(device)
        mask = ~(
            inputs["attention_mask"].bool()
        )  # [B, L], invert because HF uses False-->masked

        # generate self attention mask and position ids
        self_att_mask, pos_ids, cap_ids = (
            self._generate_self_attention_mask_and_pos_ids(inputs["input_ids"])
        )  # [B, L, L], [B, L]
        inputs[
            "attention_mask"
        ] = ~self_att_mask  # invert because HF uses False-->masked
        inputs["position_ids"] = (
            pos_ids.long()
        )  # caption level pos ids to ensure caption permutation invariance

        # forward through the text backbone + project + embed positions
        features = self.model.model(
            **inputs
        ).last_hidden_state  # [B, L, C] # type: ignore
        features = self.input_proj(features)  # [B, L, C]
        pos = self.pos_emb(pos_ids[:, :, None])  # [B, L, C]

        return self._maybe_crop(
            {
                "feat": features,
                "pos": pos,
                "mask": mask,
                "att_mask": self_att_mask,
                "cap_ids": cap_ids,
            }
        )

    def _generate_self_attention_mask_and_pos_ids(self, input_ids):
        """
        generate self attention mask that prevents interactions between tokens in different captions
        e.g. "token1 token2 token3 [SEP] token1 token2 [SEP]" gets:
           [[False, False, False, False,  True,  True,  True],
            [False, False, False, False,  True,  True,  True],
            [False, False, False, False,  True,  True,  True],
            [False, False, False, False,  True,  True,  True],
            [ True,  True,  True,  True, False, False, False],
            [ True,  True,  True,  True, False, False, False],
            [ True,  True,  True,  True, False, False, False],]
            True --> Padded
        also generate pos_ids -- index of token within caption
        e.g. (same example as above) gets [0, 1, 2, 3, 0, 1, 2]

        also generate caption_ids -- index of caption to which the token belongs to
        e.g. (same example as above) gets [0, 0, 0, 0, 1, 1, 1]
        """

        B, L = input_ids.shape
        device = input_ids.device

        # prepare buffers
        attention_mask = torch.ones(B, L, L, dtype=torch.bool, device=device)
        pos_ids = torch.zeros(B, L, device=device)
        cap_ids = -1 * torch.ones(B, L, device=device)

        # generate special token mask
        sep_token_mask = torch.isin(
            input_ids, torch.tensor(self.special_ids, device=device)
        )

        # unpad blocks for captions
        attention_mask[:, range(L), range(L)] = False
        for b in range(B):
            sep_idxs = sep_token_mask[b].nonzero().squeeze(1)
            stops = (
                sep_idxs + 1
            ).tolist()  # +1 because "." at the end is included
            starts = [0] + stops
            for i, (start, stop) in enumerate(zip(starts, stops)):
                cap_len = int(stop - start)
                attention_mask[b, start:stop, start:stop] = False
                pos_ids[b, start:stop] = torch.arange(cap_len, device=device)
                cap_ids[b, start : stop - 1] = (
                    i - 1
                )  # NOTE: "." will not contribute to caption prob like in original, i-1 because we start with [CLS] token which is unused

        return attention_mask, pos_ids, cap_ids

    def backbone_parameters(self):
        return self.model.parameters()

    def non_backbone_parameters(self):
        return (
            p for n, p in self.named_parameters() if not n.startswith("model.")
        )

    def freeze_backbone(self):
        self.model.eval()
        for p in self.backbone_parameters():
            p.requires_grad_(False)

    def _maybe_crop(self, out: TextFeatures) -> TextFeatures:
        if self.cfg.max_txt_len is None:
            return out
        window = slice(self.cfg.max_txt_len)
        return {
            "feat": out["feat"][:, window],
            "pos": out["pos"][:, window],
            "mask": out["mask"][:, window],
            "att_mask": out["att_mask"][:, window, window],
            "cap_ids": out["cap_ids"][:, window],
        }


LegacyTextBackboneConfig._target_class = LegacyTextBackbone
