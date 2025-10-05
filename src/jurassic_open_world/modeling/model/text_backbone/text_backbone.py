from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from .feature_extractor import get_feature_extractor
from ..common_blocks.position_embedding import SineEmbedding
from ....utils.config import BaseConfig
from ....utils.types import TextFeatures


@dataclass
class TextBackboneConfig(BaseConfig["TextBackbone"]):
    provider: Literal["transformers_auto", "transformers_clip", "openclip"]
    cfg_str: str
    use_pretrained: bool
    backbone_kwargs: dict
    emb_dim: int
    use_pooling: bool


class TextBackbone(nn.Module):
    """
    Text backbone.

    Supports more flexible backbones which include CLIP and so on.
    """

    def __init__(self, cfg: TextBackboneConfig):
        super().__init__()
        self.cfg = cfg

        # load pretrained tokenizer and model
        self.model = get_feature_extractor(
            cfg.provider, cfg.cfg_str, cfg.use_pretrained, cfg.backbone_kwargs
        )

        # input projection layer
        self.model_dim = self.model.get_channels()
        self.input_proj = nn.Linear(self.model_dim, cfg.emb_dim)

        # position embedding for text tokens
        self.pos_emb = SineEmbedding(cfg.emb_dim, temperature=10_000)

    def forward(self, txt_lists: list[list[str]]) -> TextFeatures:
        B = len(txt_lists)
        device = self.input_proj.weight.device

        # send through the text backbone
        batch_feat, batch_mask = [], []
        for txt_list in txt_lists:
            if self.cfg.use_pooling:
                # [N, 1, C], [N, 1] False -> padded
                feat, mask = self.model.forward_pool(txt_list)
            else:
                # [N, T, C], [N, T] False -> padded
                feat, mask = self.model.forward_embed(txt_list)
            batch_feat.append(feat)
            batch_mask.append(mask)

        # prepare output buffers
        # fmt: off
        L = max(mask.sum() for mask in batch_mask)
        C = self.model_dim
        out_feat = torch.zeros(B, L, C, device=device)  # [B, L, C]
        out_cap_ids = torch.full((B, L), -1, device=device)  # [B, L]
        out_pos_ids = torch.zeros(B, L, device=device, dtype=torch.long) # [B, L]
        out_mask = torch.ones(B, L, device=device, dtype=torch.bool) # [B, L], True -> padded
        out_att_mask = (
            torch.eye(L, device=device, dtype=torch.bool)
            .expand(B, L, L)
            .logical_not()
        )  # [B, L, L], True -> padded
        # fmt: on

        # fill them
        for i, (feat, mask) in enumerate(zip(batch_feat, batch_mask)):
            num_tokens = mask.sum()
            N, T = mask.shape  # prompts, tokens
            cap_ids = torch.arange(N, device=device)[:, None].expand(N, T)
            pos_ids = torch.arange(T, device=device)[None, :].expand(N, T)

            out_feat[i, :num_tokens] = feat[mask]
            out_cap_ids[i, :num_tokens] = cap_ids[mask]
            out_pos_ids[i, :num_tokens] = pos_ids[mask]
            out_mask[i, :num_tokens] = False
            out_att_mask[i, :num_tokens, :num_tokens] = ~(
                cap_ids[mask][:, None] == cap_ids[mask][None, :]
            )

        # project + embed positions
        out_feat = self.input_proj(out_feat)  # [B, L, C]
        out_pos = self.pos_emb(out_pos_ids[:, :, None])  # [B, L, C]

        return {
            "feat": out_feat,
            "pos": out_pos,
            "mask": out_mask,
            "att_mask": out_att_mask,
            "cap_ids": out_cap_ids,
        }

    def backbone_parameters(self):
        return self.model.backbone_parameters()

    def non_backbone_parameters(self):
        backbone_param_ids = {id(p) for p in self.backbone_parameters()}
        return (
            p for p in self.parameters() if id(p) not in backbone_param_ids
        )

    def freeze_backbone(self):
        self.model.freeze_backbone()


TextBackboneConfig._target_class = TextBackbone
