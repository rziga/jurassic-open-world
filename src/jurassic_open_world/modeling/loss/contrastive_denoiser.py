from dataclasses import dataclass

import torch
from torch import nn

from ..model.grounding_dino import GroundingDINO
from ...utils.config import BaseConfig
from ...utils.types import (
    ImageFeatures,
    ModelTarget,
    OutputFeatures,
    QueryFeatures,
    TextFeatures,
)


@dataclass
class ContrastiveDenoiserConfig(BaseConfig["ContrastiveDenoiser"]):
    box_noise_rate: float
    label_noise_rate: float
    num_cdn_queries: int
    emb_dim: int


class ContrastiveDenoiser(nn.Module):
    """Contrastive denoiser.
    **NOTE**: This implementation is a bit slower and less efficient than the original,
    since it does not process cdn queries in parallel with the model queries.
    """

    def __init__(self, cfg: ContrastiveDenoiserConfig):
        super().__init__()
        self.box_noise = cfg.box_noise_rate
        self.label_noise = cfg.box_noise_rate
        self.num_cdn_queries = cfg.num_cdn_queries
        self.cls_init = nn.Parameter(
            torch.empty(cfg.num_cdn_queries, cfg.emb_dim)
        )
        nn.init.normal_(self.cls_init)

    def forward(
        self,
        img: ImageFeatures,
        txt: TextFeatures,
        targets: ModelTarget,
        model: GroundingDINO,
    ) -> tuple[
        list[OutputFeatures],
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        # fmt: off
        # crop targets if there are more of them than supported by self.num_cdn_queries
        if targets["bbox"].shape[1] > (self.num_cdn_queries // 2):
            targets = targets.copy() # copy so we modify only in scope
            targets["bbox"] = targets["bbox"][:, :self.num_cdn_queries//2, :]
            targets["mask"] = targets["mask"][:, :self.num_cdn_queries//2]
            targets["cap_ids"] = targets["cap_ids"][:, :self.num_cdn_queries//2]

        B, N_gt = targets["cap_ids"].shape
        device = targets["cap_ids"].device
        C = self.cls_init.shape[-1]
        G = 2 * N_gt # group size is N_gt positive queries and N_gt negative queries
        num_groups = self.num_cdn_queries // G

        # --- generate cdn query inputs by noising the targets --- #

        cdn_cls  = torch.zeros(B, num_groups*G, C, device=device)
        cdn_bbox = torch.zeros(B, num_groups*G, 4, device=device)
        cdn_mask =  torch.ones(B, num_groups*G, device=device, dtype=torch.bool)
        cdn_att_mask = torch.ones(
            B, num_groups*G, num_groups*G, device=device, dtype=torch.bool
        )
        batch_idxs = []
        target_idxs = []
        pred_idxs = []
        for i in range(num_groups):
            group_slice = slice(i*G, (i+1)*G)

            # noise bboxes
            box_noise_scale = (
                  targets["bbox"][:, :, 2:].repeat(1, 1, 2)     # whwh
                * torch.tensor([0.5, 0.5, 1, 1], device=device) # cxcy will have 0.5 noise of wh
                * self.box_noise                                # noise hyperparam
            )
            cdn_bbox_g = torch.cat([
                targets["bbox"] + box_noise_scale * self.symmetric_noise(    torch.rand_like(targets["bbox"])),
                targets["bbox"] + box_noise_scale * self.symmetric_noise(1 + torch.rand_like(targets["bbox"]))
            ], dim=1) # [B, G, 4]
            cdn_bbox_g = torch.logit(torch.clamp(cdn_bbox_g, 0.01, 0.99))
            cdn_bbox[:, group_slice, :] = cdn_bbox_g

            # noise cls by shuffling
            shuffle_mask = torch.rand(B, G, device=device) > self.label_noise # [B, G]
            shuffled_idxs = torch.randperm(G) # [G]
            cdn_cls_g = (
                  ~shuffle_mask[:, :, None] * self.cls_init[:G] # [G, C]
                +  shuffle_mask[:, :, None] * self.cls_init[:G][shuffled_idxs] # [G, C]
            ).expand(B, -1, -1) # [B, G, C]
            cdn_cls[:, group_slice, :] = cdn_cls_g

            # masks
            cdn_mask[:, group_slice] = targets["mask"].repeat(1, 2)
            cdn_att_mask[:, group_slice, group_slice] = 0

            # matching indices
            batch_idxs.append(
                torch.arange(B, device=device) # [N_gt]
                [:, None].expand(B, N_gt) # [1, N_gt] -> [B, N_gt]
                [~targets["mask"]] # [B*N_gt]
            )
            pred_idxs.append(
                torch.arange(i*G, i*G + N_gt, device=device) # [N_gt]
                .expand(B, N_gt) # [B, N_gt]
                [~targets["mask"]] # [B*N_gt]
            )
            target_idxs.append(
                torch.arange(N_gt, device=device) # [N_gt]
                .expand(B, N_gt) # [B, N_gt]
                [~targets["mask"]] # [B*N_gt]
            )

        # fmt: on

        # input query features for the decoder
        cdn_query_feat: QueryFeatures = {
            "feat": cdn_cls,
            "bbox": cdn_bbox,
            "mask": cdn_mask,
            "att_mask": cdn_att_mask,
        }

        # send through the decoder
        cdn_outputs = model.decoder(cdn_query_feat, img, txt)

        # matching idxs
        pred_idxs = (torch.cat(batch_idxs), torch.cat(pred_idxs))
        target_idxs = (torch.cat(batch_idxs), torch.cat(target_idxs))

        return cdn_outputs, pred_idxs, target_idxs

    def symmetric_noise(self, noise):
        sign = 2 * (torch.rand_like(noise) > 0.5) - 1
        return noise * sign


ContrastiveDenoiserConfig._target_class = ContrastiveDenoiser
