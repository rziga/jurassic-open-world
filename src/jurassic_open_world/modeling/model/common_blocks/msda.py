import torch
from torch import nn
from torch.nn import functional as F


try:
    import msda_triton

    def multiscale_deformable_attention(
        img: torch.Tensor,
        img_shapes: torch.Tensor,
        sampling_points: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> torch.Tensor:
        return msda_triton.multiscale_deformable_attention(
            img,
            img_shapes,
            sampling_points,
            attention_weights,
            padding_mode="zeros",
            align_corners=False,
        ).flatten(-2)

except ModuleNotFoundError:

    def multiscale_deformable_attention(
        img: torch.Tensor,
        img_shapes: torch.Tensor,
        sampling_points: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> torch.Tensor:
        B, _, H, C = img.shape
        B, N, H, _, P, _ = sampling_points.shape

        # split the image into levels
        img_levels = img.split_with_sizes(img_shapes.prod(-1).tolist(), dim=1)

        # normalize points to from [0, 1] to [-1, 1]
        sampling_points = 2 * sampling_points - 1

        samples = []
        for img_level, points_level, (h, w) in zip(
            img_levels, sampling_points.unbind(-3), img_shapes
        ):
            # reshape for sampling
            img_level = (
                img_level.permute(
                    0, 2, 3, 1
                ).reshape(  # [B, I, H, C]  # [B, H, C, I]
                    B * H, C, int(h), int(w)
                )  # [B*H, C, H, W]
            )
            points_level = (
                points_level.permute(
                    0, 2, 1, 3, 4
                ).reshape(  # [B, N, H, P, 2]  # [B, H, N, P, 2]
                    B * H, N, P, 2
                )  # [B*H, N, P, 2]
            )

            # sample
            samples_level = F.grid_sample(
                img_level,
                points_level,
                mode="bilinear",
                align_corners=False,
                padding_mode="zeros",
            )
            samples_level = (
                samples_level.reshape(
                    B, H, C, N, P
                ).permute(  # [B*H, C, N, P]  # [B, H, C, N, P]
                    0, 3, 1, 4, 2
                )  # [B, N, H, P, C]
            )
            samples.append(samples_level)

        # [B, N, H, L, P, C]
        samples = torch.stack(samples, dim=3)
        # [B, N, H*C]
        out = torch.sum(
            attention_weights[..., None] * samples, dim=(3, 4)
        ).flatten(-2)

        return out


class MultiscaleDeformableAttention(nn.Module):
    """
    Multiscale deformable attention as proposed in Deformable DETR.
    """

    def __init__(
        self, emb_dim: int, num_levels: int, num_heads: int, num_points: int
    ):
        super().__init__()

        self.emb_dim = emb_dim
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points

        self.sampling_offsets_proj = nn.Linear(
            emb_dim, num_heads * self.num_levels * num_points * 2
        )
        self.attention_weights_proj = nn.Linear(
            emb_dim, num_heads * self.num_levels * num_points
        )
        self.img_proj = nn.Linear(emb_dim, emb_dim)
        self.output_proj = nn.Linear(emb_dim, emb_dim)

        # init parameters
        # fmt: off
        nn.init.zeros_(self.attention_weights_proj.weight)
        nn.init.zeros_(self.sampling_offsets_proj.weight)
        angles = torch.linspace(0, 2*torch.pi * (num_heads-1)/num_heads, num_heads) # [H] angles from 0 to 2Pi(n-1)/(n)
        circle = torch.stack([angles.sin(), angles.cos()], dim=-1) # [H, 2], NH points on a unit circle
        square = circle / circle.abs().max(-1, keepdim=True)[0] # [H, 2], project NH circle points to unit square
        squares = square[:, None, None, :].repeat(1, num_levels, num_points, 1) # [H, L, P, 2]
        squares *= torch.arange(1, num_points+1)[:, None] # [H, L, P, 2], expand the unit square up to n_points+1
        self.sampling_offsets_proj.bias.data = squares.reshape(-1)
        # fmt: on

    def forward(
        self,
        query_feat: torch.Tensor,
        query_points: torch.Tensor,
        img_feat: torch.Tensor,
        img_mask: torch.Tensor,
        img_shapes: torch.Tensor,
        img_valid_ratios: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MSDA forward pass.

        Args:
            img_feat (torch.Tensor): image pyramid features, [B, N_img, C]
            img_shapes (torch.Tensor): shapes of image pyramid levels, [L, 2]
            img_mask (torch.Tensor): image mask [B, N_img], True -> padded.
            img_valid_ratios (torch.tensor): image valid ratios [B, L, 2].
            query_feat (torch.Tensor): query latent features, [B, N_q, C] (or [B, N_img, C] in case of self MSDA)
            query_points (torch.Tensor): query sampling points, [B, N_q, 4] (or [B, N_img, 2] in case of self MSDA), in relative cxcywh

        Raises:
            ValueError: if img_shapes do not match img_feat
            ValueError: if last dim of query_points is neither 2 nor 4

        Returns:
            out (tuple[torch.Tensor, torch.Tensor]): sampled query features ([B, N_q, C]) and attention weights ([B, N_q, heads, L, points])
        """
        B, N_img, _ = img_feat.shape
        B, N_q, _ = query_feat.shape
        L, H, P = self.num_levels, self.num_heads, self.num_points
        C = self.emb_dim

        # project img
        img_feat = self.img_proj(img_feat)
        img_feat.masked_fill_(img_mask[..., None], 0.0)
        img_feat = img_feat.view(B, N_img, H, C // H)

        # project queries
        sampling_offsets = self.sampling_offsets_proj(query_feat).view(
            B, N_q, H, L, P, 2
        )
        attention_weights = self.attention_weights_proj(query_feat).view(
            B, N_q, H, L * P
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            B, N_q, H, L, P
        )

        last_dim = query_points.shape[-1]
        if last_dim == 2:
            # convert points to unpadded coordinates
            query_points_unpadded = (
                query_points[:, :, None, :]  # [B, N_img, 1, 2]
                * img_valid_ratios.flip(-1)[:, None, :, :]  # [B, 1, L, 2]
            )  # [B, N_img, L, 2]

            # [B, N, 1, L, 1, 2] + [B, N, H, L, P, 2] * [L, 1, 2] -> [B, N, H, L, P, 2]
            sampling_points = query_points_unpadded[
                :, :, None, :, None, :
            ] + sampling_offsets / img_shapes[:, None, :].flip(-1)
        elif last_dim == 4:
            # convert points to unpadded coordinates
            query_points_unpadded = (
                query_points[:, :, None, :]  # [B, N_img, 1, 4]
                * img_valid_ratios.flip(-1).repeat(1, 1, 2)[
                    :, None, :, :
                ]  # [B, 1, L, 4]
            )  # [B, N_img, L, 4]

            # [B, N, 1, L, 1, 2] + [B, N, H, L, P, 2] * [B, N, 1, L, 1, 2] -> [B, N, H, L, P, 2]
            sampling_points = query_points_unpadded[
                :, :, None, :, None, :2
            ] + sampling_offsets * query_points_unpadded[
                :, :, None, :, None, 2:
            ] / (2 * self.num_points)
        else:
            raise ValueError(
                f"`reference_points` should have the last dim either 2 or 4, but got {last_dim}."
            )

        output = multiscale_deformable_attention(
            img_feat, img_shapes, sampling_points, attention_weights
        )
        output = self.output_proj(output)

        return output
