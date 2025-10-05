import torch
from torch import nn


class SineEmbedding(nn.Module):
    """
    Sine positional embedding.

    Supports 1-D or 2-D (or higher) dimensional embedding.
    """

    def __init__(self, emb_dim: int, temperature: float):
        super().__init__()
        self.emb_dim = emb_dim
        self.temperature = temperature

    def forward(self, pos):
        """Sine embedding forward pass.

        Args:
            positions (torch.Tensor): positions, [..., N, D], where N is number of positions and D is dimension of positions (e.g. 1 for txt, 2 for xy pos in images, 4 for bboxes)

        Returns:
            torch.Tensor: position embeddings [..., N, C], where C is self.emb_dim
        """
        D = pos.shape[-1]

        # generate time dimension for both x and y
        d = self.emb_dim // (2 * D)
        time = self.temperature ** torch.linspace(
            start=0, end=(d - 1) / d, steps=d, device=pos.device
        ).repeat_interleave(
            2
        )  # [C//D] - like this [t1, t1, t2, t2, ..., td, td]

        # outer prod with time
        pos_emb = 2 * torch.pi * pos[..., None] / time  # [..., N, D, C/D]
        pos_emb[..., 0::2].sin_()
        pos_emb[..., 1::2].cos_()

        return pos_emb.flatten(-2)  # [..., N, C]
