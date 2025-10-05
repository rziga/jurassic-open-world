import torch
from torch import nn


class ContrastiveClassEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        img_feat: torch.Tensor,
        img_mask: torch.Tensor,
        txt_feat: torch.Tensor,
        txt_mask: torch.Tensor,
    ):
        """
        Args:
            img_feat (torch.Tensor): image features, [B, N_img, C] (or query features in decoder, [B, N_q, C])
            img_mask (torch.Tensor): image mask, [B, N_img] (or query mask in decoder, [B, N_q]), True -> padded
            txt_feat (torch.Tensor): txt features, [B, N_txt, C]
            txt_mask (torch.Tensor): txt mask, [B, N_txt]

        Returns:
            torch.Tensor: image-text similarity logits, [B, N_img, N_txt] (or query-text similarty logits in decoder [B, N_q, N_txt])
        """
        # calculate outer dot product
        similarity_logits = img_feat @ txt_feat.mT  # [B, N_img, N_txt]

        # fill masked elements with -inf
        # NOTE: only txt_mask is used in original code
        similarity_logits.masked_fill_(txt_mask[:, None, :], -torch.inf)
        similarity_logits.masked_fill_(img_mask[:, :, None], -torch.inf)
        return similarity_logits


class ContrastiveBiasClassEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        img_feat: torch.Tensor,
        img_mask: torch.Tensor,
        txt_feat: torch.Tensor,
        txt_mask: torch.Tensor,
    ):
        """
        Args:
            img_feat (torch.Tensor): image features, [B, N_img, C] (or query features in decoder, [B, N_q, C])
            img_mask (torch.Tensor): image mask, [B, N_img] (or query mask in decoder, [B, N_q]), True -> padded
            txt_feat (torch.Tensor): txt features, [B, N_txt, C]
            txt_mask (torch.Tensor): txt mask, [B, N_txt]

        Returns:
            torch.Tensor: image-text similarity logits, [B, N_img, N_txt] (or query-text similarty logits in decoder [B, N_q, N_txt])
        """
        # calculate outer dot product
        similarity_logits = img_feat @ txt_feat.mT  # [B, N_img, N_txt]
        similarity_logits /= img_feat.shape[-1] ** 0.5
        similarity_logits += self.bias

        # fill masked elements with -inf
        similarity_logits.masked_fill_(txt_mask[:, None, :], -torch.inf)
        similarity_logits.masked_fill_(img_mask[:, :, None], -torch.inf)
        return similarity_logits
