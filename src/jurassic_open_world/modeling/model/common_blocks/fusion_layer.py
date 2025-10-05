import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import StochasticDepth


class FusionLayer(nn.Module):
    """
    Fusion layer

    Bi-multihead attention with skip connection and droppath (stochastic depth).
    """

    def __init__(
        self,
        emb_dim: int,
        att_dim: int,
        num_heads: int,
        attention_dropout: float,
        droppath: float,
    ):
        super().__init__()

        self.img_norm = nn.LayerNorm(emb_dim)
        self.txt_norm = nn.LayerNorm(emb_dim)

        self.bi_cross_att = BiMultiheadAttention(
            emb_dim, att_dim, num_heads, attention_dropout
        )

        self.img_droppath = StochasticDepth(droppath, mode="batch")
        self.txt_droppath = StochasticDepth(droppath, mode="batch")
        self.img_coef = nn.Parameter(torch.full((emb_dim,), 1e-4))
        self.txt_coef = nn.Parameter(torch.full((emb_dim,), 1e-4))

    def forward(self, img_feat, img_mask, txt_feat, txt_mask):
        """Fusion layer forward pass

        Args:
            img_feat (torch.Tensor): image features, [B, N_img, C]
            img_mask (torch.Tensor): image mask, [B, N_img], True -> padded
            txt_feat (torch.Tensor): text features, [B, N_txt, C]
            txt_mask (torch.Tensor): text mask, [B, N_txt], True -> padded

        Returns:
            tuple[torch.Tensor, torch.Tensor]: fused image features and text features
        """

        # norm the inputs
        img_feat = self.img_norm(img_feat)
        txt_feat = self.txt_norm(txt_feat)

        # bimulithead attention
        img_feat_fused, txt_feat_fused = self.bi_cross_att(
            img_feat, img_mask, txt_feat, txt_mask
        )

        # add skip + droppath
        img_feat = img_feat + self.img_droppath(self.img_coef * img_feat_fused)
        txt_feat = txt_feat + self.txt_droppath(self.txt_coef * txt_feat_fused)

        return img_feat, txt_feat


class BiMultiheadAttention(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        att_dim: int,
        num_heads: int,
        attention_dropout: float,
    ):
        super().__init__()

        self.head_dim = att_dim // num_heads
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout

        self.img_in_proj = nn.Linear(emb_dim, 2 * att_dim)
        self.txt_in_proj = nn.Linear(emb_dim, 2 * att_dim)
        self.img_out_proj = nn.Linear(att_dim, emb_dim)
        self.txt_out_proj = nn.Linear(att_dim, emb_dim)

    def forward(self, img_feat, img_mask, txt_feat, txt_mask):
        """Bi-mulithead attention forward pass.

        Essentially img->txt cross attention and txt->img cross attention with shared parameters.

        Args:
            img_feat (torch.Tensor): image features, [B, N_img, C]
            img_mask (torch.Tensor): image mask, [B, N_img], True -> padded
            txt_feat (torch.Tensor): text features, [B, N_txt, C]
            txt_mask (torch.Tensor): text mask, [B, N_txt], True -> padded

        Returns:
            tuple(torch.Tensor, torch.Tensor): fused image features and text features
        """

        B, N_img, _ = img_feat.shape
        _, N_txt, _ = txt_feat.shape

        # [B, N, 2*E] -> view(B, N, 2, NH, HE) -> permute(2, 0, 3, 1, 4) -> [2, B, NH, N, HE]
        img_q, img_v = (
            self.img_in_proj(img_feat)
            .reshape(B, N_img, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )
        txt_q, txt_v = (
            self.txt_in_proj(txt_feat)
            .reshape(B, N_txt, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )

        img_att = F.scaled_dot_product_attention(
            img_q,
            txt_v,
            txt_q,
            attn_mask=~txt_mask[:, None, None, :],
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        txt_att = F.scaled_dot_product_attention(
            txt_v,
            img_q,
            img_v,
            attn_mask=~img_mask[:, None, None, :],
            dropout_p=self.attention_dropout if self.training else 0.0,
        )

        # out project
        img_out = self.img_out_proj(
            img_att.permute(0, 2, 1, 3).reshape(B, N_img, -1)
        )
        txt_out = self.txt_out_proj(
            txt_att.permute(0, 2, 1, 3).reshape(B, N_txt, -1)
        )

        return img_out, txt_out
