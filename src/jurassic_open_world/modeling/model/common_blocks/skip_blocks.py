from torch import nn

from .msda import MultiscaleDeformableAttention


class SkipConnection(nn.Module):
    def __init__(self, emb_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, skip, update):
        return self.norm(skip + self.dropout(update))


class PositionModulatedDeformableAttentionBlock(nn.Module):
    def __init__(self, emb_dim, num_levels, num_heads, num_points, dropout):
        super().__init__()
        self.def_att = MultiscaleDeformableAttention(
            emb_dim, num_levels, num_heads, num_points
        )
        self.skip = SkipConnection(emb_dim, dropout)

    def forward(
        self,
        img_feat,
        img_shapes,
        img_mask,
        img_valid_ratios,
        query_feat,
        query_pos,
        query_points,
    ):
        skip = query_feat
        update = self.def_att(
            query_feat + query_pos,
            query_points,
            img_feat,
            img_shapes,
            img_mask,
            img_valid_ratios,
        )
        return self.skip(skip, update)


class PositionModulatedSelfAttentionBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout, att_dropout):
        super().__init__()
        self.num_heads = num_heads
        self.self_att = nn.MultiheadAttention(
            emb_dim, num_heads, att_dropout, batch_first=True
        )
        self.skip = SkipConnection(emb_dim, dropout)

    def forward(self, x, pos, mask, att_mask):
        if att_mask is not None:
            att_mask = att_mask.repeat_interleave(self.num_heads, 0)
        skip = x
        update, _ = self.self_att(
            query=x + pos,
            key=x + pos,
            value=x,
            key_padding_mask=mask,
            attn_mask=att_mask,
            need_weights=False,
        )
        return self.skip(skip, update)


class PositionModulatedCrossAttentionBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout, att_dropout):
        super().__init__()
        self.cross_att = nn.MultiheadAttention(
            emb_dim, num_heads, att_dropout, batch_first=True
        )
        self.skip = SkipConnection(emb_dim, dropout)

    def forward(self, x, x_pos, y, y_mask):
        skip = x
        update, _ = self.cross_att(
            query=x + x_pos,
            key=y,
            value=y,
            key_padding_mask=y_mask,
            need_weights=False,
        )
        return self.skip(skip, update)


class FFNBlock(nn.Module):
    def __init__(self, emb_dim, ff_dim, dropout):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, emb_dim),
        )
        self.skip = SkipConnection(emb_dim, dropout)

    def forward(self, x):
        return self.skip(x, self.ffn(x))
