from torch import nn

from .position_embedding import SineEmbedding


class FFN(nn.Sequential):
    """Feed-forward net"""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        zero_init: bool = False,
    ):
        layers = []
        for i, o in zip([input_dim] + hidden_dims, hidden_dims + [output_dim]):
            layers += [nn.Linear(i, o), nn.ReLU(inplace=True)]
        # remove last activation to output logits
        layers.pop()

        # init last layer to 0s if zero initing
        if zero_init:
            nn.init.constant_(layers[-1].weight, 0)
            nn.init.constant_(layers[-1].bias, 0)

        super().__init__(*layers)


class BBoxPositionEmbedding(nn.Sequential):
    def __init__(self, emb_dim: int):
        super().__init__(
            SineEmbedding(2 * emb_dim, temperature=10_000),
            FFN(2 * emb_dim, [emb_dim], emb_dim),
        )


class BBoxUpdateEmbedding(FFN):
    def __init__(self, emb_dim: int, zero_init: bool):
        super().__init__(emb_dim, [emb_dim] * 2, 4, zero_init)
