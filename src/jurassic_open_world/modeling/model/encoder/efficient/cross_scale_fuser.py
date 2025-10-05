import torch
from torch import nn
from torch.nn import functional as F


class CrosScaleFuser(nn.Module):
    """
    Cross scale fusion module.

    Introduced in RT-DETR.

    Top-to-bottom fusion + bottom-to-top fusion.
    """

    def __init__(self, emb_dim: int, num_feature_levels: int):
        super().__init__()
        self.num_steps = num_feature_levels - 1

        self.top_down_projs = nn.ModuleList(
            [
                ConvNormBlock(emb_dim, emb_dim, 1, activation=nn.GELU)
                for _ in range(self.num_steps)
            ]
        )
        self.top_down_blocks = nn.ModuleList(
            [
                CSPRepBlock(2 * emb_dim, emb_dim, 3)
                for _ in range(self.num_steps)
            ]
        )

        self.bottom_up_projs = nn.ModuleList(
            [
                ConvNormBlock(emb_dim, emb_dim, 3, 2, 1, activation=nn.GELU)
                for _ in range(self.num_steps)
            ]
        )
        self.bottom_up_blocks = nn.ModuleList(
            [
                CSPRepBlock(2 * emb_dim, emb_dim, 3)
                for _ in range(self.num_steps)
            ]
        )

    def forward(
        self, img_feat_levels: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Cross scale fusion from RT-DETR.

        Args:
            img_feat_levels (list[torch.Tensor]): image features for each level,
                list of [[B, C, H_l, W_l] for l in num_levels]

        Returns:
            list[torch.Tensor]: updated image features for each level
        """
        img_feat_levels = (
            img_feat_levels.copy()
        )  # just so we modify list only in scope

        for i, proj, block in zip(
            reversed(range(self.num_steps)),
            self.top_down_projs,
            self.top_down_blocks,
        ):
            top, bot = img_feat_levels[i + 1], img_feat_levels[i]

            top = proj(top)
            top = F.interpolate(top, bot.shape[-2:], mode="nearest")
            bot_fused = block(torch.cat([top, bot], dim=1))

            img_feat_levels[i] = bot_fused

        for i, (proj, block) in enumerate(
            zip(self.bottom_up_projs, self.bottom_up_blocks)
        ):
            top, bot = img_feat_levels[i + 1], img_feat_levels[i]

            bot = proj(bot)
            bot = F.interpolate(bot, top.shape[-2:], mode="nearest")
            top_fused = block(torch.cat([top, bot], dim=1))

            img_feat_levels[i + 1] = top_fused

        return img_feat_levels


class ConvNormBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | str = 0,
        activation: type[nn.Module] = nn.Identity,
        eps: float = 1e-5,
    ):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, eps),
            activation(),
        )


class VGGRepBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: type[nn.Module] = nn.Identity,
    ):
        super().__init__()
        self.main = ConvNormBlock(in_channels, out_channels, 3, padding="same")
        self.skip = ConvNormBlock(in_channels, out_channels, 1)
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(x) + self.skip(x))


class CSPRepBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int):
        super().__init__()
        # fmt: off
        self.main = nn.Sequential(*(
              [ConvNormBlock(in_channels, out_channels, 1, activation=nn.SiLU)]
            + [VGGRepBlock(out_channels, out_channels, activation=nn.SiLU) for _ in range(num_blocks)]
        ))
        # fmt: on
        self.skip = ConvNormBlock(
            in_channels, out_channels, 1, activation=nn.SiLU
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x) + self.skip(x)
