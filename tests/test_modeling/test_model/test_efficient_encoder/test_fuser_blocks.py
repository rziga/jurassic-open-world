# type: ignore
import torch
from torch import nn

from ....utils.convert_huggingface import copy_convnormblock_params, copy_vggrepblock_params, copy_csprepblock_params

from transformers.models.omdet_turbo.modeling_omdet_turbo import OmDetTurboConfig, OmDetTurboConvNormLayer, OmDetTurboRepVggBlock, OmDetTurboCSPRepLayer
from jurassic_open_world.modeling.model.encoder.efficient.cross_scale_fuser import ConvNormBlock, VGGRepBlock, CSPRepBlock

def test_convnormblock_gt():
    config = OmDetTurboConfig()
    config.dropout = 0.0
    config.fusion_dropout = 0.0
    config.fusion_droppath = 0.0
    config.attention_dropout = 0.0
    config.activation_dropout = 0.0

    gt = OmDetTurboConvNormLayer(config, config.encoder_hidden_dim, config.encoder_hidden_dim, 3, 1, 1, None)
    test = ConvNormBlock(config.encoder_hidden_dim, config.encoder_hidden_dim, 3, 1, 1)
    copy_convnormblock_params(test, gt)

    x = torch.rand(2, config.encoder_hidden_dim, 224, 244)

    a = gt.forward(x)
    b = test.forward(x)

    assert torch.allclose(a, b)

def test_vggrepblock_gt():
    config = OmDetTurboConfig()
    config.dropout = 0.0
    config.fusion_dropout = 0.0
    config.fusion_droppath = 0.0
    config.attention_dropout = 0.0
    config.activation_dropout = 0.0

    gt = OmDetTurboRepVggBlock(config)
    test = VGGRepBlock(config.encoder_hidden_dim, config.encoder_hidden_dim, nn.SiLU)
    copy_vggrepblock_params(test, gt)

    x = torch.rand(2, config.encoder_hidden_dim, 224, 244)

    a = gt.forward(x)
    b = test.forward(x)

    assert torch.allclose(a, b)

def test_vggrep_block_gt():
    config = OmDetTurboConfig()
    config.dropout = 0.0
    config.fusion_dropout = 0.0
    config.fusion_droppath = 0.0
    config.attention_dropout = 0.0
    config.activation_dropout = 0.0

    gt = OmDetTurboCSPRepLayer(config)
    test = CSPRepBlock(2*config.encoder_hidden_dim, config.encoder_hidden_dim, 3)
    copy_csprepblock_params(test, gt)

    x = torch.rand(2, 2*config.encoder_hidden_dim, 224, 244)

    a = gt.forward(x)
    b = test.forward(x)

    assert torch.allclose(a, b)
