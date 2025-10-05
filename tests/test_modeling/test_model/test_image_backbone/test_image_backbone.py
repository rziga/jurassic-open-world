# type: ignore
import torch
from transformers import GroundingDinoConfig

from jurassic_open_world.modeling.model.image_backbone.image_backbone import ImageBackbone

from .huggingface_impl import HuggingFaceImageBackboneAdapter
from ....utils.convert_huggingface import copy_image_backbone_params
from ....utils.configs import get_simple_image_backbone_config

def test_gt():
    config = GroundingDinoConfig()
    #config = AutoConfig.from_pretrained("IDEA-Research/grounding-dino-tiny")
    config.dropout = 0.0
    config.fusion_dropout = 0.0
    config.fusion_droppath = 0.0
    config.attention_dropout = 0.0
    config.activation_dropout = 0.0
    good = HuggingFaceImageBackboneAdapter(config).eval()
    test = ImageBackbone(get_simple_image_backbone_config()).eval()
    copy_image_backbone_params(test, good.model)

    img = torch.rand(2, 3, 256, 256)
    mask = torch.zeros_like(img[:, 0, :, :]).bool()

    a_f, a_pos, a_mask, a_shape, a_valid = good.forward(img, ~mask)
    b = test.forward(img, mask)

    assert torch.allclose(a_f, b["feat"], atol=1e-4)
    #assert torch.allclose(a_pos, b_pos, atol=1e-4)
    assert torch.allclose(~a_mask, b["mask"], atol=1e-4)
    assert torch.allclose(a_shape, b["shapes"], atol=1e-4)
    assert torch.allclose(a_valid, b["valid_ratios"], atol=1e-4)
