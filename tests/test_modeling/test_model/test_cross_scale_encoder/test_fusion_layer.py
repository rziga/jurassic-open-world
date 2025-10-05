import torch

from transformers import GroundingDinoConfig
from transformers.models.grounding_dino.modeling_grounding_dino import GroundingDinoFusionLayer

from jurassic_open_world.modeling.model.common_blocks.fusion_layer import FusionLayer

from ....utils.convert_huggingface import copy_encoder_fusionlayer_params
from ....utils.data_generation_utils import generate_random_even_data


def test_gt():
    
    config = GroundingDinoConfig()
    config.dropout = 0.0
    config.fusion_dropout = 0.0
    config.fusion_droppath = 0.0
    config.attention_dropout = 0.0
    config.activation_dropout = 0.0
    good = GroundingDinoFusionLayer(config)
    test = FusionLayer(config.d_model, config.encoder_ffn_dim//2, config.num_attention_heads, config.attention_dropout, config.fusion_droppath)
    copy_encoder_fusionlayer_params(test, good)

    img_f, txt_f, _ = generate_random_even_data(2, (8, 4, 2), 16, 2, config.d_model)
    x_img, mask_img = img_f["feat"], img_f["mask"]
    x_txt, mask_txt = txt_f["feat"], txt_f["mask"]

    (a_img, _), (a_txt, _) = good(x_img, x_txt, mask_img, mask_txt)
    b_img, b_txt = test(x_img, mask_img, x_txt, mask_txt)

    assert torch.allclose(a_img, b_img, atol=1e-4)
    assert torch.allclose(a_txt, b_txt, atol=1e-4)

def test_gt_mask():
    
    config = GroundingDinoConfig()
    config.dropout = 0.0
    config.fusion_dropout = 0.0
    config.fusion_droppath = 0.0
    config.attention_dropout = 0.0
    config.activation_dropout = 0.0
    good = GroundingDinoFusionLayer(config)
    test = FusionLayer(config.d_model, config.encoder_ffn_dim//2, config.num_attention_heads, config.attention_dropout, config.fusion_droppath)
    copy_encoder_fusionlayer_params(test, good)

    img_f, txt_f, _ = generate_random_even_data(2, (8, 4, 2), 16, 2, config.d_model)
    x_img, mask_img = img_f["feat"], img_f["mask"]
    x_txt, mask_txt = txt_f["feat"], txt_f["mask"]

    mask_img.bernoulli_(0.1)
    mask_txt.bernoulli_(0.1)

    (a_img, _), (a_txt, _) = good(x_img, x_txt, mask_img, mask_txt)
    b_img, b_txt = test(x_img, mask_img, x_txt, mask_txt)

    assert torch.allclose(a_img, b_img, atol=1e-4)
    assert torch.allclose(a_txt, b_txt, atol=1e-4)
