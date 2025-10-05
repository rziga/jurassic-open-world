# type: ignore
import torch

from transformers import GroundingDinoConfig
from transformers.models.grounding_dino.modeling_grounding_dino import GroundingDinoEncoderLayer

from jurassic_open_world.modeling.model.encoder.cross_scale.encoder_layer import CrossScaleEncoderLayer

from ....utils.convert_huggingface import copy_encoder_layer_params
from ....utils.data_generation_utils import generate_random_even_data
from ....utils.configs import get_simple_encoder_layer_config


def test_forward():
    cfg = get_simple_encoder_layer_config()
    scales = [16//(2**i) for i in range(cfg.num_levels)]
    test = CrossScaleEncoderLayer(cfg)
    img_f, txt_f, _ = generate_random_even_data(2, scales, 16, 2, cfg.emb_dim)
    (b_img, *_), (b_txt, *_) = test.forward(img_f, txt_f)

def test_gt():
    
    config = GroundingDinoConfig()
    config.dropout = 0.0
    config.fusion_dropout = 0.0
    config.fusion_droppath = 0.0
    config.attention_dropout = 0.0
    config.activation_dropout = 0.0
    good = GroundingDinoEncoderLayer(config).eval()

    test_config = get_simple_encoder_layer_config()
    test_config.emb_dim_fusion = 1024
    test = CrossScaleEncoderLayer(test_config).eval()
    copy_encoder_layer_params(test, good)

    img_f, txt_f, _ = generate_random_even_data(2, (16, 8, 4, 2), 16, 2, config.d_model)
    x_img, pos_img, mask_img, coor_img, shapes_img, valid_ratio_img = img_f.values()
    x_txt, pos_txt, mask_txt, self_att_mask_txt, cap_ids = txt_f.values()

    coor_normed = coor_img[:, :, None, :] * valid_ratio_img[:, None, :, :]
    level_start_idx = [0] + shapes_img.prod(-1).cumsum(0).tolist()

    (a_img, a_txt), _ = good.forward(
        x_img, pos_img, shapes_img, shapes_img, level_start_idx, mask_img, coor_normed,
        x_txt, mask_txt, pos_txt, self_att_mask_txt, None
    )
    b_img, b_txt = test.forward(img_f, txt_f)

    assert torch.allclose(a_img, b_img["feat"], atol=1e-6)
    assert torch.allclose(a_txt, b_txt["feat"], atol=1e-6)
