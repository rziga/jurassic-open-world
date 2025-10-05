# type: ignore
import torch
from transformers.models.grounding_dino.modeling_grounding_dino import GroundingDinoEncoder, GroundingDinoConfig

from jurassic_open_world.modeling.model.encoder.cross_scale.encoder import CrossScaleEncoder

from ....utils.data_generation_utils import generate_random_even_data
from ....utils.convert_huggingface import copy_encoder_params
from ....utils.configs import get_simple_encoder_config

def test_forward():

    cfg = get_simple_encoder_config()
    img_dims = [16//(2**i) for i in range(cfg.layer_cfg.num_levels)]
    C = cfg.layer_cfg.emb_dim
    img_feat, txt_feat, _ = generate_random_even_data(2, img_dims, 2, 4, C)

    model = CrossScaleEncoder(cfg)
    output = model(img_feat, txt_feat)
    assert output is not None

def test_gt():
    cfg = GroundingDinoConfig()
    good_model = GroundingDinoEncoder(cfg).eval()
    test_config = get_simple_encoder_config()
    test_config.layer_cfg.emb_dim_fusion = 1024
    test_model = CrossScaleEncoder(test_config).eval()

    copy_encoder_params(test_model, good_model)

    img_dims = (64, 32, 16, 8)
    C = cfg.d_model
    img_feat, txt_feat, _ = generate_random_even_data(2, img_dims, 2, 900, C)
    level_start_idx = [0] + img_feat["shapes"].prod(-1).cumsum(0).tolist()

    good_out = good_model.forward(
        img_feat["feat"], img_feat["mask"], img_feat["pos"], img_feat["shapes"], img_feat["shapes"], level_start_idx, img_feat["valid_ratios"],
        txt_feat["feat"], txt_feat["mask"], txt_feat["pos"], txt_feat["att_mask"]
    )
    test_out = test_model.forward(img_feat, txt_feat)

    assert torch.allclose(good_out.last_hidden_state_vision, test_out[0]["feat"], atol=1e-3), "image feature mismatch"
    assert torch.allclose(good_out.last_hidden_state_text, test_out[1]["feat"], atol=1e-3), "text feature mismatch"
