from jurassic_open_world.modeling.model.encoder.efficient.encoder_layer import EfficientEncoderLayer
from jurassic_open_world.utils.image import split_fpn_levels

from ....utils.data_generation_utils import generate_random_even_data
from ....utils.configs import get_simple_efficient_encoder_layer_config


def test_forward():
    cfg = get_simple_efficient_encoder_layer_config()
    img_dims = [16//(2**i) for i in range(3)]
    img_feat, txt_feat, query_feat = generate_random_even_data(2, img_dims, 16, 8, cfg.emb_dim) 
    img_level = split_fpn_levels(img_feat)[0] 

    model = EfficientEncoderLayer(cfg)
    model(img_level, txt_feat)