from jurassic_open_world.modeling.model.encoder.efficient.encoder import EfficientEncoder

from ....utils.data_generation_utils import generate_random_even_data
from ....utils.configs import get_simple_efficient_encoder_config


def test_forward():
    cfg = get_simple_efficient_encoder_config()
    img_dims = [16//(2**i) for i in range(cfg.num_levels)]
    img_feat, txt_feat, query_feat = generate_random_even_data(2, img_dims, 16, 8, cfg.emb_dim) 

    model = EfficientEncoder(cfg)
    model(img_feat, txt_feat)