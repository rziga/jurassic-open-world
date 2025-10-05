# type: ignore
import pytest

import torch
from transformers.models.grounding_dino.modeling_grounding_dino import GroundingDinoForObjectDetection
from transformers.models.grounding_dino.configuration_grounding_dino import GroundingDinoConfig

from jurassic_open_world.modeling.model.decoder.decoder import Decoder

from .original_impl import gt_DecoderAdapter
from ....utils.convert_huggingface import copy_decoder_params
from ....utils.data_generation_utils import generate_random_even_data
from ....utils.configs import get_simple_decoder_config

def test_forward():
    cfg = get_simple_decoder_config()
    img_dims = [16//(2**i) for i in range(cfg.layer_cfg.num_levels)]
    img_feat, txt_feat, query_feat = generate_random_even_data(2, img_dims, 16, 8, cfg.layer_cfg.emb_dim) 

    model = Decoder(cfg)
    model(query_feat, img_feat, txt_feat)


@pytest.mark.parametrize(
    argnames=["grad_check_layer"],
    argvalues=[("bbox_emb",), ("decoder_layer_cls",)]
)
def test_gt_full(grad_check_layer):
    config = GroundingDinoConfig()
    config.dropout = 0.0
    config.fusion_dropout = 0.0
    config.fusion_droppath = 0.0
    config.attention_dropout = 0.0
    config.activation_dropout = 0.0
    config.decoder_layers = 3
    config.d_model = 64
    
    good = gt_DecoderAdapter(GroundingDinoForObjectDetection(config))

    test_config = get_simple_decoder_config()
    test_config.num_layers = config.decoder_layers
    test_config.layer_cfg.emb_dim = config.d_model
    test = Decoder(test_config)
    copy_decoder_params(test, good.model.model.decoder)
    good.eval(), test.eval()

    N_txt = 16
    img_f, txt_f, query_f = generate_random_even_data(1, (16, 8, 4, 2), N_txt, config.num_queries, config.d_model)
    txt_f["feat"] = torch.nn.functional.layer_norm(txt_f["feat"], (config.d_model,))

    a = good.forward(query_f, img_f, txt_f)
    b = test.forward(query_f, img_f, txt_f)

    assert all(torch.allclose(a_[..., :N_txt], b_["cls"], atol=1e-2) for (a_, _), b_ in zip(a, b))
    assert all(torch.allclose(a_, b_["bbox"], atol=1e-6) for (_, a_), b_ in zip(a, b))
    
    match grad_check_layer:
        
        case "bbox_emb":
            sum(a_.sum() for _, a_ in a).backward()
            sum(b_["bbox"].sum() for b_ in b).backward()
            assert torch.allclose(
                good.model.model.decoder.bbox_embed[0].layers[0].weight.grad,
                test.layers[0].bbox_update_embed[0].weight.grad
            )

        case "decoder_layer_cls":
            sum(a_.sum() for a_, _ in a).backward()
            sum(b_["cls"].sum() for b_ in b).backward()
            assert torch.allclose(
                good.model.model.decoder.layers[0].fc1.weight,
                test.layers[0].ffn.ffn[0].weight
            )
