# type: ignore
import torch

from transformers import GroundingDinoConfig
from transformers.models.grounding_dino.modeling_grounding_dino import GroundingDinoDecoderLayer

from jurassic_open_world.modeling.model.decoder.decoder_layer import DecoderLayer

from ....utils.convert_huggingface import copy_decoder_layer_params
from ....utils.data_generation_utils import generate_random_even_data
from ....utils.configs import get_simple_decoder_layer_config


def test_gt():
    config = GroundingDinoConfig()
    config.dropout = 0.0
    config.fusion_dropout = 0.0
    config.fusion_droppath = 0.0
    config.attention_dropout = 0.0
    config.activation_dropout = 0.0
    good = GroundingDinoDecoderLayer(config)
    test = DecoderLayer(get_simple_decoder_layer_config())
    test.bbox_pos_embed[1][-1].weight.data *= 0
    test.bbox_pos_embed[1][-1].bias.data *= 0
    copy_decoder_layer_params(test, good)

    img_f, txt_f, query_f = generate_random_even_data(2, (16, 8, 4, 2), 16, config.num_queries, config.d_model)
    x_img, _, mask_img, _, shapes_img, valid_ratio_img = img_f.values()
    x_txt, _, mask_txt, _, _ = txt_f.values()
    x_query, bbox_query, _, _  = query_f.values()

    level_start_idx = [0] + shapes_img.prod(-1).cumsum(0).tolist()
    bbox_query_unnorm = (
        bbox_query[:, :, None, :]
        * valid_ratio_img.flip(-1).repeat(1, 1, 2)[:, None, :, :]
    ).sigmoid()
    pos_query = torch.zeros_like(x_query)
    text_encoder_attention_mask = mask_txt[:, None, None, :]
    text_encoder_attention_mask = text_encoder_attention_mask.repeat(
        1, config.decoder_attention_heads, config.num_queries, 1
    )

    (a, ) = good.forward(
        x_query, pos_query, bbox_query_unnorm,
        shapes_img, shapes_img, level_start_idx, x_img, ~mask_img,
        x_txt, ~text_encoder_attention_mask, None, None
    )
    _, b = test.forward(query_f, img_f, txt_f)

    assert torch.allclose(a, b["feat"], atol=1e-6)


def test_gt2():
    config = GroundingDinoConfig()
    config.dropout = 0.0
    config.fusion_dropout = 0.0
    config.fusion_droppath = 0.0
    config.attention_dropout = 0.0
    config.activation_dropout = 0.0
    good = GroundingDinoDecoderLayer(config)
    test = DecoderLayer(get_simple_decoder_layer_config())
    test.bbox_pos_embed[1][-1].weight.data *= 0
    test.bbox_pos_embed[1][-1].bias.data *= 0
    copy_decoder_layer_params(test, good)

    img_f, txt_f, query_f = generate_random_even_data(2, (16, 8, 4, 2), 16, config.num_queries, config.d_model)
    x_img, _, mask_img, _, shapes_img, valid_ratio_img = img_f.values()
    x_txt, _, mask_txt, _, _ = txt_f.values()
    x_query, bbox_query, mask_query, att_mask_query = query_f.values()

    level_start_idx = [0] + shapes_img.prod(-1).cumsum(0).tolist()
    bbox_query_unnorm = (
        bbox_query[:, :, None, :]
        * valid_ratio_img.flip(-1).repeat(1, 1, 2)[:, None, :, :]
    ).sigmoid()
    pos_query = torch.zeros_like(x_query)
    text_encoder_attention_mask = mask_txt[:, None, None, :]
    text_encoder_attention_mask = text_encoder_attention_mask.repeat(
        1, config.decoder_attention_heads, config.num_queries, 1
    )

    (a, ) = good.forward(
        x_query, pos_query, bbox_query_unnorm,
        shapes_img, shapes_img, level_start_idx, x_img, ~mask_img,
        x_txt, ~text_encoder_attention_mask, None, None
    )
    _, b = test.forward(query_f, img_f, txt_f)

    assert torch.allclose(a, b["feat"], atol=1e-6)
