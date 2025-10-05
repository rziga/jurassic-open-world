# type: ignore
from warnings import warn
from itertools import chain

import torch

# NOTE: Extremely hacky way of parameter conversion.
#       If any of the modules parameter assignment order changes, everything in here breaks... LOL

def copy_checking(source_ignore_keys=[], target_ignore_keys=[], strict_target=True, strict_source=False):

    def decorator(copy_fcn):
        def wrapped(target: torch.nn.Module, source: torch.nn.Module):
            out = copy_fcn(target, source)
            for name, p in target.named_parameters():
                if name in target_ignore_keys:
                    continue
                if not getattr(p, "_copy_mark", False):
                    warn(f"parameter {name} in target module {target._get_name()} was not copied!")
                if strict_target:
                    assert getattr(p, "_copy_mark", False), f"parameter {name} in target module {target._get_name()} was not copied!"
            for name, p in source.named_parameters():
                if name in source_ignore_keys:
                    continue
                if not getattr(p, "_copy_mark", False):
                    warn(f"parameter {name} in source module {target._get_name()} was not copied!")
                if strict_source:
                    assert getattr(p, "_copy_mark", False), f"parameter {name} in source module {target._get_name()} was not copied!"
            return out
        return wrapped
    
    return decorator

def mark_param(param):
    param._copy_mark = True

def copy_all_params(target_params, source_params):
    for target_param, source_param in zip(target_params, source_params):
        target_param.data[:] = source_param
        mark_param(target_param)
        mark_param(source_param)

def copy_concat_params(target_param, source_params):
    target_param.data[:] = torch.cat(source_params, dim=0)
    mark_param(target_param)
    for source_param in source_params:
        mark_param(source_param)

#################
# Image Backbone
#################

from transformers.models.grounding_dino.modeling_grounding_dino import GroundingDinoModel
from jurassic_open_world.modeling.model.image_backbone.image_backbone import ImageBackbone

@copy_checking()
def copy_image_backbone_params(target: ImageBackbone, source: GroundingDinoModel):
    copy_all_params(target.model.parameters(), source.backbone.parameters())
    copy_all_params([target.level_emb], [source.level_embed])
    copy_all_params(target.input_projs.parameters(), source.input_proj_vision[:3].parameters())
    copy_all_params(target.extra_input_down_projs.parameters(), source.input_proj_vision[3:].parameters())

################
# Text Backbone
################

from transformers.models.grounding_dino.modeling_grounding_dino import GroundingDinoModel
from jurassic_open_world.modeling.model.text_backbone.text_backbone import TextBackbone

@copy_checking()
def copy_text_backbone_params(target: TextBackbone, source: GroundingDinoModel):
    copy_all_params(target.model.parameters(), source.text_backbone.parameters())
    copy_all_params(target.input_proj.parameters(), source.text_projection.parameters())


######################
# Cross scale encoder
######################

from jurassic_open_world.modeling.model.common_blocks.fusion_layer import FusionLayer
from transformers.models.grounding_dino.modeling_grounding_dino import GroundingDinoFusionLayer

@copy_checking()
def copy_encoder_fusionlayer_params(target: FusionLayer, source: GroundingDinoFusionLayer):
    copy_all_params([target.img_coef], [source.vision_param])
    copy_all_params([target.txt_coef], [source.text_param])
    copy_all_params(target.img_norm.parameters(), source.layer_norm_vision.parameters())
    copy_all_params(target.txt_norm.parameters(), source.layer_norm_text.parameters())

    copy_concat_params(target.bi_cross_att.img_in_proj.weight, [
        source.attn.vision_proj.weight,
        source.attn.values_vision_proj.weight,
    ])
    copy_concat_params(target.bi_cross_att.img_in_proj.bias, [
        source.attn.vision_proj.bias,
        source.attn.values_vision_proj.bias,
    ])

    copy_concat_params(target.bi_cross_att.txt_in_proj.weight, [
        source.attn.values_text_proj.weight,
        source.attn.text_proj.weight,
    ])
    copy_concat_params(target.bi_cross_att.txt_in_proj.bias, [
        source.attn.values_text_proj.bias,
        source.attn.text_proj.bias,
    ])

    copy_all_params(target.bi_cross_att.img_out_proj.parameters(), source.attn.out_vision_proj.parameters())
    copy_all_params(target.bi_cross_att.txt_out_proj.parameters(), source.attn.out_text_proj.parameters())

from transformers.models.grounding_dino.modeling_grounding_dino import GroundingDinoEncoderLayer
from jurassic_open_world.modeling.model.encoder.cross_scale.encoder_layer import CrossScaleEncoderLayer

@copy_checking()
def copy_encoder_layer_params(target: CrossScaleEncoderLayer, source: GroundingDinoEncoderLayer):
    # copy fusion layer params
    copy_encoder_fusionlayer_params(target.fusion, source.fusion_layer)
    
    # copy txt self att params
    copy_concat_params(target.txt_self_att.self_att.in_proj_weight, [
        source.text_enhancer_layer.self_attn.query.weight,
        source.text_enhancer_layer.self_attn.key.weight,
        source.text_enhancer_layer.self_attn.value.weight
    ])
    copy_concat_params(target.txt_self_att.self_att.in_proj_bias, [
        source.text_enhancer_layer.self_attn.query.bias,
        source.text_enhancer_layer.self_attn.key.bias,
        source.text_enhancer_layer.self_attn.value.bias
    ])
    copy_all_params(
        target.txt_self_att.self_att.out_proj.parameters(), 
        source.text_enhancer_layer.self_attn.out_proj.parameters(),
    )
    copy_all_params(
        target.txt_self_att.skip.norm.parameters(),
        source.text_enhancer_layer.layer_norm_before.parameters(),
    )

    # copy txt ffn params
    copy_all_params(
        target.txt_ffn.ffn.parameters(),
        chain(source.text_enhancer_layer.fc1.parameters(), source.text_enhancer_layer.fc2.parameters()),
    )
    copy_all_params(
        target.txt_ffn.skip.norm.parameters(),
        source.text_enhancer_layer.layer_norm_after.parameters(),
    )

    # copy image attention self attention params
    copy_all_params(
        target.img_self_att.def_att.parameters(),
        source.deformable_layer.self_attn.parameters(),
    )
    copy_all_params(
        target.img_self_att.skip.norm.parameters(),
        source.deformable_layer.self_attn_layer_norm.parameters(),
    )
    
    # copy image ffn params
    copy_all_params(
        target.img_ffn.ffn.parameters(),
        chain(source.deformable_layer.fc1.parameters(), source.deformable_layer.fc2.parameters())
    )
    copy_all_params(
        target.img_ffn.skip.norm.parameters(),
        source.deformable_layer.final_layer_norm.parameters(),
    )
    

from transformers.models.grounding_dino.modeling_grounding_dino import GroundingDinoEncoder
from jurassic_open_world.modeling.model.encoder.cross_scale.encoder import CrossScaleEncoder

@copy_checking()
def copy_encoder_params(target: CrossScaleEncoder, source: GroundingDinoEncoder):
    for target_layer, source_layer in zip(target.layers, source.layers):
        copy_encoder_layer_params(target_layer, source_layer)

####################
# Efficient encoder
####################

from transformers.models.omdet_turbo.modeling_omdet_turbo import OmDetTurboConvNormLayer
from jurassic_open_world.modeling.model.encoder.efficient.cross_scale_fuser import ConvNormBlock

@copy_checking()
def copy_convnormblock_params(target: ConvNormBlock, source: OmDetTurboConvNormLayer):
    copy_all_params(target.parameters(), source.parameters())

from transformers.models.omdet_turbo.modeling_omdet_turbo import OmDetTurboRepVggBlock
from jurassic_open_world.modeling.model.encoder.efficient.cross_scale_fuser import VGGRepBlock

@copy_checking()
def copy_vggrepblock_params(target: VGGRepBlock, source: OmDetTurboRepVggBlock):
    copy_all_params(target.parameters(), source.parameters())

from transformers.models.omdet_turbo.modeling_omdet_turbo import OmDetTurboCSPRepLayer
from jurassic_open_world.modeling.model.encoder.efficient.cross_scale_fuser import CSPRepBlock

@copy_checking()
def copy_csprepblock_params(target: CSPRepBlock, source: OmDetTurboCSPRepLayer):
    copy_all_params(target.main[0].parameters(), source.conv1.parameters())
    copy_all_params(target.main[1:].parameters(), source.bottlenecks.parameters())
    copy_all_params(target.skip.parameters(), source.conv2.parameters())

##################################
# Language guided query selection
##################################

from transformers.models.grounding_dino.modeling_grounding_dino import GroundingDinoModel
from jurassic_open_world.modeling.model.query_selector.query_selector import QuerySelector

@copy_checking(
    target_ignore_keys=[
        "cls_embed.txt_norm.weight", "cls_embed.txt_norm.bias",
    ]
)
def copy_query_selection_params(target: QuerySelector, source: GroundingDinoModel):
    copy_all_params(target.input_proj.parameters(), chain(source.enc_output.parameters(), source.enc_output_norm.parameters()))
    copy_all_params([target.cls_init], source.query_position_embeddings.parameters())
    copy_all_params(target.bbox_update_embed.parameters(), source.encoder_output_bbox_embed.parameters())

#########################
# Cross modality decoder
#########################

from transformers.models.grounding_dino.modeling_grounding_dino import GroundingDinoDecoderLayer
from jurassic_open_world.modeling.model.decoder.decoder_layer import DecoderLayer

@copy_checking(
    target_ignore_keys=[        
        "bbox_pos_embed.1.0.weight", "bbox_pos_embed.1.0.bias",
        "bbox_pos_embed.1.2.weight", "bbox_pos_embed.1.2.bias",

        "bbox_update_embed.0.weight", "bbox_update_embed.0.bias",
        "bbox_update_embed.2.weight", "bbox_update_embed.2.bias",
        "bbox_update_embed.4.weight", "bbox_update_embed.4.bias",

        "norm.weight", "norm.bias",
    ]
)
def copy_decoder_layer_params(target: DecoderLayer, source: GroundingDinoDecoderLayer):
    copy_concat_params(target.self_att.self_att.in_proj_weight, [
        source.self_attn.query.weight,
        source.self_attn.key.weight,
        source.self_attn.value.weight,
    ])
    copy_concat_params(target.self_att.self_att.in_proj_bias, [
        source.self_attn.query.bias,
        source.self_attn.key.bias,
        source.self_attn.value.bias
    ])
    copy_all_params(target.self_att.self_att.out_proj.parameters(), source.self_attn.out_proj.parameters())
    copy_all_params(target.self_att.skip.norm.parameters(), source.self_attn_layer_norm.parameters())

    # text 2 query cross att
    copy_concat_params(target.txt_cross_att.cross_att.in_proj_weight, [
        source.encoder_attn_text.query.weight,
        source.encoder_attn_text.key.weight,
        source.encoder_attn_text.value.weight,
    ])
    copy_concat_params(target.txt_cross_att.cross_att.in_proj_bias, [
        source.encoder_attn_text.query.bias,
        source.encoder_attn_text.key.bias,
        source.encoder_attn_text.value.bias,
    ])
    copy_all_params(target.txt_cross_att.cross_att.out_proj.parameters(), source.encoder_attn_text.out_proj.parameters())
    copy_all_params(target.txt_cross_att.skip.norm.parameters(), source.encoder_attn_text_layer_norm.parameters())

    # image 2 query cross att
    copy_all_params(target.img_cross_att.def_att.parameters(), source.encoder_attn.parameters())
    copy_all_params(target.img_cross_att.skip.norm.parameters(), source.encoder_attn_layer_norm.parameters())

    # ffn
    copy_all_params(target.ffn.ffn.parameters(), [source.fc1.weight, source.fc1.bias, source.fc2.weight, source.fc2.bias])
    copy_all_params(target.ffn.skip.norm.parameters(), source.final_layer_norm.parameters())


from transformers.models.grounding_dino.modeling_grounding_dino import GroundingDinoDecoder
from jurassic_open_world.modeling.model.decoder.decoder import Decoder

@copy_checking()
def copy_decoder_params(target: Decoder, source: GroundingDinoDecoder):
    for target_layer, source_layer in zip(target.layers, source.layers):
        copy_decoder_layer_params(target_layer, source_layer)
    
    copy_all_params(target.layers[0].bbox_pos_embed.parameters(), source.reference_points_head.parameters())
    copy_all_params(target.layers[0].norm.parameters(), source.layer_norm.parameters())
    copy_all_params(target.layers[0].bbox_update_embed.parameters(), source.bbox_embed.parameters())


##############
# Whole model
##############

from transformers.models.grounding_dino.modeling_grounding_dino import GroundingDinoForObjectDetection
from jurassic_open_world.modeling.model.grounding_dino import GroundingDINO

@copy_checking(strict_target=True)
def copy_grounding_dino_params(target: GroundingDINO, source: GroundingDinoForObjectDetection):
    copy_image_backbone_params(target.img_backbone, source.model)
    copy_text_backbone_params(target.txt_backbone, source.model)
    copy_encoder_params(target.encoder, source.model.encoder)
    copy_query_selection_params(target.query_selector, source.model)
    copy_decoder_params(target.decoder, source.model.decoder)

    #copy_all_params(target.query_selector.cls_embed.txt_norm.parameters())
    #copy_all_params(target.decoder.layers[0].cls_embed.txt)

#########
# Config
#########

from transformers.models.grounding_dino.configuration_grounding_dino import GroundingDinoConfig
from jurassic_open_world.modeling.model.grounding_dino import GroundingDINOConfig

def convert_config(source: GroundingDinoConfig) -> GroundingDINOConfig:
    return GroundingDINOConfig(
            # image backbone
            img_backbone_provider="transformers_auto",
            img_backbone_cfg_str="microsoft/swin-tiny-patch4-window7-224",
            img_backbone_use_pretrained=True,
            img_backbone_kwargs={"out_features": ["stage2", "stage3", "stage4"]},
            img_backbone_fpn_levels=source.num_feature_levels,
            img_backbone_num_extra_up_projs=0,
            img_backbone_num_extra_down_projs=1,
            img_backbone_legacy_pos=True,
            img_mean=[0.485, 0.456, 0.406],
            img_std=[0.229, 0.224, 0.225],

            # text backbone
            txt_backbone_type="legacy",
            txt_backbone_provider="transformers_auto",
            txt_backbone_cfg_str="bert-base-uncased",
            txt_backbone_use_pretrained=True,
            txt_backbone_kwargs={"add_pooling_layer": False},
            txt_backbone_token_pooling=False,
            txt_backbone_max_text_len=source.max_text_len,

            # query selection
            num_queries=source.num_queries,

            # shared for encoder and decoder
            emb_dim=source.d_model,
            num_heads=source.num_attention_heads,
            num_points=source.encoder_n_points,
            ff_dim=source.encoder_ffn_dim,
            dropout=source.dropout,
            attention_dropout=source.attention_dropout,
            droppath=source.fusion_droppath,
            cls_emb_type="grounding_dino",

            # encoder
            enc_type="cross_scale",
            enc_num_layers=source.encoder_layers,
            enc_use_fusion=True,
            enc_use_txt_self_att=True,
            enc_emb_dim_fusion=source.encoder_ffn_dim//2,
            enc_ff_dim_txt=source.encoder_ffn_dim//2,
            enc_num_heads_txt=source.encoder_attention_heads//2,
            enc_num_heads_fusion=source.encoder_attention_heads//2,

            # decoder
            dec_num_layers=source.decoder_layers,
            dec_use_fusion=True,
            dec_share_bbox_head=True,
            dec_share_cls_head=True,
            dec_share_pos_head=True,
            dec_share_norm=True,
        )
