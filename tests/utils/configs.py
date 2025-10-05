from jurassic_open_world.modeling.model.grounding_dino import GroundingDINOConfig
from jurassic_open_world.modeling.model.image_backbone.image_backbone import ImageBackboneConfig
from jurassic_open_world.modeling.model.text_backbone.text_backbone import TextBackboneConfig
from jurassic_open_world.modeling.model.text_backbone.legacy_text_backbone import LegacyTextBackboneConfig
from jurassic_open_world.modeling.model.encoder.efficient.encoder import EfficientEncoderConfig, EfficientEncoderLayerConfig
from jurassic_open_world.modeling.model.encoder.cross_scale.encoder import CrossScaleEncoderConfig, CrossScaleEncoderLayerConfig
from jurassic_open_world.modeling.model.query_selector.query_selector import QuerySelectorConfig
from jurassic_open_world.modeling.model.decoder.decoder import DecoderConfig, DecoderLayerConfig


def get_simple_image_backbone_config() -> ImageBackboneConfig:
    return ImageBackboneConfig(
        provider="transformers_auto",
        cfg_str="microsoft/swin-tiny-patch4-window7-224",
        use_pretrained=True,
        backbone_kwargs={"out_features": ["stage2", "stage3", "stage4"]},
        emb_dim=256,
        num_extra_up_projs=0,
        num_extra_down_projs=1,
        use_legacy_pos=True,
    )

def get_simple_text_backbone_config() -> TextBackboneConfig:
    return TextBackboneConfig(
        provider="transformers_auto",
        cfg_str="bert-base-uncased",
        use_pretrained=True,
        backbone_kwargs={"add_pooling_layer": False},
        emb_dim=256,
        use_pooling=False,
    )

def get_simple_legacy_text_backbone_config() -> LegacyTextBackboneConfig:
    return LegacyTextBackboneConfig(
        provider="transformers_auto",
        cfg_str="bert-base-uncased",
        use_pretrained=True,
        backbone_kwargs={"add_pooling_layer": False},
        emb_dim=256,
        use_pooling=False,
        max_txt_len=None,
    )

def get_simple_encoder_layer_config() -> CrossScaleEncoderLayerConfig:
    return CrossScaleEncoderLayerConfig(
            emb_dim=256,
            emb_dim_fusion=1024,
            ffn_dim=2048,
            ffn_dim_txt=1024,
            num_heads=8,
            num_heads_fusion=4,
            num_heads_txt=4,
            num_points=4,
            num_levels=4,
            dropout=0.0,
            attention_dropout=0.0,
            droppath=0.0,
            use_fusion=True,
            use_txt_self_att=True,
        )

def get_simple_encoder_config() -> CrossScaleEncoderConfig:
    return CrossScaleEncoderConfig(
        get_simple_encoder_layer_config(),
        num_layers=6,
    )

def get_simple_efficient_encoder_layer_config() -> EfficientEncoderLayerConfig:
    return EfficientEncoderLayerConfig(
        emb_dim=256,
        emb_dim_fusion=256,
        ffn_dim=1024,
        num_heads=8,
        num_heads_fusion=8,
        dropout=0,
        attention_dropout=0,
        droppath=0,
        use_fusion=True,
    )

def get_simple_efficient_encoder_config() -> EfficientEncoderConfig:
    return EfficientEncoderConfig(
        get_simple_efficient_encoder_layer_config(),
        num_layers=1,
        emb_dim=256,
        num_levels=3,
    )

def get_simple_query_selector_config() -> QuerySelectorConfig:
    return QuerySelectorConfig(
        num_queries=900,
        emb_dim=256,
        cls_emb_type="grounding_dino",
    )

def get_simple_decoder_layer_config() -> DecoderLayerConfig:
    return DecoderLayerConfig(
        emb_dim=256,
        ffn_dim=2048,
        num_heads=8,
        num_points=4,
        num_levels=4,
        dropout=0,
        attention_dropout=0,
        use_fusion=True,
        cls_emb_type="grounding_dino",
        use_legacy_pos=True,
    )

def get_simple_decoder_config() -> DecoderConfig:
    return DecoderConfig(
        get_simple_decoder_layer_config(),
        num_layers=6,
        share_cls_head=True,
        share_bbox_head=True,
        share_norm=True,
        share_pos_head=True,
    )

def get_simple_config():
    return GroundingDINOConfig(
        get_simple_image_backbone_config(),
        get_simple_text_backbone_config(),
        get_simple_encoder_config(),
        get_simple_query_selector_config(),
        get_simple_decoder_config(),
        img_mean=(0.0, 0.0, 0.0),
        img_std=(1.0, 1.0, 1.0),
    )
    dict(       # image backbone
            img_backbone_provider="transformers_auto",
            img_backbone_cfg_str="microsot/swin-tiny-patch4-window7-224",
            img_backbone_use_pretrained=True,
            img_backbone_kwargs={"out_features": ["stage2", "stage3", "stage4"]},
            img_backbone_num_extra_up_projs=0,
            img_backbone_num_extra_down_projs=1,
            img_backbone_fpn_levels=4,
            img_backbone_legacy_pos=True,
            img_mean=[0.0, 0.0, 0.0],
            img_std=[1.0, 1.0, 1.0],

            # text backbone
            txt_backbone_type="improved",
            txt_backbone_provider="transformers_auto",
            txt_backbone_cfg_str="bert-base-uncased",
            txt_backbone_use_pretrained=True,
            txt_backbone_kwargs={"add_pooling_layer": False},
            txt_backbone_token_pooling=False,
            txt_backbone_max_text_len=None,

            # query selection
            num_queries=300,

            # shared for encoder and decoder
            emb_dim=256,
            num_heads=8,
            num_points=4,
            ff_dim=1024,
            dropout=0.0,
            attention_dropout=0.0,
            droppath=0.0,
            cls_emb_type="grounding_dino",

            # encoder
            enc_type="efficient",
            enc_num_layers=6,
            enc_use_fusion=True,
            enc_use_txt_self_att=True,
            enc_emb_dim_fusion=256,
            enc_ff_dim_txt=1024,
            enc_num_heads_txt=8,
            enc_num_heads_fusion=8,

            # decoder
            dec_num_layers=6,
            dec_use_fusion=True,
            dec_share_bbox_head=True,
            dec_share_cls_head=True,
            dec_share_norm=True,
            dec_share_pos_head=True,
        )