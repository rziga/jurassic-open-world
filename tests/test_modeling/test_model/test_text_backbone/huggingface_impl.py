# type: ignore
import torch
from torch import nn
from transformers.models.grounding_dino.modeling_grounding_dino import GroundingDinoForObjectDetection, generate_masks_with_special_tokens_and_transfer_map
from transformers.models.grounding_dino.processing_grounding_dino import GroundingDinoProcessor


class HuggingFaceTextAdapter(nn.Module):

    def __init__(self, cfg_str):
        super().__init__()
        self.processor = GroundingDinoProcessor.from_pretrained(cfg_str)
        self.model = GroundingDinoForObjectDetection.from_pretrained(cfg_str).model

    def forward(self, captions: list[list[str]]):
        # forward through preprocessor
        merged_captions = [". ".join(caps)+"." for caps in captions]
        inputs = self.processor(text=merged_captions, return_tensors="pt").to(self.model.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        token_type_ids = inputs.token_type_ids

        # actual model forward
        text_self_attention_masks, position_ids = generate_masks_with_special_tokens_and_transfer_map(input_ids)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        text_token_mask = attention_mask.bool()  # just to avoid renaming everywhere

        max_text_len = self.model.config.max_text_len
        if text_self_attention_masks.shape[1] > max_text_len:
            text_self_attention_masks = text_self_attention_masks[:, :max_text_len, :max_text_len]
            position_ids = position_ids[:, :max_text_len]
            input_ids = input_ids[:, :max_text_len]
            token_type_ids = token_type_ids[:, :max_text_len]
            text_token_mask = text_token_mask[:, :max_text_len]

        # Extract text features from text backbone
        text_outputs = self.model.text_backbone(
            input_ids, text_self_attention_masks, token_type_ids, position_ids, return_dict=True
        )
        text_features = text_outputs.last_hidden_state
        text_features = self.model.text_projection(text_features)

        return text_features
