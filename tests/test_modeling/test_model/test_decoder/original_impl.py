# type: ignore
import math

import torch
from torch import nn
from transformers.models.grounding_dino.modeling_grounding_dino import GroundingDinoForObjectDetection



class gt_DecoderAdapter(nn.Module):

    def __init__(self, model: GroundingDinoForObjectDetection):
        super().__init__()
        self.model = model

    def decoder(self,
            inputs_embeds,
            vision_encoder_hidden_states,
            vision_encoder_attention_mask=None,
            text_encoder_hidden_states=None,
            text_encoder_attention_mask=None,
            reference_points=None,
            spatial_shapes=None,
            level_start_index=None,
            valid_ratios=None,
            self_attn_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        ):
        output = inputs_embeds

        intermediate = []
        reference_points = reference_points # 900, B, 4
        ref_points = [reference_points]

        for layer_id, layer in enumerate(self.model.model.decoder.layers):

            if reference_points.shape[-1] == 4: # TRUE
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
                )  # nq, bs, nlevel, 4
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * valid_ratios[None, :]
            query_sine_embed = get_sine_pos_embed(reference_points_input[:, :, 0, :], num_pos_feats=self.model.model.decoder.config.d_model // 2)

            # conditional query
            raw_query_pos = self.model.model.decoder.reference_points_head(query_sine_embed)  # nq, bs, 256
            query_pos = 1 * raw_query_pos # == 1 * raw_query_pos
            # if os.environ.get("SHILONG_AMP_INFNAN_DEBUG") == '1':
            #     if query_pos.isnan().any() | query_pos.isinf().any():
            #         import ipdb; ipdb.set_trace()

            (output, ) = layer.forward(
                hidden_states=output,
                position_embeddings=query_pos,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,
                spatial_shapes_list=spatial_shapes,
                level_start_index=level_start_index,
                vision_encoder_hidden_states=vision_encoder_hidden_states,
                vision_encoder_attention_mask=vision_encoder_attention_mask,
                text_encoder_hidden_states=text_encoder_hidden_states,
                text_encoder_attention_mask=text_encoder_attention_mask,
                self_attn_mask=self_attn_mask,
                output_attentions=output_attentions,
            )

             # iter update
            if self.model.model.decoder.bbox_embed is not None:
                # box_holder = self.bbox_embed(output)
                # box_holder[..., :self.query_dim] += inverse_sigmoid(reference_points)
                # new_reference_points = box_holder[..., :self.query_dim].sigmoid()

                reference_before_sigmoid = torch.logit(reference_points, eps=1e-5)
                delta_unsig = self.model.model.decoder.bbox_embed[layer_id](output)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()

                reference_points = new_reference_points.detach()
                # if layer_id != self.num_layers - 1:
                ref_points.append(new_reference_points)

            intermediate.append(self.model.model.decoder.layer_norm(output))

        return intermediate, ref_points

    def forward(
            self,
            query_f, img_f, txt_f
        ):

        x_img, _, mask_img, _, shapes_img, valid_ratio_img = img_f.values()
        x_txt, _, mask_txt, _, _ = txt_f.values()
        x_query, bbox_query, _, _ = query_f.values()
        level_start_idx = [0] + shapes_img.prod(-1).cumsum(0).tolist()
        bbox_query = bbox_query.sigmoid()
        mask_img = ~mask_img
        mask_txt = ~mask_txt

        hs, reference = self.decoder(
            x_query,
            x_img, mask_img,
            x_txt, mask_txt,
            bbox_query,
            shapes_img, level_start_idx, valid_ratio_img,
            None, None, None
        )

        outputs = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
            zip(reference[:-1], self.model.model.decoder.bbox_embed, hs)
        ):
            # bbox update
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + torch.logit(layer_ref_sig, eps=1e-5)
            output_bbox = layer_outputs_unsig.sigmoid()

            # class predct
            output_class = self.model.class_embed[dec_lid](
                vision_hidden_state=hs[dec_lid],
                text_hidden_state=x_txt,
                text_token_mask=mask_txt,
            )

            outputs.append((output_class, output_bbox))

        return outputs
    
def get_sine_pos_embed(
    pos_tensor: torch.Tensor, num_pos_feats: int = 128, temperature: int = 10000, exchange_xy: bool = True
):
    """
    Generate sine position embeddings from a position tensor.

    Args:
        pos_tensor (torch.Tensor):
            Tensor containing positions. Shape: [..., n].
        num_pos_feats (`int`, *optional*, defaults to 128):
            Projected shape for each float in the tensor.
        temperature (`int`, *optional*, defaults to 10000):
            Temperature in the sine/cosine function.
        exchange_xy (`bool`, *optional*, defaults to `True`):
            Exchange pos x and pos y. For example, input tensor is [x,y], the results will be [pos(y), pos(x)].

    Returns:
        position_embeddings (torch.Tensor): shape: [..., n * hidden_size].
    """
    scale = 2 * math.pi
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)

    def sine_func(x: torch.Tensor):
        sin_x = x * scale / dim_t
        sin_x = torch.stack((sin_x[..., 0::2].sin(), sin_x[..., 1::2].cos()), dim=3).flatten(2)
        return sin_x

    pos_tensor = pos_tensor.split([1] * pos_tensor.shape[-1], dim=-1)
    position_embeddings = [sine_func(x) for x in pos_tensor]
    if exchange_xy:
        position_embeddings[0], position_embeddings[1] = position_embeddings[1], position_embeddings[0]
    position_embeddings = torch.cat(position_embeddings, dim=-1)
    return position_embeddings