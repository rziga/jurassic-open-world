import torch
from torch import nn
from transformers.models.grounding_dino.modeling_grounding_dino import GroundingDinoModel


class HuggingFaceQuerySelectorAdapter(nn.Module):

    def __init__(self, cfg_str: str):
        super().__init__()
        self.model = GroundingDinoModel.from_pretrained(cfg_str)
    
    def _generate_encoder_output_proposals(self, enc_output, padding_mask, spatial_shapes):
        batch_size = enc_output.shape[0]
        proposals = []
        current_position = 0
        for level, (height, width) in enumerate(spatial_shapes):
            mask_flatten_ = padding_mask[:, current_position : (current_position + height * width)]
            mask_flatten_ = mask_flatten_.view(batch_size, height, width, 1)
            valid_height = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_width = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, height - 1, height, dtype=torch.float32, device=enc_output.device),
                torch.linspace(0, width - 1, width, dtype=torch.float32, device=enc_output.device),
                indexing="ij",
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_width.unsqueeze(-1), valid_height.unsqueeze(-1)], 1).view(batch_size, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(batch_size, -1, -1, -1) + 0.5) / scale
            width_heigth = torch.ones_like(grid) * 0.05 * (2.0**level)
            proposal = torch.cat((grid, width_heigth), -1).view(batch_size, -1, 4)
            proposals.append(proposal)
            current_position += height * width

        output_proposals = torch.cat(proposals, 1)
        #output_proposals = torch.clamp(output_proposals, 0.01, 0.99)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))  # inverse sigmoid
        output_proposals = output_proposals.masked_fill(padding_mask.unsqueeze(-1), float("inf"))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))

        # assign each pixel as an object query
        object_query = enc_output
        object_query = object_query.masked_fill(padding_mask.unsqueeze(-1), float(0))
        object_query = object_query.masked_fill(~output_proposals_valid, float(0))
        object_query = self.model.enc_output_norm(self.model.enc_output(object_query))
        return object_query, output_proposals
    
    def forward(self, encoder_outputs, mask_flatten, spatial_shapes, text_token_mask):
        # encoder outputs is a tuple (img_features, text_features)

        object_query_embedding, output_proposals = self._generate_encoder_output_proposals(
            encoder_outputs[0], ~mask_flatten, spatial_shapes
        )

        # hack implementation as in two-stage Deformable DETR
        # apply a detection head to each pixel (A.4 in paper)
        # linear projection for bounding box binary classification (i.e. foreground and background)
        enc_outputs_class = self.model.encoder_output_class_embed(
            object_query_embedding, encoder_outputs[1], text_token_mask
        )
        
        # 3-layer FFN to predict bounding boxes coordinates (bbox regression branch)
        delta_bbox = self.model.encoder_output_bbox_embed(object_query_embedding)
        enc_outputs_coord_logits = delta_bbox + output_proposals

        # only keep top scoring `config.num_queries` proposals
        topk = self.model.config.num_queries
        topk_logits = enc_outputs_class.max(-1)[0]
        topk_proposals = torch.topk(topk_logits, topk, dim=1)[1]
        topk_coords_logits = torch.gather(
            enc_outputs_coord_logits, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
        )

        topk_coords_logits = topk_coords_logits.detach()
        reference_points = topk_coords_logits.sigmoid()
        init_reference_points = reference_points

        batch_size = mask_flatten.shape[0]
        query_embeds = self.model.query_position_embeddings.weight
        target = query_embeds.unsqueeze(0).repeat(batch_size, 1, 1)

        return target, init_reference_points
