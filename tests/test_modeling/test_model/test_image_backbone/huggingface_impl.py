# type: ignore
import torch
from torch import nn
from transformers.models.grounding_dino.modeling_grounding_dino import GroundingDinoConfig, GroundingDinoForObjectDetection

class HuggingFaceImageBackboneAdapter(nn.Module):

    def __init__(self, config: GroundingDinoConfig):
        super().__init__()
        self.model = GroundingDinoForObjectDetection(config).model

    def forward(self, img, mask):
        x, pos_embs = self.model.backbone(img, mask)
        
        feature_maps = []
        masks = []
        for level, (source, mask) in enumerate(x):
            feature_maps.append(self.model.input_proj_vision[level](source))
            masks.append(mask)

        if self.model.config.num_feature_levels > len(feature_maps):
            _len_sources = len(feature_maps)
            for level in range(_len_sources, self.model.config.num_feature_levels):
                if level == _len_sources:
                    source = self.model.input_proj_vision[level](x[-1][0])
                else:
                    source = self.model.input_proj_vision[level](feature_maps[-1])
                mask = nn.functional.interpolate(mask[None].float(), size=source.shape[-2:]).to(torch.bool)[0]
                pos_l = self.model.backbone.position_embedding(source, mask).to(source.dtype)
                feature_maps.append(source)
                masks.append(mask)
                pos_embs.append(pos_l)
        
        # Prepare encoder inputs (by flattening)
        source_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for level, (source, mask, pos_embed) in enumerate(zip(feature_maps, masks, pos_embs)):
            batch_size, num_channels, height, width = source.shape
            spatial_shape = (height, width)
            spatial_shapes.append(spatial_shape)
            source = source.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.model.level_embed[level].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            source_flatten.append(source)
            mask_flatten.append(mask)
        source_flatten = torch.cat(source_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=source_flatten.device)
        #level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.model.get_valid_ratio(m) for m in masks], 1)
        valid_ratios = valid_ratios.float()
        
        return source_flatten, lvl_pos_embed_flatten, mask_flatten, spatial_shapes, valid_ratios
