# type: ignore
import pytest
import torch

from transformers.models.grounding_dino.modeling_grounding_dino import GroundingDinoMultiscaleDeformableAttention
from jurassic_open_world.modeling.model.common_blocks.msda import MultiscaleDeformableAttention


def test_forward():
    
    emb_dim = 256
    dims = (32, 16, 8, 4)

    att = MultiscaleDeformableAttention(emb_dim, 4, 8, 4)
    
    x = torch.cat([torch.rand(1, d*d, emb_dim) for d in dims], 1)
    ref = torch.rand(*x.shape[:-1], 2)
    mask = torch.ones(x.shape[:-1], dtype=torch.bool)
    spatial_shapes = torch.tensor([[d, d] for d in dims])
    valid_ratios = torch.ones(1, len(dims), 2)

    output = att.forward(
        x, ref,
        x, mask, spatial_shapes, valid_ratios,
    )
    assert all(s1 == s2 for s1, s2 in zip(output.shape, x.shape))


@pytest.mark.parametrize("dim", [2, 4])
def test_msda_gt(dim):
    
    emb_dim = 256
    dims = (32, 16, 8, 4)

    att = MultiscaleDeformableAttention(emb_dim, len(dims), 8, 4)
    class cfg:
        d_model = emb_dim
        num_feature_levels = len(dims)
        disable_custom_kernels = False
    gt_att = GroundingDinoMultiscaleDeformableAttention(cfg, 8, 4)
    
    # copy params
    att.load_state_dict({
        k
        .replace("value_proj", "img_proj")
        .replace("sampling_offsets", "sampling_offsets_proj")
        .replace("attention_weights", "attention_weights_proj"): v for k, v in gt_att.state_dict().items()})

    x = torch.cat([torch.rand(1, d*d, emb_dim) for d in dims], 1)
    ref = torch.rand(*x.shape[:-1], dim)
    ref_expand = ref[:, :, None, :].expand(-1, -1, len(dims), -1)
    mask = torch.zeros(x.shape[:-1], dtype=torch.bool)
    spatial_shapes = torch.tensor([[d, d] for d in dims])
    valid_ratios = torch.ones(1, len(dims), 2)
    
    output = att.forward(
        x, ref,
        x, mask, spatial_shapes, valid_ratios,
    )
    gt, _ = gt_att.forward(
        x, ~mask, x, None, None, 
        ref_expand, spatial_shapes, spatial_shapes
    )

    torch.testing.assert_close(output, gt)
