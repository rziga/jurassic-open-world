import pytest
import torch
from lightning.pytorch.utilities import move_data_to_device
from transformers.models.grounding_dino.configuration_grounding_dino import GroundingDinoConfig

from jurassic_open_world.modeling.model.query_selector.query_selector import QuerySelector

from .huggingface_impl import HuggingFaceQuerySelectorAdapter
from ....utils.data_generation_utils import generate_random_even_data
from ....utils.convert_huggingface import copy_query_selection_params
from ....utils.configs import get_simple_query_selector_config


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_forward(device):
    emb_dim = 256
    num_q = 300
    B = 2
    ntxt = 12
    dims = (16, 8, 4)
    img_f, txt_f, _ = generate_random_even_data(B, dims, ntxt, num_q, emb_dim)
    img_f = move_data_to_device(img_f, device)
    txt_f = move_data_to_device(txt_f, device)

    gt_cfg = GroundingDinoConfig.from_pretrained("IDEA-Research/grounding-dino-tiny")
    gt_cfg.num_queries = num_q

    test_cfg = get_simple_query_selector_config()
    test_cfg.num_queries = num_q
    test = QuerySelector(test_cfg).to(device)
    enc_out, query = test(img_f, txt_f)
    assert enc_out is not None and not any(t.isnan().any() for t in enc_out.values() if not isinstance(t, list))
    assert query is not None and not any(t.isinf().any() for t in query.values() if not isinstance(t, list))

def test_gt():
    
    emb_dim = 256
    num_q = 100
    B = 2
    ntxt = 12
    dims = (32, 16, 8, 4)
    img_f, txt_f, _ = generate_random_even_data(B, dims, ntxt, num_q, emb_dim)
    txt_f["feat"] = torch.nn.functional.layer_norm(txt_f["feat"], (emb_dim,))

    # init modules
    good = HuggingFaceQuerySelectorAdapter("IDEA-Research/grounding-dino-tiny")
    test = QuerySelector(get_simple_query_selector_config())

    # map parameters
    copy_query_selection_params(test, good.model)
    
    a = good.forward((img_f["feat"], txt_f["feat"]), ~img_f["mask"], img_f["shapes"], ~txt_f["mask"])
    _, b = test.forward(img_f, txt_f)

    assert torch.allclose(a[0], b["feat"], atol=1e-6), "cls mismatch"
    assert torch.allclose(a[1], b["bbox"].sigmoid(), atol=1e-6), "bbox mismatch"