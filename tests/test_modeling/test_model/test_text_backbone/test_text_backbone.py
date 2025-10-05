import pytest

import torch
from jurassic_open_world.modeling.model.text_backbone.legacy_text_backbone import LegacyTextBackbone
from jurassic_open_world.modeling.model.text_backbone.text_backbone import TextBackbone

from .huggingface_impl import HuggingFaceTextAdapter
from ....utils.convert_huggingface import copy_text_backbone_params
from ....utils.configs import get_simple_text_backbone_config, get_simple_legacy_text_backbone_config


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_forward(device):
    captions = [["mouse", "cat"], ["dog"]]

    model = TextBackbone(get_simple_text_backbone_config())

    model(captions)

def test_gt_legacy():
    good = HuggingFaceTextAdapter("IDEA-Research/grounding-dino-tiny").eval()
    test = LegacyTextBackbone(get_simple_legacy_text_backbone_config()).eval()
    copy_text_backbone_params(test, good.model)

    captions = [["mouse"]]

    x_good = good(captions)
    x_test = test(captions)
    torch.testing.assert_close(x_test["feat"], x_good)

def test_gt_new():
    good = HuggingFaceTextAdapter("IDEA-Research/grounding-dino-tiny").eval()
    test = TextBackbone(get_simple_text_backbone_config()).eval()
    copy_text_backbone_params(test, good.model)

    captions = [["mouse"]]
    test_captions = [["[CLS]", "mouse.", "[SEP]"]]

    x_good = good(captions)
    x_test = test(test_captions)
    torch.testing.assert_close(x_test["feat"], x_good)