# type: ignore
import pytest
import requests
from PIL import Image

import torch
from transformers.models.grounding_dino.modeling_grounding_dino import GroundingDinoForObjectDetection
from transformers.models.grounding_dino.processing_grounding_dino import GroundingDinoProcessor

from jurassic_open_world.modeling.model.grounding_dino import GroundingDINO

from ....utils.convert_huggingface import copy_grounding_dino_params
from ....utils.configs import get_simple_config, get_simple_efficient_encoder_config, get_simple_legacy_text_backbone_config


@pytest.mark.parametrize("device,use_efficient", [("cpu", True), ("cuda", True), ("cpu", False), ("cuda", False)])
def test_forward(device, use_efficient):
    cfg = get_simple_config()
    if use_efficient:
        cfg.encoder_cfg = get_simple_efficient_encoder_config()
    model = GroundingDINO(cfg).to(device)

    img = torch.rand(2, 3, 256, 256, device=device)
    mask = torch.zeros_like(img[:, 0, :, :], dtype=torch.bool)
    captions = [["test", "123"], ["test hahaha"]]

    model.forward(img, mask, captions)

def test_gt():
    good_model = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").eval()
    good_processor = GroundingDinoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")

    test_config = get_simple_config()
    test_config.txt_backbone_cfg = get_simple_legacy_text_backbone_config()
    test_model = GroundingDINO(test_config).eval()
    copy_grounding_dino_params(test_model, good_model)

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    img = Image.open(requests.get(url, stream=True).raw)
    captions = [["cat", "remote"]]
    inputs = good_processor(img, [". ".join(c)+"." for c in captions], return_tensors="pt")

    with torch.no_grad():
        good_out = good_model.forward(**inputs)
        test_out = test_model.forward(inputs.pixel_values, inputs.pixel_mask.bool().logical_not(), captions)

    # TODO: the outputs still do not match completely :/
    torch.testing.assert_close(good_out.pred_boxes[0, :4], test_out["outputs"]["bbox"][0, :4], atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(good_out.logits[0, :4, :6], test_out["outputs"]["cls"][0, :4], atol=1e-1, rtol=1e-2)
