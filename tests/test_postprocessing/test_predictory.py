# type: ignore
from itertools import product

import pytest
import torch
from torchvision.transforms.functional import to_pil_image
import numpy as np

from jurassic_open_world.modeling.model.grounding_dino import GroundingDINO
from jurassic_open_world.inference.predictor import GroundingDINOPredictor

from ..utils.configs import get_simple_config


@pytest.mark.parametrize(
    argnames=["processor_type","image_type","device"],
    argvalues=product(
        ["exclusive", "inclusive"],
        ["torch", "pil", "numpy"],
        ["cuda", "cpu"],
    )
)
def test_predictor(processor_type, image_type, device):

    predictor = GroundingDINOPredictor(GroundingDINO(get_simple_config()), processor_type, device=device)
    if image_type == "torch":
        predictor.set_image(torch.rand(3, 256, 256, device=device))
    elif image_type == "pil":
        predictor.set_image(to_pil_image(torch.rand(3, 256, 256)))
    elif image_type == "numpy":
        predictor.set_image(np.random.rand(256, 256, 3))

    predictor.set_text(["test", "dog", "cell phone"]).predict()

@pytest.mark.parametrize(
    argnames=["processor_type","target_hw","label_map"],
    argvalues=product(
        ["exclusive", "inclusive"],
        [None, "tuple"],
        [None, "dict", "list"],
    )
)
def test_predictor_postprocessing_args(processor_type, target_hw, label_map):
    if target_hw == "tuple":
        target_hw = (141, 124)
    if label_map == "dict":
        label_map = {0: 1, 1: 2, 2: 3}
    elif label_map == "list":
        label_map = [1, 2, 3]

    predictor = GroundingDINOPredictor(GroundingDINO(get_simple_config()), processor_type, device="cpu")
    predictor.set_image(torch.rand(3, 256, 256)).set_text(["test", "dog", "cell phone"]).predict(
        target_hw=target_hw, label_map=label_map
    )

@pytest.mark.parametrize(
    argnames=["draw"],
    argvalues=[(True,), (False,)]
)
def test_predictor_postprocessing_draw(draw):
    target_hw = (141, 124)
    label_map = {0: 1, 1: 2, 2: 3}

    predictor = GroundingDINOPredictor(GroundingDINO(get_simple_config()), device="cpu")
    predictor.set_image(torch.rand(3, 256, 256)).set_text(["test", "dog", "cell phone"]).predict(
        draw=draw, target_hw=target_hw, label_map=label_map
    )


def test_predictor_fwd():
    predictor = GroundingDINOPredictor(GroundingDINO(get_simple_config()), "exclusive", device="cpu")
    predictor.forward(torch.rand(3, 256, 256), ["test", "dog", "cell phone"], target_hw=(141, 124), label_map=[1, 2, 3], box_format="xyxy")