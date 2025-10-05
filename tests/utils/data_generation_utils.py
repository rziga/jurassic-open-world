# type: ignore
import torch
from jurassic_open_world.utils.image import generate_xy_coordinates
from jurassic_open_world.utils.types import ImageFeatures, TextFeatures, QueryFeatures


def generate_random_data(B, img_dims, N_txt, N_q, C) -> tuple[ImageFeatures, TextFeatures, QueryFeatures]:
    N_img = sum(d1*d2 for d1, d2 in img_dims)

    # img features
    x_img = torch.rand(B, N_img, C)
    pos_img = torch.rand(B, N_img, C)
    mask_img = torch.zeros(B, N_img).bool()
    coor_img = torch.cat([
        generate_xy_coordinates(
            torch.zeros(B, d1, d2, dtype=torch.bool)
        ).flatten(-2).mT # [B, 2, H, W] -> [B, 2, H*W] -> [B, H*W, 2]
        for d1, d2 in img_dims],
    dim=1)
    dims_img = torch.tensor(img_dims)
    valid_ratios_img = torch.ones(B, len(img_dims), 2)

    # text features
    x_txt = torch.rand(B, N_txt, C)
    pos_txt = torch.rand(B, N_txt, C)
    mask_txt = torch.zeros(B, N_txt).bool()
    self_att_mask_txt = torch.zeros(B, N_txt, N_txt).bool()
    cap_ids_txt = torch.zeros(B, N_txt).long()

    # query features
    x_query = torch.rand(B, N_q, C)
    bbox_query = torch.rand(B, N_q, 4)
    mask_query = torch.zeros(B, N_q).bool()
    att_mask_query = torch.zeros(B, N_q, N_q).bool()

    return (
        {"feat": x_img, "pos": pos_img, "mask": mask_img, "coor": coor_img, "shapes": dims_img, "valid_ratios": valid_ratios_img},
        {"feat": x_txt, "pos": pos_txt, "mask": mask_txt, "att_mask": self_att_mask_txt, "cap_ids": cap_ids_txt},
        {"feat": x_query, "bbox": bbox_query, "mask": mask_query, "att_mask": att_mask_query},
    )

def generate_random_even_data(B, single_img_dims, N_txt, N_q, C):
    return generate_random_data(B, [(d, d) for d in single_img_dims], N_txt, N_q, C)

def generate_random_uneven_data(B, single_img_dims, N_txt, N_q, C):
    return generate_random_data(B, [(d, d//2) for d in single_img_dims], N_txt, N_q, C)
