import torch

from .types import OutputFeatures


def get_token_to_caption_map(
    txt_cap_ids: torch.Tensor, normalize: bool = True
) -> torch.Tensor:
    max_cap_id = int(txt_cap_ids.max())
    cap_ids = torch.arange(max_cap_id + 1, device=txt_cap_ids.device)
    cap_map = (
        txt_cap_ids[:, :, None] == cap_ids[None, None, :]
    )  # [B, N_txt, N_cap]
    if normalize:
        cap_map = cap_map / (
            cap_map.sum(dim=-2, keepdim=True) + 1e-8
        )  # [B, N_txt, N_cap]
    return cap_map


def get_output_caption_probs(out: OutputFeatures) -> torch.Tensor:
    out_cls = torch.masked_fill(
        out["cls"], out["mask"][:, :, None], -torch.inf
    )  # [B, N_q, N_txt]
    probs = torch.sigmoid(out_cls)  # [B, N_q, N_txt]
    cap_map = get_token_to_caption_map(out["cap_ids"], normalize=True)
    cap_probs = probs @ cap_map  # [B, N_q, N_cap]
    return cap_probs
