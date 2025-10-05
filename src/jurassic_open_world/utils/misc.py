from typing import Optional, Sequence

import torch


def stack_pad(
    tensor_list: Sequence[torch.Tensor],
    target_size: Optional[list[int]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stacks tensors from list into a single tensor
    if target_size is None, the contaier tensor shape is inferred by largest dims (e.g. tensor_list=[[3, 512, 384], [3, 420, 640]] -> [2, 3, 512, 640])
    also returns mask of the same shape as the output where True -> padded

    Args:
        tensor_list (list[torch.Tensor]): list of tensors to stack
        target_size (Optional[list[int]], optional): target shape of tensors. Defaults to None.
            Can be a list of 1 element to only constrain the last dim, 2 elements to constrain the last 2 dim etc.

    Returns:
        tuple[torch.Tensor,torch.Tensor]: stacked tensors and mask of the same shape where True -> padded.
    """

    # check if all tensors have the same number of dims
    ndims = [t.ndim for t in tensor_list]
    assert all(n == ndims[0] for n in ndims), (
        f"Tensors must have the same number of dimensions, but got {ndims} instead."
    )

    # check if all tensors are smaller than target_size if it is passed
    shapes = torch.tensor([t.shape for t in tensor_list])
    if target_size is not None:
        assert (
            shapes[:, -len(target_size) :] <= torch.tensor(target_size)
        ).all(), (
            f"Tensors need to be smaller than 'target_size': {target_size}, but got shapes: {shapes}"
        )

    # calculate the container shape
    max_shape = shapes.max(dim=0).values
    container_shape = [
        len(tensor_list),
    ] + max_shape.tolist()

    # expand the container size to target size if needed
    if target_size is not None:
        container_shape[-len(target_size) :] = target_size

    # prepare buffers
    stacked_tensors = torch.zeros(container_shape, dtype=tensor_list[0].dtype)
    mask = torch.ones_like(stacked_tensors, dtype=torch.bool)

    # fill the buffers
    for i, tensor in enumerate(tensor_list):
        idx = (i,) + tuple(slice(0, dim) for dim in tensor.shape)
        stacked_tensors[idx] = tensor
        mask[idx] = False

    return stacked_tensors, mask
