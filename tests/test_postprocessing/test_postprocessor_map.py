import torch
from transformers import AutoTokenizer

from jurassic_open_world.inference.postprocessor import PostProcessorInclusive
from .original_impl import gt_Postprocessor

def test_gt():
    num_q = 900
    num_select = 100
    model_cfg = "bert-base-uncased"
    captions = [["cat", "dog"]]
    tokenizer = AutoTokenizer.from_pretrained(model_cfg)

    tokenized = tokenizer([tokenizer.sep_token.join(cs) for cs in captions], padding="longest", return_tensors="pt")
    input_ids = tokenized.input_ids
    sep_idxs = input_ids == tokenizer.sep_token_id
    cap_ids = torch.full_like(input_ids, -1)
    for i in range(len(cap_ids)):
        idxs = torch.nonzero(sep_idxs[i]).squeeze(1).tolist()
        for j, (start, stop) in enumerate(zip([0] + idxs, idxs)):
            cap_ids[i, start+1:stop] = j


    out_cls = torch.rand(1, num_q, cap_ids.shape[-1])
    out_cls_padded = torch.full((1, num_q, 256), -torch.inf)
    out_cls_padded[:, :, :cap_ids.shape[-1]] = out_cls
    out_bbox = torch.rand(1, num_q, 4)
    out_mask = torch.zeros(1, num_q, dtype=torch.bool)

    good = gt_Postprocessor(num_select, model_cfg)
    test = PostProcessorInclusive(num_select)

    a = good.forward(out_cls_padded, out_bbox, captions[0])
    b = test.forward({"cls": out_cls, "bbox": out_bbox, "mask": out_mask, "cap_ids": cap_ids})

    assert all(torch.allclose(a_["scores"], b_["scores"]) for a_, b_ in zip(a, b))
    assert all(torch.allclose(a_["labels"], b_["labels"]) for a_, b_ in zip(a, b))
    assert all(torch.allclose(a_["boxes"], b_["boxes"]) for a_, b_ in zip(a, b))

def test_gt_bs():
    num_q = 900
    num_select = 100
    model_cfg = "bert-base-uncased"
    captions = [["cat", "dog"], ["red umbrella", "banana", "empty cup of water", "slime"]]
    B = len(captions)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg)

    tokenized = tokenizer([tokenizer.sep_token.join(cs) for cs in captions], padding="longest", return_tensors="pt")
    input_ids = tokenized.input_ids
    sep_idxs = input_ids == tokenizer.sep_token_id
    cap_ids = torch.full_like(input_ids, -1)
    for i in range(len(cap_ids)):
        idxs = torch.nonzero(sep_idxs[i]).squeeze(1).tolist()
        for j, (start, stop) in enumerate(zip([0] + idxs, idxs)):
            cap_ids[i, start+1:stop] = j


    out_cls = torch.rand(B, num_q, cap_ids.shape[-1])
    out_cls_padded = torch.full((B, num_q, 256), -torch.inf)
    out_cls_padded[:, :, :cap_ids.shape[-1]] = out_cls
    out_bbox = torch.rand(B, num_q, 4)
    out_mask = torch.zeros(B, num_q, dtype=torch.bool)

    good = gt_Postprocessor(num_select, model_cfg)
    test = PostProcessorInclusive(num_select)

    a = [
        good.forward(out_cls_padded[[i]], out_bbox[[i]], captions[i])[0]
        for i in range(B)
    ]
    b = test.forward({"cls": out_cls, "bbox": out_bbox, "mask": out_mask, "cap_ids": cap_ids})

    assert all(torch.allclose(a_["scores"], b_["scores"]) for a_, b_ in zip(a, b))
    assert all(torch.allclose(a_["labels"], b_["labels"]) for a_, b_ in zip(a, b))
    assert all(torch.allclose(a_["boxes"], b_["boxes"]) for a_, b_ in zip(a, b))