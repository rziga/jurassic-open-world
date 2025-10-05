# type: ignore
# noqa
import torch
from torch import nn
from transformers import AutoTokenizer

def create_positive_map(tokenized, tokens_positive,cat_list,caption):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j"""
    positive_map = torch.zeros((len(tokens_positive), 256), dtype=torch.float)

    for j,label in enumerate(tokens_positive):

        start_ind = caption.find(cat_list[label])
        end_ind = start_ind + len(cat_list[label]) - 1
        beg_pos = tokenized.char_to_token(start_ind)
        try:
            end_pos = tokenized.char_to_token(end_ind)
        except:  # noqa: E722
            end_pos = None
        if end_pos is None:
            try:
                end_pos = tokenized.char_to_token(end_ind - 1)
                if end_pos is None:
                    end_pos = tokenized.char_to_token(end_ind - 2)
            except:  # noqa: E722
                end_pos = None
        if beg_pos is None or end_pos is None:
            continue
        if beg_pos < 0 or end_pos < 0:
            continue
        if beg_pos > end_pos:
            continue
        # assert beg_pos is not None and end_pos is not None
        positive_map[j,beg_pos: end_pos + 1].fill_(1)
    return positive_map 


class gt_Postprocessor(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, num_select=100, text_encoder_type="bert-base-uncased"):
        super().__init__()
        self.num_select = num_select
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_type)

    @torch.no_grad()
    def raw_forward(self, outputs, cat_list):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        num_select = self.num_select
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        
        caption = " . ".join(cat_list) + ' .'
        tokenized = self.tokenizer(caption, padding="longest", return_tensors="pt")
        label_list = torch.arange(len(cat_list))
        pos_maps=create_positive_map(tokenized, label_list, cat_list, caption).to(out_logits.device)

        prob_to_token = out_logits.sigmoid()
        for label_ind in range(len(pos_maps)):
            if pos_maps[label_ind].sum() != 0:
                pos_maps[label_ind]=pos_maps[label_ind]/pos_maps[label_ind].sum()

        prob_to_label = prob_to_token @ pos_maps.T

        prob = prob_to_label
        topk_values, topk_indexes = torch.topk(prob.view(prob.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, prob.shape[2], rounding_mode='trunc')
        labels = topk_indexes % prob.shape[2]
        
        #if not_to_xyxy:
        boxes = out_bbox
        #else:
        #    boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        # if test:
        #     assert not not_to_xyxy
        #     boxes[:,:,2:] = boxes[:,:,2:] - boxes[:,:,:2]
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        
        # and from relative [0, 1] to absolute [0, height] coordinates
        #img_h, img_w = target_sizes.unbind(1)
        #scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        #boxes = boxes * scale_fct[:, None, :]

        #if self.nms_iou_threshold > 0:
        #    item_indices = [nms(b, s, iou_threshold=self.nms_iou_threshold) for b,s in zip(boxes, scores)]
        #
        #    results = [{'scores': s[i], 'labels': l[i], 'boxes': b[i]} for s, l, b, i in zip(scores, labels, boxes, item_indices)]
        #else:
        #    results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]  # noqa: E741
        return results
    
    def forward(self, out_cls, out_bbox, captions):
        return self.raw_forward({"pred_logits": out_cls, "pred_boxes": out_bbox}, captions)