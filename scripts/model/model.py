import os
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss

from .encoder import ExtTransformerEncoder
from .longformer import Longformer, create_long_model


class ExtSummarizer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.is_seg = args.is_seg
        self.is_dpp = args.is_dpp
        self.loss_fct = BCEWithLogitsLoss(reduction='none')

        model_path = None
        if self.args.is_longer_seq:
            model_name = '{}-{}'.format(self.args.model_name.split('/')[-1], self.args.max_input_len)
            model_path = os.path.join(self.args.save_longmodel_to, model_name)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            if not os.path.exists(os.path.join(model_path, 'pytorch_model.bin')):
                create_long_model(self.args, model_path)

        self.sent_emb_model = Longformer(self.args, model_path)
        emb_size = self.sent_emb_model.model.config.hidden_size
        self.ext_layer = ExtTransformerEncoder(self.args, emb_size)

    def forward(self, input_ids, cls_ids, mask_cls, sent_sum_labels, sent_seg_labels):
        top_vec = self.sent_emb_model(input_ids)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), cls_ids]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        outputs = self.ext_layer(sents_vec, mask_cls, sent_sum_labels, sent_seg_labels)

        # loss
        loss = torch.tensor(0.0, device=sent_sum_labels.device)

        logit_sum = outputs['logit_sum'].squeeze(-1)
        sent_sum_labels = sent_sum_labels.squeeze(-1)
        loss_sum = self.loss_fct(logit_sum, sent_sum_labels.float())
        loss_sum[torch.isnan(loss_sum) == 1] = 0
        loss_sum = (loss_sum * mask_cls.float()).sum() / mask_cls.sum()
        loss = loss + loss_sum

        if self.is_seg:
            logit_seg = outputs['logit_seg'].squeeze(-1)
            sent_seg_labels = sent_seg_labels.squeeze(-1)
            loss_seg = self.loss_fct(logit_seg, sent_seg_labels.float())
            loss_seg[torch.isnan(loss_seg) == 1] = 0
            loss_seg = (loss_seg * mask_cls.float()).sum() / mask_cls.sum()
            loss = loss + loss_seg

        if self.is_dpp:
            loss_dpp = outputs['loss_dpp']
            loss = loss + loss_dpp

        # unused parameters in model.longformer        
        unused_params = ['model.pooler.dense.bias',
                         'model.pooler.dense.weight']
        for n, p in self.sent_emb_model.named_parameters():
            if n in unused_params:
                # p.requires_grad = False
                loss += 0.0 * p.sum()

        return {'logit_sum': logit_sum, 'loss': loss}
