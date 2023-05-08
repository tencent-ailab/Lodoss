import math
import torch
import torch.nn as nn
from .dpp import DPP


class ClassificationHead(nn.Module):
    def __init__(self, input_dim, inner_dim, num_classes, pooler_dropout):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class PositionalEncoding(nn.Module):
    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super().__init__()
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if step:
            emb = emb + self.pe[:, step][:, None, :]
        else:
            emb = emb + self.pe[:, : emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, : emb.size(1)]


class ExtTransformerEncoder(nn.Module):
    def __init__(self, args, d_model):
        super().__init__()
        self.args = args
        self.is_seg = args.is_seg
        self.is_dpp = args.is_dpp
        d_ff = 2048     # d_model * 4
        dropout = args.dropout_transformer

        # position emb
        self.pos_emb = PositionalEncoding(dropout, d_model)
        # self.norm = nn.LayerNorm(d_model)

        # transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=args.n_transformer_head,
                                                   dim_feedforward=d_ff,
                                                   dropout=dropout, activation='gelu', batch_first=True,
                                                   norm_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=args.n_transformer_layer,
                                                         norm=nn.LayerNorm(d_model))

        # prediction layer
        self.pred_sum = ClassificationHead(d_model, d_model, 1, dropout)

        # segmentation layer
        if self.is_seg:
            self.pred_seg = ClassificationHead(d_model, d_model, 1, dropout)

        # DPP layer
        if self.is_dpp:
            self.dpp_layer = DPP(dpp_weight=args.dpp_weight, is_seg_label_1st_sent=args.seg_label_pos == 0)

    def forward(self, sents_vec, mask, sent_sum_labels, sent_seg_labels):
        batch_size, n_sents = sents_vec.size(0), sents_vec.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = sents_vec
        x = x + pos_emb

        x = self.transformer_encoder(x)

        logit_sum = self.pred_sum(x)
        logit_sum = logit_sum.squeeze(-1) + (mask.float() - 1) * 10000
        outputs = {'logit_sum': logit_sum}

        if self.is_seg:
            logit_seg = self.pred_seg(x)
            logit_seg = logit_seg.squeeze(-1) * mask.float()
            outputs['logit_seg'] = logit_seg

        if self.is_dpp:
            # x = self.sim_dpp_sum(x)
            dpp_outputs = self.dpp_layer(x, logit_sum, mask, sent_sum_labels, sent_seg_labels)
            outputs['loss_dpp'] = dpp_outputs['loss_dpp']

        return outputs
