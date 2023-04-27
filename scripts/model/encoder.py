import math
import torch
import torch.nn as nn


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


class ClassificationHeadLN(nn.Module):
    def __init__(self, input_dim, inner_dim, num_classes, pooler_dropout):
        super().__init__()
        self.predict_net = nn.Sequential(
                nn.Linear(input_dim, inner_dim),
                nn.LayerNorm(inner_dim),
                nn.GELU(),
                nn.Dropout(p=pooler_dropout),
                nn.Linear(inner_dim, num_classes),
        )

    def forward(self, hidden_states):
        hidden_states = self.predict_net(hidden_states)
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
        d_ff = d_model * 4
        dropout = args.dropout_transformer

        # position emb
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.norm = nn.LayerNorm(d_model)

        # transformer
        pytorch_version = int(torch.__version__.split('.')[1])
        self.is_pt_ver_lt9 = True if pytorch_version < 9 else False
        if self.is_pt_ver_lt9:
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=args.n_transformer_head,
                                                       dim_feedforward=d_ff,
                                                       dropout=dropout, activation='gelu')
        else:
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=args.n_transformer_head,
                                                       dim_feedforward=d_ff,
                                                       dropout=dropout, activation='gelu', batch_first=True,
                                                       norm_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=args.n_transformer_layer)

        # prediction layer
        self.pred_sum = ClassificationHeadLN(d_model, d_model, 1, dropout)

    def forward(self, sents_vec, mask):
        batch_size, n_sents = sents_vec.size(0), sents_vec.size(1)
        x = sents_vec

        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = self.norm(x + pos_emb)

        if self.is_pt_ver_lt9:
            x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        if self.is_pt_ver_lt9:
            x = x.transpose(0, 1)

        logit_sum = self.pred_sum(x)
        logit_sum = logit_sum.squeeze(-1) + (mask.float() - 1) * 10000
        outputs = {'logit_sum': logit_sum}

        return outputs
