import torch
import torch.nn as nn
from transformers import LongformerTokenizer
from transformers import LongformerConfig
from transformers.models.longformer.modeling_longformer import LongformerModel


class Longformer(nn.Module):
    def __init__(self, args, model_path=None):
        super(Longformer, self).__init__()
        self.args = args

        model_name = self.args.model_name if model_path is None else model_path
        self.tokenizer = LongformerTokenizer.from_pretrained(model_name)
        self.config = LongformerConfig.from_pretrained(model_name)
        if self.args.increasing_window_size:
            if 'base' in self.args.model_name:
                windows = [32, 32, 64, 64, 128, 128,
                           256, 256, 256, 512, 512, 512]
            elif 'large' in self.args.model_name:
                windows = [32, 32, 32, 32, 64, 64, 64, 64, 128, 128, 128, 128,
                           256, 256, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512]
            self.config.attention_window = windows
        else:
            # fixed window size
            self.config.attention_window = [self.args.attention_window] * len(self.config.attention_window)

        self.model = LongformerModel.from_pretrained(model_name, config=self.config)
        if args.grad_ckpt:
            self.model.gradient_checkpointing_enable()

    def _set_global_attention_mask(self, input_ids):
        global_attention_mask = torch.zeros(
            input_ids.shape, dtype=torch.long, device=input_ids.device)
        global_attention_mask[:, 0] = 1
        return global_attention_mask

    def forward(self, input_ids):
        outputs = self.model(input_ids, attention_mask=(input_ids != self.tokenizer.pad_token_id),
                             global_attention_mask=self._set_global_attention_mask(input_ids))
        return outputs.last_hidden_state


def create_long_model(args, model_path):
    longformer = Longformer(args)
    model = longformer.model
    tokenizer = longformer.tokenizer
    config = longformer.config
    config.architectures = ['LongformerEncoder16K', ]

    # extend position embeddings
    tokenizer.model_max_length = args.max_input_len
    tokenizer.init_kwargs['model_max_length'] = args.max_input_len
    current_max_pos, embed_size = model.embeddings.position_embeddings.weight.shape
    # NOTE: RoBERTa has positions 0,1 reserved, so embedding size is max position + 2
    args.max_input_len += 2
    config.max_position_embeddings = args.max_input_len
    assert args.max_input_len > current_max_pos
    # allocate a larger position embedding matrix
    new_pos_embed = model.embeddings.position_embeddings.weight.new_empty(
        args.max_input_len, embed_size)

    # copy position embeddings over and over to initialize the new position embeddings
    k = 2
    step = current_max_pos - 2
    while k < args.max_input_len - 1:
        new_pos_embed[k:(k + step)] = model.embeddings.position_embeddings.weight[2:]
        k += step
    model.embeddings.position_embeddings.weight.data = new_pos_embed

    if hasattr(model.embeddings, 'position_ids'):
        model.embeddings.position_ids.data = torch.tensor(
                [i for i in range(args.max_input_len)]).reshape(1, args.max_input_len)

    print(f'saving long sequence model to {model_path}')
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
