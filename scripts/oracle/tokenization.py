import os
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk, Dataset, DatasetDict
from transformers import AutoTokenizer



tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096", use_fast=True)
sep_token = tokenizer.sep_token
sep_token_id = tokenizer.sep_token_id
cls_token = tokenizer.cls_token
cls_token_id = tokenizer.cls_token_id
max_token_len = 4096


def parse():
    parser = argparse.ArgumentParser(description='tokenize data')
    parser.add_argument("--data_dir", type=str, default="~/workspace/data")
    parser.add_argument("--data_set", default='openasp', choices=["openasp"])
    parser.add_argument("--hf_dir", type=str, default='huggingface_dataset')
    parser.add_argument("--model_name", type=str, default="allenai/longformer-base-4096", choices=[
                                                        "allenai/led-base-16384", "allenai/led-large-16384",
                                                    ])
    parser.add_argument("--num_jobs", type=int, default=1)
    parser.add_argument("--target_tok", type=int, default=4096)
    return parser.parse_args()

def process_tokenize(data):
    # ['title', 'document', 'aspect', 'query', 'aspect_sents', 'summary', 'oracle_id']
    sentences = data['document']
    entry_text = f'{cls_token}'.join(sentences)
    entry_text = cls_token + entry_text + sep_token
    input_ids = tokenizer.encode(entry_text, truncation=True, max_length=max_token_len, add_special_tokens=False)

    cls_ids = [i for i, iid in enumerate(input_ids) if iid == cls_token_id]
    num_sent = len(cls_ids)

    # summary label
    oid = data['oracle_id']
    summary_label = np.asarray([0 for _ in range(len(sentences))])        
    summary_label[oid] = 1
    label_sum = list(summary_label)
    label_sum = label_sum[:num_sent]

    data['input_ids'] = input_ids
    data['cls_ids'] = cls_ids
    data['input_text'] = sentences[:num_sent]
    data['label_sum'] = label_sum
    return data

args = parse()

dataset = load_from_disk(os.path.join(args.data_dir, args.data_set, args.hf_dir))
print ('data loaded')
dataset = dataset.map(process_tokenize,
                    remove_columns=['title', 'document', 'aspect', 'query', 'aspect_sents', 'oracle_id'],
                    num_proc=args.num_jobs)
print ('data processed')
dataset.save_to_disk(os.path.join(args.data_dir, args.data_set, args.hf_dir+f'_tokenized_{args.target_tok}'))