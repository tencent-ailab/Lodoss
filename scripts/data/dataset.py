import numpy as np
import torch
from torch.utils.data import Dataset


class SummarizationDataset(Dataset):
    def __init__(self, dataset, tokenizer, args):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.cls_token = self.tokenizer.cls_token
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token = self.tokenizer.sep_token
        self.sep_token_id = self.tokenizer.sep_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.args = args

    def __len__(self):
        """Returns length of the dataset"""
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        # keys: article_id, abstract_list, section_list, section_names, selected_ids
        section_list = data['section_list']
        summary = data['abstract_list']
        oracle_ids = data['selected_ids']

        sentences = [sent for section in section_list for sent in section]

        # input
        entry_text = f'{self.sep_token}{self.cls_token}'.join(sentences)
        entry_text = self.cls_token + entry_text + self.sep_token
        input_ids = self.tokenizer.encode(entry_text,
                                          truncation=True,
                                          max_length=self.args.max_input_len,
                                          add_special_tokens=False)

        cls_ids = [i for i, iid in enumerate(input_ids) if iid == self.cls_token_id]
        num_sent = len(cls_ids)

        # summary label
        summary_label = np.asarray([0 for section in section_list for _ in range(len(section))])
        summary_label[oracle_ids] = 1
        label_sum = list(summary_label)

        # segment label
        label_seg = []
        for sid, section in enumerate(section_list):
            segment_label = [0 for _ in section]
            if segment_label:
                segment_label[self.args.seg_label_pos] = 1
                label_seg += segment_label

        input_text = sentences[:num_sent]
        input_ids = torch.tensor(input_ids)

        return input_ids, label_sum, label_seg, cls_ids, input_text, summary, idx

    @staticmethod
    def collate(batch):
        def _pad(data, pad_id, width=-1):
            if width == -1:
                width = max(len(d) for d in data)
            rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
            return rtn_data

        pad_token_id = 1
        input_ids, label_sum, label_seg, cls_ids, input_text, summary, index = list(zip(*batch))

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids,
                                                    batch_first=True,
                                                    padding_value=pad_token_id)

        sent_sum_labels = torch.tensor(_pad(label_sum, 0))
        sent_seg_labels = torch.tensor(_pad(label_seg, 0))

        cls_ids = torch.tensor(_pad(cls_ids, -1))
        mask_cls = 1 - (cls_ids == -1).float()
        cls_ids[cls_ids == -1] = 0

        # size check
        max_size_sumlabel = sent_sum_labels.size(-1)
        max_size_seglabel = sent_seg_labels.size(-1)
        assert max_size_sumlabel == max_size_seglabel, \
            f'sum label len:{max_size_sumlabel} != seg lebel len:{max_size_seglabel}'
        max_size_cls = cls_ids.size(-1)
        if max_size_sumlabel < max_size_cls:
            cls_ids = cls_ids[:, :max_size_sumlabel]
            mask_cls = mask_cls[:, :max_size_sumlabel]
        elif max_size_sumlabel > max_size_cls:
            sent_sum_labels = sent_sum_labels[:, :max_size_cls]
            sent_seg_labels = sent_seg_labels[:, :max_size_cls]

        return input_ids, cls_ids, mask_cls, sent_sum_labels, sent_seg_labels, input_text, summary, index

    @staticmethod
    def collate_fn(batch):
        data = SummarizationDataset.collate(batch)
        return data[:-3]

    @staticmethod
    def collate_fn_test(batch):
        return SummarizationDataset.collate(batch)
