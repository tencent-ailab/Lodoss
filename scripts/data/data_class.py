import os
import torch
from datasets import load_from_disk
from torch.utils.data.dataloader import DataLoader
from pytorch_lightning import LightningDataModule
from .dataset import SummarizationDataset


class DataSetModule(LightningDataModule):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.dataset = None
        self.args = args
        self.tokenizer = tokenizer

    def setup(self, stage=None):
        self.dataset = load_from_disk(os.path.join(self.args.data_dir, self.args.data_set))

    def _get_dataloader(self, split_name):
        """Get training and validation dataloaders"""
        is_train = True if 'train' in split_name else False
        dataset_split = self.dataset[split_name]
        dataset = SummarizationDataset(dataset=dataset_split, tokenizer=self.tokenizer, args=self.args)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=is_train)
        collate_fn = SummarizationDataset.collate_fn if is_train else SummarizationDataset.collate_fn_test
        return DataLoader(dataset,
                          batch_size=self.args.batch_size,
                          shuffle=False,
                          num_workers=self.args.num_workers,
                          sampler=sampler,
                          collate_fn=collate_fn)

    def train_dataloader(self):
        return self._get_dataloader('train')

    def val_dataloader(self):
        return self._get_dataloader('validation')

    def test_dataloader(self):
        return self._get_dataloader('test')
