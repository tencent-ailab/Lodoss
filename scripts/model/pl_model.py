import pytorch_lightning as pl
from torch.optim import AdamW, Adam
from apex.optimizers import FusedAdam, FusedLAMB
from transformers.optimization import get_linear_schedule_with_warmup
from typing import Any

class Summarizer(pl.LightningModule):
    def __init__(self, args, model):
        super().__init__()
        self.args = args
        self.model = model
    
    def forward(self, *args, **kwargs):
        pass
    
    def training_step(self, *args, **kwargs):
        pass

    def validation_step(self, *args, **kwargs):
        pass

    def validation_epoch_end(self, outputs):
        pass
    
    def test_step(self, *args, **kwargs):
        pass

    def test_epoch_end(self, outputs):
        pass
    
    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]

        eps = 1e-8
        if self.args.optimizer == 'adam':
            optimizer_fn = Adam
            weight_decay = 0
        elif self.args.optimizer == 'adamw':
            optimizer_fn = AdamW
            weight_decay = 0.01
        elif self.args.optimizer == 'fusedadam':
            optimizer_fn = FusedAdam
            weight_decay = 0.01
        elif self.args.optimizer == 'fusedlamb':
            optimizer_fn = FusedLAMB
            weight_decay = 0.01

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            },
        ]

        optimizer = optimizer_fn(
            optimizer_grouped_parameters,
            lr=self.args.lr,
            eps=eps
        )

        # if self.args.adamW:
        #     optimizer = AdamW(self.model.parameters(), lr=self.args.lr)
        #     # optimizer = AdamW([
        #     #         {'params': self.model.model.encoder.parameters()},
        #     #         {'params': self.model.model.decoder.parameters(), 'lr': 1e-3},
        #     #         {'params': self.model.cls.parameters(), 'lr': 1e-3}], lr=self.args.lr)
        # else:
        #     optimizer = Adam(self.model.parameters(), lr=self.args.lr)
        
        total_train_steps = self.args.train_steps
        num_warmup_steps = int(total_train_steps * self.args.warmup)
        print('total_train_steps:', total_train_steps, 'num_warmup_steps:', num_warmup_steps)
        scheduler = get_linear_schedule_with_warmup(
                        optimizer, num_warmup_steps=num_warmup_steps,
                        num_training_steps=total_train_steps)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}