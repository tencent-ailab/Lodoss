import os
import json
import argparse
import datetime

import torch
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as default

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from transformers import set_seed
from transformers import AutoTokenizer

from model import (
    LongDocSummarizer,
    ExtSummarizer,
)
from data import DataSetModule
from params import add_model_specific_args
from utils import (
    load_parameters,
    count_num_param,
    count_available_gpus,
    print_rank_0,
)

if __name__ == "__main__":
    # Setup command line args
    main_arg_parser = argparse.ArgumentParser(description="extractive summarization")
    parser = add_model_specific_args(main_arg_parser)
    args = parser.parse_args()
    pl.seed_everything(args.seed)  # pl
    set_seed(args.seed)  # random, np, torch, tensorflow

    # debug mesages
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # number of gpus
    args = count_available_gpus(args)

    model_name = os.path.basename(args.model_name)
    if args.local_backbone_dir:
        args.model_name = os.path.join(args.local_backbone_dir, args.model_name)
    print_rank_0(json.dumps(vars(args), indent=4, sort_keys=True))

    # model
    model = ExtSummarizer(args)

    if args.grad_ckpt:
        model.gradient_checkpointing_enable()

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # load weights
    if args.from_pretrained is not None:
        model = load_parameters(model, args.from_pretrained, strict=False)
    total_num_param = count_num_param(model)
    print_rank_0('num. of param in model with grad True:', total_num_param)

    # load data
    data = DataSetModule(args=args, tokenizer=tokenizer)
    data.setup()
    real_batch_size = args.batch_size * args.gpu_count * args.num_nodes * args.grad_accum
    train_size = len(data.dataset['train'])
    train_steps_per_epoch = train_size // real_batch_size
    args.train_steps = train_steps_per_epoch * args.epochs
    print_rank_0(f'train step: {args.train_steps}')

    # model class
    model_longdoc = LongDocSummarizer(args, model, tokenizer)

    # model path
    str_model_in = f'_Len{args.max_input_len}'
    str_model = f'_Layer{args.n_transformer_layer}_Head{args.n_transformer_head}        '
    model_name = f'{args.data_set}_{model_name}{str_model_in}{str_model}'
    default_root_dir = os.path.join(args.model_save_dir, 'model_train', f'{model_name}')

    checkpoint_callback = ModelCheckpoint(filename='{epoch}-{step}-{train_loss:.8f}',
                                          monitor='train_loss',
                                          mode='min',
                                          save_top_k=args.save_top_k, save_last=True,
                                          every_n_epochs=args.every_n_epochs,
                                          save_weights_only=True)

    progress_callback = TQDMProgressBar(refresh_rate=args.refresh_rate)

    strategy = DDPStrategy(
            timeout=datetime.timedelta(3600),
            ddp_comm_hook=default.fp16_compress_hook
    )

    trainer = pl.Trainer(
            default_root_dir=default_root_dir,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=args.gpus,
            num_nodes=args.num_nodes,
            strategy=strategy,  # DDPStrategy(find_unused_parameters=False, process_group_backend="nccl"),
            accumulate_grad_batches=args.grad_accum,
            precision=16 if args.fp16 else 32,
            gradient_clip_val=args.gradient_clip_val if args.gradient_clip_val > 1e-4 else None,  # avoid fp16 nan
            max_epochs=args.epochs,
            callbacks=[checkpoint_callback, progress_callback],
            logger=TensorBoardLogger(default_root_dir, name=''),
            sync_batchnorm=True,
            replace_sampler_ddp=True,
            val_check_interval=1.0 / args.every_n_train_steps,
            limit_train_batches=args.limit_train_batches,
            limit_val_batches=args.limit_val_batches,
            limit_test_batches=args.limit_test_batches,
            profiler=None,  # advanced, simple, None
    )

    # training
    if args.train_on:
        print_rank_0("start training")
        trainer.fit(model=model_longdoc, datamodule=data, ckpt_path=args.resume_ckpt)

    if args.test_on:
        result = trainer.test(model=model_longdoc, dataloaders=data.test_dataloader())
        ckpt_path = os.path.dirname(args.from_pretrained)
        with open(os.path.join(ckpt_path, f"test_rouge_result.txt"), "w") as f:
            print(result, file=f)

        result = trainer.test(model=model_longdoc, datamodule=data.test_dataloader())
        print_rank_0(result)
