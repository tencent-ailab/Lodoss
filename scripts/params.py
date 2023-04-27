from argparse import ArgumentParser

def add_model_specific_args(parser):
    parser = ArgumentParser(parents=[parser], add_help=False)

    # **************** Parameters for training **************** #
    parser.add_argument("--train_on", action="store_true",
                        help='set to train a model')
    parser.add_argument("--test_on", action="store_true",
                        help='set to train a model')
    parser.add_argument("--batch_size", type=int,
                        default=2, help="Batch size")
    parser.add_argument("--lr", type=float,
                        default=0.00003, help="Learning rate")
    parser.add_argument("--warmup", type=float,
                        default=0.1, help="Percent of warmup steps")
    parser.add_argument("--epochs", type=int,
                        default=20, help="Number of epochs")
    parser.add_argument("--num_workers", type=int,
                        default=4, help="Number of data loader workers")
    parser.add_argument("--limit_train_batches", default=0.001,
                        type=float, help='Percent of training data used')
    parser.add_argument("--limit_val_batches", default=0.05,
                        type=float, help='Percent of validation data used')
    parser.add_argument("--limit_test_batches", default=0.05,
                        type=float, help='Percent of test data used')
    parser.add_argument("--cosine_scheduler", action="store_true",
                        help="Set to use cosine scheduler; default: linear scheduler")
    parser.add_argument("--optimizer", type=str, default='adam', choices=['adam', 'adamw'],
                        help="Choose optimizer; default: Adam")
    parser.add_argument("--train_ds", action="store_true",
                        help="Set to use deepspeed strategy for training; default: DDP")
    parser.add_argument("--monitor_var", type=str, default='rouge_avg',
                        choices=['rouge1', 'rouge2', 'rouge_avg', 'loss'])
    parser.add_argument("--save_top_k", type=int, default=-1, help="number of checkpoints to store")
    parser.add_argument("--every_n_epochs", type=int, default=5, help="Number of epochs between checkpoints")
    parser.add_argument("--gradient_clip_val", type=float, default=0)

    parser.add_argument("--refresh_rate", type=int,
                        default=10, help="Progress bar refresh rate")
    # parser.add_argument("--version", type=str, default='', help="set version_{} directory for resumed training")
    

    # **************** Parameters for testing **************** #
    parser.add_argument("--block_trigram", action="store_true",
                        help="Set to block trigram when val/test")
    parser.add_argument("--num_sent_inf", type=int,
                        default=-1, help="Number of sentence for inference")
    parser.add_argument("--oracle_inf", action="store_true",
                        help="Set to compute rouge scores with oracle labels")
    parser.add_argument("--sort_sum_pred", action="store_true",
                        help="Set to sort summary predictions by temporal order")


    # **************** Parameters for general **************** #
    parser.add_argument("--seed", type=int, default=1234567, help="Seed")
    parser.add_argument("--gpus", type=int, default=-1, nargs='+',
                        help="Set GPU IDs to use. default=-1 to use all GPUs e.g. 1 3 => GPU [1, 3]")
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="number of gradient accumulation steps")
    parser.add_argument("--fp16", action='store_true', help="Use fp16 ")
    parser.add_argument('--grad_ckpt', action='store_true',
                        help='Enable gradient checkpointing to save memory')
    parser.add_argument("--debug", action='store_true', help="debug run")
    parser.add_argument("--amp_level", type=str, default=None,
                        help="O0:FP32 training, O1:Mixed Precision, O2:“Almost FP16” Mixed Precision, O3:FP16 training")
    parser.add_argument("--amp_backend", type=str,
                        default='native', choices=['native', 'apex'])
    parser.add_argument("--num_nodes", type=int, default=1, help="number of nodes")


    # **************** Parameters for model **************** #
    parser.add_argument("--n_transformer_layer", type=int, default=2,
                        help="number of sentence-level transformer layers")
    parser.add_argument("--n_transformer_head", type=int, default=1,
                        help="number of multi-heads in the Transformer layer")
    parser.add_argument("--dropout_transformer", type=float, default=0.1,
                        help="dropout rate for the Transformer layer")
    parser.add_argument("--max_input_len", type=int, default=4096,
                        help="maximum num of wordpieces in the input")
    parser.add_argument("--attention_window", type=int,
                        default=512, help="Local attention window size for Longformer")
    parser.add_argument("--increasing_window_size", action="store_true",
                        help="Apply increasing attention window size from low to high")        
    parser.add_argument("--model_name", type=str, default="allenai/longformer-base-4096", choices=[
        "allenai/longformer-base-4096", "allenai/longformer-large-4096",
    ])
    parser.add_argument("--is_longer_seq", action="store_true",
                        help="Set to choose a Longformer with more than 4096 tokens (16K)")


    # **************** Parameters for data **************** #
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--data_set", type=str, default="openasp", choices=["openasp", "cnndm"])
    parser.add_argument("--hf_dir", type=str, default='huggingface_dataset')
    parser.add_argument("--data_suffix", type=str, default="")
    # parser.add_argument("--string_data", action="store_true", help="Set to use string input, not tokenized data")
    

    # **************** Parameters for model storing and loading **************** #
    parser.add_argument("--local_backbone_dir", type=str, default='',
                        help="If not empty string, load huggingface official models from local directory it specifies.")
    parser.add_argument("--model_save_dir", type=str, default=".",
                        help="Directory for saving the model checkpoints.")
    parser.add_argument("--every_n_train_steps", type=float, default=1.0,
                        help="Store trained models at every n steps; "
                             "Use float to check within a training epoch, use int to check every n steps")
    parser.add_argument("--resume_ckpt", type=str, default=None,
                        help="Path of a checkpoint to resume from")
    parser.add_argument("--from_pretrained", type=str, default=None,
                        help="Path to a checkpoint to load model weights but not training state")
    parser.add_argument("--save_longmodel_to", type=str, default='./longer_models',
                        help="model path to save a model with repeating weights of Longformer")
    parser.add_argument("--cache_eval_dir", type=str, default='/data/home/swcho/workspace/huggingface_evaluation_metric')

    return parser