import torch
from pytorch_lightning.utilities.distributed import rank_zero_only


def load_parameters(model, path, strict=False):
    checkpoint = torch.load(path)
    pretrain_model_dict = checkpoint['state_dict']

    model_dict = model.state_dict()
    pretrain_model_dict = {'.'.join(k.split('.')[1:]): v for k, v in pretrain_model_dict.items()}
    pretrain_model_dict = {k: v for k, v in pretrain_model_dict.items()
                           if k in model_dict and v.shape == model_dict[k].shape}
    model.load_state_dict(pretrain_model_dict, strict=strict)
    print_rank_0(f'Model loaded from [{path}] (loaded dict size:{len(pretrain_model_dict)}, '
                 f'model dict size:{len(model_dict)})')
    return model


def count_num_param(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def count_available_gpus(args):
    if torch.cuda.is_available():
        if args.gpus == -1:
            gpu_count = torch.cuda.device_count()
        elif isinstance(args.gpus, list):
            gpu_count = len(args.gpus)
    else:
        gpu_count = 1
    print_rank_0('available gpus:', gpu_count)
    args.gpu_count = gpu_count
    return args


@rank_zero_only
def print_rank_0(message):
    print(message, flush=True)
