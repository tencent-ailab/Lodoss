import os
import re
import shutil
import errno
import json
import pickle
import time
import numpy as np
import torch

from rouge_score import rouge_scorer
from pyrouge import Rouge155

REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",
         "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'}


def covert_to_cpu_weights(ckpt_path):
    ckpt = torch.load(ckpt_path)
    weights = ckpt["state_dict"]
    del ckpt
    weights_cpu = {}
    for key, value in weights.items():
        if key.startswith("model."):
            weights_cpu[key[6:]] = value.to("cpu")
    torch.save(weights_cpu, ckpt_path + ".weights")


def load_parameters(model, path, strict=False):
    checkpoint = torch.load(path)  # , map_location=map_location)
    pretrain_model_dict = checkpoint['state_dict']
    del checkpoint

    model_dict = model.state_dict()
    pretrain_model_dict = {'.'.join(k.split('.')[1:]): v for k, v in pretrain_model_dict.items()}
    pretrain_model_dict = {k: v for k, v in pretrain_model_dict.items()
                           if k in model_dict and v.shape == model_dict[k].shape}

    # model_dict.update(pretrain_model_dict)        
    # del pretrain_model_dict
    model.load_state_dict(pretrain_model_dict, strict=strict)
    print(f'Model loaded from [{path}] (loaded dict size:{len(pretrain_model_dict)}, '
          f'model dict size:{len(model_dict)})')
    del pretrain_model_dict
    del model_dict
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
    print('available gpus:', gpu_count)
    args.gpu_count = gpu_count
    return args


def clean(x):
    return re.sub(
            r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''",
            lambda m: REMAP.get(m.group()), x)


def load_folder(folder, suffix):
    files = []
    for f in sorted(os.listdir(folder)):
        if f.endswith(suffix):
            files.append(f)
    print(f'num. of files found in {folder}: {len(files)}')
    return files


def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def save_pkl(data_pkl, file):
    with open(file, 'wb') as fb:
        pickle.dump(data_pkl, fb)  # , protocol=pickle.HIGHEST_PROTOCOL)


def load_pkl(file):
    if not os.path.exists(file):
        print('{} does not exist'.format(file))
        return None

    with open(file, 'rb') as fb:
        data = pickle.load(fb)
    return data


def save_json(data_json, file):
    with open(file, 'w') as fj:
        json.dump(data_json, fj)


def load_json(file):
    if not os.path.exists(file):
        print('{} does not exist'.format(file))
        return None

    with open(file, 'r') as fj:
        data = json.load(fj)
    return data


def print_stats_all(data, desc=''):
    print(f'{desc} - len:{len(data)}, min:{np.min(data)}, max:{np.max(data)}, mean:{np.mean(data):.3f},\
            median:{np.median(data):.3f}, std:{np.std(data):.3f}, sum:{np.sum(data)},\
            80percentile:{np.percentile(data, 80)}, 90percentile:{np.percentile(data, 90)},\
            95percentile:{np.percentile(data, 95)}, 99percentile:{np.percentile(data, 99)}')


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def print_stats_core(data, desc=''):
    d_min = np.min(data)
    d_max = np.max(data)
    d_mean = np.mean(data)
    d_median = np.median(data)
    d_std = np.std(data)
    print_rank_0(f'{desc} - len:{len(data)}, min:{d_min}, max:{d_max}, mean:{d_mean:.3f}, '
                 f'median:{d_median:.3f}, std:{d_std:.3f}')
    return {'min': d_min, 'max': d_max, 'mean': d_mean}


def compute_pyrouge(refs, preds, rouge=None):
    # Rouge scores
    if rouge is None:
        rouge = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
    rouge1 = rouge2 = rougel = rougelsum = 0.0
    rouge1_P = rouge2_P = rougel_P = rougelsum_P = 0.0
    rouge1_R = rouge2_R = rougel_R = rougelsum_R = 0.0
    for ref, pred in zip(refs, preds):
        score = rouge.score(ref, pred)
        precision, recall, fmeasure = score['rouge1']
        rouge1_P += precision
        rouge1_R += recall
        rouge1 += fmeasure
        precision, recall, fmeasure = score['rouge2']
        rouge2_P += precision
        rouge2_R += recall
        rouge2 += fmeasure
        precision, recall, fmeasure = score['rougeL']
        rougel_P += precision
        rougel_R += recall
        rougel += fmeasure
        precision, recall, fmeasure = score['rougeLsum']
        rougelsum_P += precision
        rougelsum_R += recall
        rougelsum += fmeasure
    results = np.asarray([rouge1_P, rouge1_R, rouge1, rouge2_P, rouge2_R, rouge2,
                          rougel_P, rougel_R, rougel, rougelsum_P, rougelsum_R, rougelsum]) / len(preds)
    return results


def compute_rouge_1_5_5(references, candidates, temp_dir='', rouge_args=None):
    # references, candidates: list of reference, generated summaries
    if temp_dir == '':
        temp_dir = './temp_summary'
        create_dir(temp_dir)

    # rouge_args default: '-c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -a'
    if rouge_args is None:
        options = ['-e', '../ROUGE-1.5.5/data',
                   '-c', 95,
                   '-2', 4,
                   # '-1',
                   '-U',
                   '-r', 1000,
                   '-n', 4,
                   '-w', 1.2,
                   '-a',
                   ]
        options = list(map(str, options))
        rouge_args = ' '.join(options)

    # references, candidates = list(zip(*self.summ_data))       

    print('candidates:', len(candidates))
    print('references:', len(references))
    assert len(candidates) == len(references), '{} != {}'.format(len(candidates), len(references))

    create_dir(temp_dir)
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = os.path.join(temp_dir, "rouge-tmp-{}".format(current_time))
    create_dir(temp_dir)
    create_dir(tmp_dir + "/candidate")
    create_dir(tmp_dir + "/reference")

    try:
        for i in range(len(candidates)):
            if len(references[i]) < 1:
                continue
            with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(candidates[i])
            with open(tmp_dir + "/reference/ref.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(references[i])
        r = Rouge155()
        r.model_dir = tmp_dir + "/reference/"
        r.system_dir = tmp_dir + "/candidate/"
        r.model_filename_pattern = 'ref.#ID#.txt'
        r.system_filename_pattern = r'cand.(\d+).txt'
        rouge_results = r.convert_and_evaluate(rouge_args=rouge_args)
        results_dict = r.output_to_dict(rouge_results)
        rouge_outputs = print_rouge_scores(results_dict)
    finally:
        pass
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
    return rouge_outputs


def rouge_results_to_str(results_dict):
    return ">> ROUGE-F(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-R(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\n".format(
            results_dict["rouge_1_f_score"] * 100,
            results_dict["rouge_2_f_score"] * 100,
            # results_dict["rouge_3_f_score"] * 100,
            results_dict["rouge_l_f_score"] * 100,
            results_dict["rouge_1_recall"] * 100,
            results_dict["rouge_2_recall"] * 100,
            # results_dict["rouge_3_f_score"] * 100,
            results_dict["rouge_l_recall"] * 100

            # ,results_dict["rouge_su*_f_score"] * 100
    )


def print_rouge_scores(rouges):
    r1p, r1r, r1f = rouges['rouge_1_precision'] * \
                    100, rouges['rouge_1_recall'] * 100, rouges['rouge_1_f_score'] * 100
    r2p, r2r, r2f = rouges['rouge_2_precision'] * \
                    100, rouges['rouge_2_recall'] * 100, rouges['rouge_2_f_score'] * 100
    rlp, rlr, rlf = rouges['rouge_l_precision'] * \
                    100, rouges['rouge_l_recall'] * 100, rouges['rouge_l_f_score'] * 100
    if 'rouge_su4_precision' in rouges:
        rs4p, rs4r, rs4f = rouges['rouge_su4_precision'] * \
                           100, rouges['rouge_su4_recall'] * \
                           100, rouges['rouge_su4_f_score'] * 100
        outputs = [r1p, r1r, r1f, r2p, r2r, r2f,
                   rlp, rlr, rlf, rs4p, rs4r, rs4f]
        print(
            'Rouge-1: {:.2f} {:.2f} {:.2f}\nRouge-2: {:.2f} {:.2f} {:.2f}\nRouge-l: {:.2f} {:.2f} {:.2f}\nRouge-SU4: {:.2f} {:.2f} {:.2f}\n'.format(
                *outputs))
    else:
        outputs = [r1p, r1r, r1f, r2p, r2r, r2f, rlp, rlr, rlf]
        print(
                'Rouge-1: {:.2f} {:.2f} {:.2f}\nRouge-2: {:.2f} {:.2f} {:.2f}\nRouge-l: {:.2f} {:.2f} {:.2f}\n'.format(
                    *outputs))

    outputs = [float('{:.2f}'.format(out)) for out in outputs]
    return outputs


def print_prf(p_r_f1, prefix='', scale100=True):
    precision, recall, f1_score = [], [], []
    for i, (p, r, f1) in enumerate(p_r_f1):
        precision.append(p)
        recall.append(r)
        f1_score.append(f1)
    outputs = [np.mean(precision), np.std(precision), np.mean(
            recall), np.std(recall), np.mean(f1_score), np.std(f1_score)]
    if scale100:
        outputs = [elem * 100 for elem in outputs]
        print('[{}] Avg. Precision:{:.2f} (std:{:.2f}), Avg. Recall:{:.2f} (std:{:.2f}), Avg. F1:{:.2f} (std:{:.2f})'.format(
                prefix, *outputs))
        outputs = [float('{:.2f}'.format(out)) for out in outputs]
    else:
        print('[{}] Avg. Precision:{:.3f} (std:{:.3f}), Avg. Recall:{:.3f} (std:{:.3f}), Avg. F1:{:.3f} (std:{:.3f})'.format(
                prefix, *outputs))
        outputs = [float('{:.3f}'.format(out)) for out in outputs]
    return outputs