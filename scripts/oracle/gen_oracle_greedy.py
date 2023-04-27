import os
import re
import json
import spacy
import errno
import argparse
import numpy as np
from copy import deepcopy as cp
from rouge_score import rouge_scorer
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict


# sentence segmentation using spaCy
nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser'])     # sm, md, lg, trf
nlp.add_pipe('sentencizer')
nlp.max_length = 300000

# rouge
rouge_type = ['rouge1', 'rouge2']
rouge_metric = rouge_scorer.RougeScorer(rouge_type, use_stemmer=True)

def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

def compute_rouge_scores(refs, preds):
    # Rouge scores
    rouge_eval = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
    rouge1 = rouge2 = rougel = rougelsum = 0.0
    rouge1_P = rouge2_P = rougel_P = rougelsum_P = 0.0
    rouge1_R = rouge2_R = rougel_R = rougelsum_R = 0.0
    for ref, pred in zip(refs, preds):
        score = rouge_eval.score(ref, pred)
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

def compute_rouge_avg(ref, prd, rouge_metric, rouge_type):
    scores = rouge_metric.score(ref, prd)
    score_avg = []
    for r_type in rouge_type:
        score_avg.append(scores[r_type][-1])
    return np.mean(score_avg)

def oracle_greedy(input_txt, summary):
    summary_budget = 3 #len(summary)   # adpative number of oracles

    # greedy method
    summary_str = '\n'.join(summary)
    sent_id_set = set([i for i in range(len(input_txt))])

    selected_ids = []
    rouge_pre = 0
    while len(sent_id_set) > 1 and len(selected_ids) < summary_budget:
        rouge_score_sid = []
        for sid in sent_id_set:
            sents = '\n'.join([input_txt[idx] for idx in selected_ids + [sid]])
            score = compute_rouge_avg(summary_str, sents, rouge_metric, rouge_type)
            rouge_score_sid.append((sid, score))
        id_max = np.argmax([x[1] for x in rouge_score_sid])
        sid_max, rouge_cur = rouge_score_sid[id_max]
        sent_id_set.remove(sid_max)

        if rouge_pre < rouge_cur:
            selected_ids.append(sid_max)
            rouge_pre = rouge_cur
        else:
            break
    return selected_ids

def oracle_greedy_kaiqiang(input_txt, summary, eps=1e-6):
    # Get summary budge
    L = len(input_txt)
    N = len(summary)
    M = min(L, max(round(N * 3), N + 3))

    # Get reference summary
    ref_summary = '\n'.join(summary)

    best_rouge = 0.0
    selected_set = set()
    remaining_set = set(list(range(L)))
    for i in range(M):
        if len(remaining_set) == 0:
            break
        
        # Test all different combinations
        indices = []
        rouge_scores = []

        for j in list(remaining_set):
            temp_set = cp(selected_set)
            temp_set.add(j)
            cur_summary = "\n".join([input_txt[idx] for idx in list(sorted(list(temp_set)))])
            score = compute_rouge_avg(ref_summary, cur_summary, rouge_metric, rouge_type)
            indices.append(j)
            rouge_scores.append(score)

        # Get best option    
        k = np.argmax(rouge_scores)
        cur_best_rouge = rouge_scores[k]
        cur_best_index = indices[k]

        # Update the set if rouge gains
        if cur_best_rouge > best_rouge + eps:
            selected_set.add(cur_best_index)
            remaining_set.remove(cur_best_index)
            best_rouge = cur_best_rouge
        else:
            break
    
    return sorted(list(selected_set))


def segment_text(text):
    doc = nlp(text)            
    sentences = []
    for sent in doc.sents:
        if str(sent).strip():
            sentences.append(str(sent))
    return sentences

def update_data(data, args):
    def remove_new_line(str):
        return re.sub(r'\n', '', str)
    data[args.doc_key] = segment_text(remove_new_line(data[args.doc_key]))
    data[args.sum_key] = [sent for sent in data[args.sum_key].split('\n') if sent]
    data['oracle_id'] = oracle_greedy(data[args.doc_key], data[args.sum_key], args)
    return data

def update_data_openasp(data):
    data['oracle_id'] = oracle_greedy_kaiqiang(data[args.doc_key], data[args.sum_key])
    return data

def process_dataset(dataset, update_fn, args, str_app=''):
    """Caller for update_data to pre-process data"""
    updated_dataset = dataset.map(update_fn, num_proc=args.num_jobs)
    save_dir = os.path.join(args.data_dir, args.data_set, args.hf_dir+str_app)
    create_dir(save_dir)
    updated_dataset.save_to_disk(save_dir)
    print (f'processed data is stored at {save_dir}')

def load_text(file):
    with open(file, 'r') as f:
        data = f.readlines()
    return data

def add_model_specific_args(parser):
    parser = argparse.ArgumentParser(description="generate oracle sentences")
    parser.add_argument("--num_jobs", type=int, default=100)
    parser.add_argument("--data_set", type=str, default="openasp", choices=["cnndm", "openasp"])    
    parser.add_argument("--data_dir", type=str, default="~/workspace/data", help="source dataset directory or file")
    parser.add_argument("--hf_dir", type=str, default='huggingface_dataset')
    parser.add_argument("--cache_eval_dir", type=str, default='/data/home/swcho/workspace/huggingface_evaluation_metric')
    parser.add_argument("--process_train", action="store_true")
    parser.add_argument("--process_val", action="store_true")
    parser.add_argument("--process_test", action="store_true")
    return parser

if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(description="summarization")
    parser = add_model_specific_args(main_arg_parser)
    args = parser.parse_args()

    # data
    if args.data_set == "cnndm":
        dataset = load_dataset('ccdv/cnn_dailymail', '3.0.0')
        args.doc_key, args.sum_key = 'article', 'highlights'
        update_fn = update_data

    elif args.data_set == "openasp":
        save_dir = os.path.join(args.data_dir, args.data_set, args.hf_dir)
        if os.path.exists(save_dir):
            dataset = load_from_disk(save_dir)
        else:
            dataset_dir = os.path.join(args.data_dir, args.data_set)
            data_file_train = {"train": dataset_dir + '/train_matched.jsonl'}
            data_file_val = {"validation": dataset_dir + '/valid_matched.jsonl'}
            data_file_test = {"test": dataset_dir + '/test_matched.jsonl'}
            if args.process_train:
                dataset_train = load_dataset("json", data_files=data_file_train, split=[f"train[{k}%:{k+10}%]" for k in range(0, 100, 10)])
            if args.process_val:
                dataset_val = load_dataset("json", data_files=data_file_val)
            if args.process_test:
                dataset_test = load_dataset("json", data_files=data_file_test)

            # dataset_dir = os.path.join(args.data_dir, args.data_set)
            # keys = ['title', 'document', 'aspect', 'query', 'aspect_sents', 'summary']
            # dataset_dict = {}
            # for split in ['train', 'valid', 'test']:
            #     lines = load_text(os.path.join(dataset_dir, f'{split}_matched.jsonl'))
            #     data = [json.loads(line) for line in lines]
            #     data_dict = {k:[] for k in keys}
            #     for dt in data:
            #         for k in keys:
            #             data_dict[k].append(dt[k])
            #     dataset_dict[split] = Dataset.from_dict(data_dict)
            # dataset = DatasetDict(dataset_dict)

            # create_dir(save_dir)
            # dataset.save_to_disk(save_dir)
        args.doc_key, args.sum_key = 'document', 'summary'
        update_fn = update_data_openasp
    
    # process data
    if args.process_test:
        str_app = '_test'
        process_dataset(dataset_test, update_fn, args, str_app)

        # test oracle rouge
        save_dir = os.path.join(args.data_dir, args.data_set, args.hf_dir+str_app)
        dataset_processed = load_from_disk(save_dir)
        refs, preds = [], []
        for data in dataset_processed['test']:
            input_txt = data[args.doc_key]
            summary = data[args.sum_key]
            oracle_id = data['oracle_id']
            refs.append('\n'.join(summary))
            preds.append('\n'.join(input_txt[oid] for oid in oracle_id))
        scores = compute_rouge_scores(refs, preds)
        names = ['rouge1_P', 'rouge1_R', 'rouge1_F1',
                'rouge2_P', 'rouge2_R', 'rouge2_F1',
                'rougeL_P', 'rougeL_R', 'rougeL_F1',
                'rougeLsum_P', 'rougeLsum_R', 'rougeLsum_F1']
        print(dict(zip(*[names, scores])))

    # ref count: 1.7784 (1, 34, 1.5228)

    # kaiqiang
    # {'rouge1_P': 0.4283223686660191, 'rouge1_R': 0.5654780805610575, 'rouge1_F1': 0.46842928647292364, 
    # 'rouge2_P': 0.21251665557778224, 'rouge2_R': 0.2701378167257998, 'rouge2_F1': 0.22735066693868985, 
    # 'rougeL_P': 0.304164605335233, 'rougeL_R': 0.39692657941270065, 'rougeL_F1': 0.32975099886408327, 
    # 'rougeLsum_P': 0.3599098599748398, 'rougeLsum_R': 0.47088610724751334, 'rougeLsum_F1': 0.3917143353196488}

    # N * 3
    # {'rouge1_P': 0.42821981922959645, 'rouge1_R': 0.5657044213807274, 'rouge1_F1': 0.46844773871066747, 
    # 'rouge2_P': 0.21247969558469795, 'rouge2_R': 0.27023290648969966, 'rouge2_F1': 0.22736321708542612, 
    # 'rougeL_P': 0.30409635401878593, 'rougeL_R': 0.39701049160406776, 'rougeL_F1': 0.32973877405120067, 
    # 'rougeLsum_P': 0.35982591638551226, 'rougeLsum_R': 0.47108700796428565, 'rougeLsum_F1': 0.39173558906390415}

    # N * 4
    # {'rouge1_P': 0.4282150051457807, 'rouge1_R': 0.5657194626477775, 'rouge1_F1': 0.46844940779885136, 
    # 'rouge2_P': 0.21247816841619507, 'rouge2_R': 0.27023934309071385, 'rouge2_F1': 0.2273644211126464, 
    # 'rougeL_P': 0.3040925586240326, 'rougeL_R': 0.39701671227519414, 'rougeL_F1': 0.32973828111078474, 
    # 'rougeLsum_P': 0.3598224839981669, 'rougeLsum_R': 0.4711000202831269, 'rougeLsum_F1': 0.3917375845557689}

    # me
    # {'rouge1_P': 0.45265018541230473, 'rouge1_R': 0.4901407518642115, 'rouge1_F1': 0.4497389152746894, 
    # 'rouge2_P': 0.22616468549200286, 'rouge2_R': 0.23785980040403917, 'rouge2_F1': 0.22009558276417454, 
    # 'rougeL_P': 0.32665357595485356, 'rougeL_R': 0.35287377618657334, 'rougeL_F1': 0.32258835035284894, 
    # 'rougeLsum_P': 0.37832508030876566, 'rougeLsum_R': 0.40753106417601276, 'rougeLsum_F1': 0.37472709354291095}

    # target = 3
    # {'rouge1_P': 0.4349853574986879, 'rouge1_R': 0.5423168155094169, 'rouge1_F1': 0.46004213918217374, 
    # 'rouge2_P': 0.2159094603258727, 'rouge2_R': 0.2597718950674617, 'rouge2_F1': 0.2235886656820111, 
    # 'rougeL_P': 0.3009852911255448, 'rougeL_R': 0.3759598779718684, 'rougeL_F1': 0.31790922646311415, 
    # 'rougeLsum_P': 0.3652659498124805, 'rougeLsum_R': 0.4500704788131912, 'rougeLsum_F1': 0.3837783012953569}

    # validation
    if args.process_val:
        str_app = '_val'
        process_dataset(dataset_val, update_fn, args, str_app)

    # train
    if args.process_train:
        for i, data_train in enumerate(dataset_train):
            print (f'train {i} processing')
            str_app = f'_train_{i}:{i+10}%'
            process_dataset(DatasetDict({f'train':data_train}), update_fn, args, str_app)
