import os
import pickle
from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict, Dataset
import evaluate
from tqdm import tqdm

data_dir = '/data/home/swcho/workspace/data/openasp'
# data_dir = '/data2/swcho_data/'
cache_eval_dir = '/data/home/swcho/workspace/huggingface_evaluation_metric'
# cache_eval_dir = '/apdcephfs_cq2/share_1603164/data/huggingface_evaluate_metrics'

def load_pkl(file):
    if not os.path.exists(file):
        print('{} does not exist'.format(file))
        return None

    with open(file, 'rb') as fb:
        data = pickle.load(fb)
    return data

data_file_train = {"train": data_dir + '/train_matched.jsonl'}
data_file_val = {"validation": data_dir + '/valid_matched.jsonl'}
data_file_test = {"test": data_dir + '/test_matched.jsonl'}
dataset_train = load_dataset("json", data_files=data_file_train)
dataset_val = load_dataset("json", data_files=data_file_val)
dataset_test = load_dataset("json", data_files=data_file_test)

data_oracle_tt = load_pkl(os.path.join(data_dir, 'oracle_tt.pkl'))
data_oracle_v = load_pkl(os.path.join(data_dir, 'oracle_val.pkl'))

data_oracle_train = Dataset.from_dict({'oracle_id': data_oracle_tt[0]})
print (dataset_train, data_oracle_train)
dataset_train_merged = concatenate_datasets([dataset_train['train'], data_oracle_train], axis=1)
data_oracle_val = Dataset.from_dict({'oracle_id': data_oracle_v})
print (dataset_val, data_oracle_val)
dataset_val_merged = concatenate_datasets([dataset_val['validation'], data_oracle_val], axis=1)
data_oracle_test = Dataset.from_dict({'oracle_id': data_oracle_tt[2]})
print (dataset_test, data_oracle_test)
dataset_test_merged = concatenate_datasets([dataset_test['test'], data_oracle_test], axis=1)

dataset = DatasetDict({'train': dataset_train_merged, 'validation': dataset_val_merged, 'test': dataset_test_merged}) # 'validation': dataset_val_merged,
dataset.save_to_disk(os.path.join(data_dir, 'huggingface_dataset'))

'''
# check rouge score in testset
rouge_eval = evaluate.load(f'{cache_eval_dir}/rouge', use_stemmer=True)

dataset = load_from_disk('huggingface_dataset')

def compute_metrics(document, reference, oracle):
    decoded_preds = "\n".join([document[oid] for oid in oracle])
    decoded_labels = "\n".join(reference)
    rouge_eval.add_batch(predictions=[decoded_preds], references=[decoded_labels])

print ('rouge test...')
for data in tqdm(dataset['test']):
    compute_metrics(data['document'], data['summary'], data['oracle_id'])

rouge_scores = rouge_eval.compute(use_stemmer=True)
print (rouge_scores)
# {'rouge1': 0.46846221251232245, 'rouge2': 0.22735681742619085, 'rougeL': 0.32972328721415384, 'rougeLsum': 0.39175054511535434}
'''