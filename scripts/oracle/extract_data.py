import os
from datasets import load_from_disk, concatenate_datasets, DatasetDict
import pickle

data_dir = '/data/home/swcho/workspace/data/openasp'

def save_pkl(data_pkl, file):
    with open(file, 'wb') as fb:
        pickle.dump(data_pkl, fb)

hfile_list = []
for hfile in os.listdir(data_dir):
    if hfile.startswith('huggingface'):
        hfile_list.append(hfile)

train, val, test = [], [], []
for hfile in hfile_list:
    if 'train' in hfile:
        train.append(hfile)
    elif 'val' in hfile:
        val.append(hfile)
    elif 'test' in hfile:
        test.append(hfile)
print (len(train), train)
print (len(val), val)
print (len(test), test)

def concat(datasets, key):
    dataset_list = []
    for hfile in sorted(datasets):
        data = load_from_disk(os.path.join(data_dir, hfile))
        dataset_list.append(data[key])
    return concatenate_datasets(dataset_list)

# train_dataset = concat(train, 'train')
# print (train_dataset)
val_dataset = concat(val, 'validation')
print (val_dataset)
# test_dataset = concat(test, 'test')
# print (test_dataset)

# oracle_ids = (train_dataset['oracle_id'], val_dataset['oracle_id'], test_dataset['oracle_id'])
oracle_ids = (val_dataset['oracle_id'])
save_pkl(oracle_ids, os.path.join(data_dir, 'oracle_val.pkl'))

# dataset = DatasetDict({'train': train_dataset, 'validation': val_dataset, 'test': test_dataset}) # 'validation': dataset_val_merged,
# dataset.save_to_disk(os.path.join(data_dir, 'huggingface_dataset'))