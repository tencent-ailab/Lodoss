# [Arxiv]
# epoch: 12, warmup: 0.083 (=1/12) => increase epoch 20 / warmup 0.1
# [Pubmed]
# epoch: 20, warmup: 0.1

# training
# ranking loss
python run_summarization_ext.py --train_on --batch_size 16 --grad_accum 4 --epochs 10 --grad_ckpt --fp16 --warmup 0.1 --data_dir ~/workspace/data --data_set cnndm --model_name allenai/longformer-base-4096 --limit_train_batches 1.0 --limit_val_batches 1.0 --limit_test_batches 1.0  --increasing_window_size --num_sent_lim 55 --num_sent_inf 3 --sort_sum_pred --block_trigram --every_n_train_steps 0.25 --local_backbone_dir

# test
# large model: --gpus 0 1 2 
# base model: --gpus 0 1 2 3 4 5 6
python run_summarization_ext.py --batch_size 1 --grad_accum 1 --epochs 1 --grad_ckpt --fp16 --data_dir ./data --data_set arxiv --model_name allenai/longformer-base-4096 --limit_train_batches 1.0 --limit_val_batches 1.0 --limit_test_batches 1.0  --increasing_window_size --num_sent_inf 3 --sort_sum_pred --block_trigram --num_sent_lim 55 --from_pretrained model_train/longformer-base-4096_arxiv_paper/epoch\=...


# longer Longformer (16K)
--is_longer_seq --max_input_len 16384


# openasp
python run_ext_summ.py --train_on --batch_size 2 --grad_accum 8 --epochs 20 --grad_ckpt --fp16 --warmup 0 --data_dir ~/workspace/data --data_set openasp --hf_dir huggingface_dataset_tokenized_4096 --limit_train_batches 1.0 --limit_val_batches 0 --limit_test_batches 1.0 --every_n_train_steps 1.0 --local_backbone_dir ~/workspace/pretrained_models --max_input_len 4096 --model_name allenai/longformer-large-4096 --n_transformer_layer 2 --n_transformer_head 1 --lr 5e-5 --save_top_k -1 --increasing_window_size --cache_eval_dir ~/workspace/huggingface_evaluation_metric --is_test_on_dir --every_n_epochs 1 --optimizer adamw 
# --num_sent_inf 2 --gpus 0 1 --refresh_rate 10