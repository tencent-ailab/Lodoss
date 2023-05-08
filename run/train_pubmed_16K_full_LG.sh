python scripts/run_long_summ.py --train_on \
--batch_size 1 \
--grad_accum 64 \
--epochs 20  \
--grad_ckpt \
--data_dir [your_data_path] \
--data_set pubmed \
--num_sent_inf 7 \
--model_name allenai/longformer-large-4096 \
--increasing_window_size \
--sort_sum_pred \
--max_input_len 16384 \
--is_longer_seq \
--save_top_k 3 \
--every_n_train_steps 2 \
--is_seg \
--is_dpp \
--dpp_weight 0.1
