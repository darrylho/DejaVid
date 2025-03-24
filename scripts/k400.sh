#!/bin/bash
set -x

python -m torch.distributed.launch  \
--nproc_per_node=$1 --node_rank=$2 --nnodes=$3    \
--master_addr="$4" --master_port=1242  \
main_dist.py  \
--num_classes 400 \
--num_feats_per_timestep 1808 \
--batch_size $(( 48*(8/$1) / $3 )) \
--seed 1337 \
--dataset K400DatasetDist \
--id2splits_pkl_path /path/to/pickle/mapping/MTS_id/to/train_or_val/ \
--id2labels_pkl_path /path/to/pickle/mapping/MTS_id/to/class_label_int/ \
--id2fname_pkl_path /path/to/pickle/mapping/MTS_id/to/MTS_filename/ \
--id2feats_pkl_path_root /path/to/root/of/trainingAndTesting/MTSs/ \
--mean_series_cached_filename_prefix /path/to/centroids/ \
--use_mm 0 \
--lr "$5" \
--first_order_penalty 0 \
--second_order_penalty 0 \
--use_fixed_path 1 \
--prefetch_factor 2 \
--num_timesteps $6 \
--ms_lr_div 3 \
--num_epochs $7 


