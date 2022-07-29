#!/bin/bash
UNIQUE_EXPR_ID=1
PATH_TO_DATA_DIR=/data2/users/fz920/NVAE/data
PATH_TO_CODE_DIR=/data2/users/fz920/NVAE
PATH_TO_CHECKPOINT_DIR=/data2/users/fz920/NVAE/checkpoint
export EXPR_ID=UNIQUE_EXPR_ID
export DATA_DIR=PATH_TO_DATA_DIR
export CHECKPOINT_DIR=PATH_TO_CHECKPOINT_DIR
export CODE_DIR=PATH_TO_CODE_DIR
CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --master_port 66660 train.py --data $DATA_DIR/cifar10 --root $CHECKPOINT_DIR --save $EXPR_ID --dataset cifar10 \
        --num_channels_enc 16 --num_channels_dec 16 --epochs 1 --num_postprocess_cells 2 --num_preprocess_cells 2 \
        --num_latent_scales 1 --num_latent_per_group 20 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 \
        --num_preprocess_blocks 1 --num_postprocess_blocks 1 --num_groups_per_scale 30 --batch_size 32 \
        --weight_decay_norm 1e-2 --num_nf 0 --num_process_per_node 1 --use_se --res_dist --fast_adamax 
