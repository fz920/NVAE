#!/bin/bash
export EXPR_ID=5
export DATA_DIR=/data/users/fz920/data
export CHECKPOINT_DIR=/data/users/fz920/NVAE/checkpoint
export CODE_DIR=/data/users/fz920/NVAE
cd $CODE_DIR
CUDA_VISIBLE_DEVICES=1 python train.py --data $DATA_DIR/cifar10 --root $CHECKPOINT_DIR --save $EXPR_ID --dataset cifar10 \
        --num_channels_enc 128 --num_channels_dec 128 --epochs 20 --num_postprocess_cells 2 --num_preprocess_cells 2 \
        --num_latent_scales 1 --num_latent_per_group 20 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 \
        --num_preprocess_blocks 1 --num_postprocess_blocks 1 --num_groups_per_scale 30 --batch_size 32 \
        --weight_decay_norm 1e-2 --num_nf 1 --num_process_per_node 1 --use_se --res_dist --fast_adamax
