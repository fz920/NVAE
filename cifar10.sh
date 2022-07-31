#!/bin/bash
export EXPR_ID=2
export DATA_DIR=/data2/users/fz920/data
export CHECKPOINT_DIR=/data2/users/fz920/NVAE/checkpoint
export CODE_DIR=/data2/users/fz920/NVAE
cd $CODE_DIR
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --master_port 1235 train.py --data $DATA_DIR/cifar10 --root $CHECKPOINT_DIR --save $EXPR_ID --dataset cifar10 \
        --num_channels_enc 16 --num_channels_dec 16 --epochs 1 --num_postprocess_cells 2 --num_preprocess_cells 2 \
        --num_latent_scales 1 --num_latent_per_group 20 --num_cell_per_cond_enc 1 --num_cell_per_cond_dec 1 \
        --num_preprocess_blocks 1 --num_postprocess_blocks 1 --num_groups_per_scale 10 --batch_size 32 \
        --weight_decay_norm 1e-2 --num_nf 0 --num_process_per_node 1 --use_se --res_dist --fast_adamax 
