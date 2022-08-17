export EXPR_ID=celeb1
export DATA_DIR=/data/users/fz920/data
export CHECKPOINT_DIR=/data/users/fz920/NVAE/checkpoint
export CODE_DIR=/data/users/fz920/NVAE

cd $CODE_DIR

CUDA_VISIBLE_DEVICES=1 python3 train.py --data $DATA_DIR/celeba64_lmdb --root $CHECKPOINT_DIR --save $EXPR_ID --dataset celeba_64 \
        --num_channels_enc 64 --num_channels_dec 64 --epochs 90 --num_postprocess_cells 2 --num_preprocess_cells 2 \
        --num_latent_scales 3 --num_latent_per_group 20 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 \
        --num_preprocess_blocks 1 --num_postprocess_blocks 1 --weight_decay_norm 1e-1 --num_groups_per_scale 20 \
        --batch_size 16 --num_nf 1 --ada_groups --num_process_per_node 1 --use_se --res_dist --fast_adamax