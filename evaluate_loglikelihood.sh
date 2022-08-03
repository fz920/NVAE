export EXPR_ID=5
export DATA_DIR=/data/users/fz920/data
export CHECKPOINT_DIR=/data/users/fz920/NVAE/checkpoint
export CODE_DIR=/data/users/fz920/NVAE

cd $CODE_DIR
CUDA_VISIBLE_DEVICES=0 python evaluate.py --checkpoint $CHECKPOINT_DIR/eval-$EXPR_ID/checkpoint.pt --data $DATA_DIR/cifar10 --eval_mode=evaluate --num_iw_samples=1000