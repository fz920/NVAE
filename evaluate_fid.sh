export EXPR_ID=4
export DATA_DIR=/data/users/fz920/data
export CHECKPOINT_DIR=/data/users/fz920/NVAE/checkpoint
export CODE_DIR=/data/users/fz920/NVAE

cd $CODE_DIR

python scripts/precompute_fid_statistics.py --data $DATA_DIR/cifar10 --dataset cifar10 --fid_dir /tmp/fid-stats/
python torch.distributed.launch --nproc_per_node=1 --master_port 1102 evaluate.py --checkpoint $CHECKPOINT_DIR/eval-$EXPR_ID/checkpoint.pt --data $DATA_DIR/cifar10 --eval_mode=evaluate_fid  --fid_dir /tmp/fid-stats/ --temp=0.6 --readjust_bn