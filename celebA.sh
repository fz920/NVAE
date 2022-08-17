export DATA_DIR=/Users/zhangfengzhe/Desktop/urop_generative_model/data
export CODE_DIR=/Users/zhangfengzhe/Desktop/urop_generative_model/NVAE

cd $CODE_DIR/scripts

python create_celeba64_lmdb.py --split train --img_path $DATA_DIR/celeba_org --lmdb_path $DATA_DIR/celeba64_lmdb
python create_celeba64_lmdb.py --split valid --img_path $DATA_DIR/celeba_org --lmdb_path $DATA_DIR/celeba64_lmdb
python create_celeba64_lmdb.py --split test  --img_path $DATA_DIR/celeba_org --lmdb_path $DATA_DIR/celeba64_lmdb