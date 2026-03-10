#!/bin/bash
name=goat_reverie_pretrain_vgllmadd_register
DATA_ROOT=../datasets/REVERIE/
NODE_RANK=0
NUM_GPUS=1  # 如果你只想用单卡测试，这里设为1
export NLTK_DATA="/workspace/VLN-GOAT/nltk_data"
# 注意：每行末尾的 \ 后面千万不要有空格
CUDA_VISIBLE_DEVICES='1' python -m torch.distributed.launch \
    --use-env \
    --nproc_per_node=${NUM_GPUS} \
    --node_rank $NODE_RANK \
    --master_port 8889 \
    train_reverie_goat.py \
    --world_size ${NUM_GPUS} \
    --name ${name} \
    --vlnbert cmt \
    --model_config config/reverie_GOAT_model_config.json \
    --config config/reverie_GOAT_pretrain.json \
    --root_dir ${DATA_ROOT} \
    --cuda_first_device 0
