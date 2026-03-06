# name=reverie_goat_cfp
# DATA_ROOT=../datasets
# export NLTK_DATA="/workspace/VLN-GOAT/nltk_data"
# train_alg=dagger
# features=clip768
# ft_dim=768
# obj_features=vitbase
# obj_ft_dim=768

# ngpus=1
# seed=0

# outdir=${DATA_ROOT}/REVERIE/
# augdir=${DATA_ROOT}/REVERIE/annotations/REVERIE_aug_roberta_enc.json
# reverie_pretrain_file=${DATA_ROOT}/REVERIE/pretrain/goat_reverie_pretrain_vggt/ckpts/model_step_best.pt

# flag="--root_dir ${DATA_ROOT}
#       --dataset reverie
#       --output_dir ${outdir}
#       --world_size ${ngpus}
#       --seed ${seed}
#       --tokenizer roberta
#       --mode extract_cfp_features
#       --name ${name}

#       --enc_full_graph
#       --graph_sprels
#       --fusion dynamic
#       --multi_endpoints

#       --dagger_sample sample

#       --train_alg ${train_alg}
      
#       --num_l_layers 6
#       --num_x_layers 3
#       --num_pano_layers 2
      
#       --max_action_len 15
#       --max_instr_len 80
#       --max_objects 20

#       --batch_size 12

#       --features ${features}
#       --obj_features ${obj_features}
#       --image_feat_size ${ft_dim}
#       --angle_feat_size 4
#       --obj_feat_size ${obj_ft_dim}

#       --ml_weight 0.2
#       "

# # train
# CUDA_VISIBLE_DEVICES='1' python -u reverie/main_nav_obj.py $flag  \
#       --bert_ckpt_file ${reverie_pretrain_file}

name=reverie_goat_cfp_vgllm_add_vggtaddregister20w
DATA_ROOT=../datasets
export NLTK_DATA="/workspace/VLN-GOAT/nltk_data"
train_alg=dagger
features=clip768
ft_dim=768
obj_features=vitbase
obj_ft_dim=768

ngpus=1
seed=0

outdir=${DATA_ROOT}/REVERIE/vgllm_after_backcausal/
augdir=${DATA_ROOT}/REVERIE/annotations/REVERIE_aug_roberta_enc.json

# 你的预训练模型路径
reverie_pretrain_file=${DATA_ROOT}/REVERIE/pretrain/goat_reverie_pretrain_vgllmadd_register/ckpts/model_step_best.pt

flag="--root_dir ${DATA_ROOT}
      --dataset reverie
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer roberta
      --mode extract_cfp_features
      --name ${name}

      --enc_full_graph
      --graph_sprels
      --fusion dynamic
      --multi_endpoints

      --dagger_sample sample

      --train_alg ${train_alg}
      
      --num_l_layers 6
      --num_x_layers 3
      --num_pano_layers 2
      
      --max_action_len 15
      --max_instr_len 80
      --max_objects 20

      --batch_size 12

      --features ${features}
      --obj_features ${obj_features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4
      --obj_feat_size ${obj_ft_dim}

      --ml_weight 0.2
      "

# ==================== [最终修改] ====================
# 1. 这里的 --bert_ckpt_file 被删除了，避免报错。
# 2. 保留 --resume_file，它会在模型初始化后，把所有训练好的参数覆盖上去。
CUDA_VISIBLE_DEVICES='1' python -u reverie/main_nav_obj.py $flag  \
      --resume_file ${reverie_pretrain_file}
# ====================================================