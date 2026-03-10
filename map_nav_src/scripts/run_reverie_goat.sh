name=goat_reverie
DATA_ROOT=../datasets
export NLTK_DATA="/workspace/VLN-GOAT/nltk_data"

# ==================== [路径配置] ====================
# 1. 预训练模型路径 (Pre-training 产出的 best_model.pt)
# 请务必确认这个路径和你上一步提取特征时用的一样
reverie_pretrain_file=${DATA_ROOT}/REVERIE/pretrain/goat_reverie_pretrain_vgllmadd_register/ckpts/model_step_best.pt

# 2. CFP 特征文件路径 (你刚刚训练出来的)
# 直接填你提供的路径
cfp_features_file=/workspace/VLN-GOAT/datasets/REVERIE/test/reverie_goat_cfp/logs/cfp_features.tsv
# ====================================================

train_alg=dagger
features=clip768
ft_dim=768
obj_features=vitbase
obj_ft_dim=768
ngpus=1
seed=0

outdir=${DATA_ROOT}/REVERIE/after_backcausal_imagecaption/
aug_file=${DATA_ROOT}/REVERIE/annotations/REVERIE_train_aug_roberta_enc.json
speaker_file=${DATA_ROOT}/REVERIE/speaker/transpeaker_reverie/ckpts/best_both_bleu.pt

flag="--root_dir ${DATA_ROOT}
      --dataset reverie
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer roberta
      --mode train
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

      --batch_size 8
      --lr 2e-5
      --iters 100000
      --log_every 1000
      --optim adamW

      --features ${features}
      --obj_features ${obj_features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4
      --obj_feat_size ${obj_ft_dim}

      --ml_weight 0.2

      --feat_dropout 0.6
      --dropout 0.1

      --use_transpeaker
      --accumulateGrad
      --aug ${aug_file}
      --speaker ${speaker_file}

      --do_back_txt
      --do_back_img
      --do_back_txt_type type_2
      --do_back_imgobj_type type_1
      --do_add_method door
      --z_instr_update

      --do_front_txt
      --do_front_img
      --do_front_his
      --front_feat_file 
      "

# ==================== [关键修改] ====================
# 1. 使用 --resume_file 加载预训练权重 (Pre-train)
# 2. 删除了 --bert_ckpt_file 以避免 IsADirectoryError 报错
# 3. 脚本 flag 中已经加入了 --front_feat_file 来读取 CFP 特征
CUDA_VISIBLE_DEVICES='1' python -u reverie/main_nav_obj.py $flag \
      --resume_file ${reverie_pretrain_file}
# ====================================================