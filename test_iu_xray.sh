#!/bin/bash

# 设置Python路径
export PYTHONPATH=/root/autodl-tmp/R2GenCMN-main:$PYTHONPATH

python main_train.py \
    --image_dir data/iu_xray/images/ \
    --ann_path data/iu_xray/annotation.json \
    --dataset_name iu_xray \
    --max_seq_length 60 \
    --threshold 3 \
    --epochs 100 \
    --batch_size 16 \
    --lr_ve 1e-4 \
    --lr_ed 5e-4 \
    --step_size 10 \
    --gamma 0.8 \
    --num_layers 3 \
    --topk 32 \
    --cmm_size 512 \
    --d_model 512 \
    --cmm_dim 256 \
    --hd_dim 256 \
    --use_hd_memory False \
    --hd_memory_weight 0.8 \
    --update_hd_memory False \
    --hd_topk 16 \
    --seed 7580 \
    --beam_size 3 \
    --save_dir results/iu_xray_cmn_crft/ \
    --log_period 50 \
    --early_stop 10 \
    --use_crft True \
    --aug_flip True \
    --aug_intensity 0.1 \
    --aug_rotation 5 \
    --aug_crop_scale 0.9 \