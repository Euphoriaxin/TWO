#!/bin/bash
# train_iu_xray.sh
# HDC + 多正样本 MAL 修复；mal_weight=0.1 为候选值，可在末尾覆盖。
set -e

export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

cd "$(dirname "$0")"

python scripts/setup_meteor.py

python main_train.py \
    --image_dir       data/iu_xray/images/ \
    --ann_path        data/iu_xray/annotation.json \
    --dataset_name    iu_xray \
    --visual_extractor resnet101 \
    --max_seq_length  60 \
    --threshold       3 \
    --epochs          80 \
    --batch_size      16 \
    --lr_ve           1e-4 \
    --lr_ed           5e-4 \
    --optim           AdamW \
    --weight_decay    1e-4 \
    --lr_scheduler    CosineWithWarmup \
    --warmup_epochs   3 \
    --min_lr          1e-7 \
    --num_layers      3 \
    --num_heads       8 \
    --d_model         512 \
    --d_ff            512 \
    --dropout         0.1 \
    --seed            7580 \
    --sample_method   greedy \
    --beam_size       1 \
    --length_penalty  0.0 \
    --save_dir        results/iu_xray_hdc_mal_fixed_w01 \
    --log_period      50 \
    --hd_dim          5000 \
    --hdc_scale       0.3 \
    --use_hd_memory   true \
    --update_hd_memory true \
    --hd_memory_weight 0.2 \
    --hd_memory_temperature 0.1 \
    --num_experts     4 \
    --mal_alpha       0.1 \
    --mal_tau         0.1 \
    --mal_weight      0.1 \
    --use_evmoe       true \
    --use_mal         true \
    --monitor_metric  BLEU_4 \
    --early_stop      10 \
    "$@"



