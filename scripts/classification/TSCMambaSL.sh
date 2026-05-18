#!/usr/bin/env bash
# Launch the TSCMamba + MambaSL fused architecture on a UEA dataset.
#
# Pipeline (per `models/TSCMambaSL.py`):
#   1. TSCMamba pre-encoder + dual-axis Mamba  -> x1, x2
#   2. MambaSL branch on the raw MTS           -> xsl (adapted to (B, enc_in, projected_space*3))
#   3. x3 = x1 + x2 + alpha * xsl
#   4. Pool + flatten + classifier (TSCMamba head)
#
# Requirements: --model TSCMambaSL and --add_raw_mts 1.

set -euo pipefail

if [ ! -d "./logs" ]; then mkdir ./logs; fi
if [ ! -d "./logs/classification" ]; then mkdir ./logs/classification; fi
if [ ! -d "./csv_results" ]; then mkdir ./csv_results; fi
if [ ! -d "./csv_results/classification" ]; then mkdir ./csv_results/classification; fi

model_name=TSCMambaSL

root_path_name=./dataset/AtrialFibrillation
model_id_name=AtrialFibrillation
data_name=UEA

random_seed=2024

python -u run.py \
    --task_name classification \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --model_id $model_id_name \
    --model $model_name \
    --data $data_name \
    --add_raw_mts 1 \
    --dropout 0.2 \
    --dconv 4 \
    --d_state 128 \
    --e_fact 2 \
    --projected_space 64 \
    --num_mambas 1 \
    --no_rocket 1 \
    --additive_fusion 1 \
    --max_pooling 1 \
    --mambasl_d_model 32 \
    --mambasl_d_state 16 \
    --mambasl_d_conv 4 \
    --mambasl_expand 1 \
    --mambasl_kernel 13 \
    --tv_dt 1 \
    --tv_B 0 \
    --tv_C 0 \
    --use_D 0 \
    --mambasl_alpha_init 1 \
    --mambasl_alpha_learnable 1 \
    --des 'TSCMambaSL_AtrialFibrillation' \
    --lradj 'cosine' \
    --comment 'TSCMamba + MambaSL branch (X3 = X1 + X2 + alpha*Xsl)' \
    --train_epochs 200 \
    --itr 1 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --devices 0 \
    >logs/classification/${model_name}_${model_id_name}.log
