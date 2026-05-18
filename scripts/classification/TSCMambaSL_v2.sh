#!/usr/bin/env bash
# TSCMamba + MambaSL (Variant 2): the MambaSL branch drops the top Linear of
# the MambaBlock (out_proj) so Xsl is the post-gate hidden state.
#
# The only difference from `TSCMambaSL.sh` is `--mambasl_skip_out_proj 1`.
# Branch flow:
#   raw MTS -> Conv1d + sinusoidal PE + Dropout
#   -> Mamba_TimeVariant (in_proj, conv1d, Modular Selective SSM, gate)   <-- no out_proj
#   -> LayerNorm(d_inner) -> SiLU
#   -> AdaptiveAvgPool1d(enc_in) -> Linear(d_inner, projected_space*3)
#   -> Xsl  (B, enc_in, projected_space*3)
#
# Everything outside the MambaSL branch is identical to TSCMambaSL.sh:
#   x3 = x1 + x2 + alpha * Xsl  ->  pool + flatten + TSCMamba classifier

set -euo pipefail

if [ ! -d "./logs" ]; then mkdir ./logs; fi
if [ ! -d "./logs/classification" ]; then mkdir ./logs/classification; fi
if [ ! -d "./csv_results" ]; then mkdir ./csv_results; fi
if [ ! -d "./csv_results/classification" ]; then mkdir ./csv_results/classification; fi

model_name=TSCMambaSL

root_path_name=./dataset/PhonemeSpectra
model_id_name=PhonemeSpectra_v2
data_name=UEA

random_seed=2024

# Heartbeat (UEA): 2 classes, 61 channels, max_seq_len ~= 405.
# MambaSL paper's best Heartbeat setting (an LTI single-layer Mamba):
#   d_model=64, d_state=16, num_kernels=9, tv_dt=0, tv_B=0, tv_C=0, use_D=0

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
    --no_rocket 0 \
    --additive_fusion 1 \
    --max_pooling 0 \
    --mambasl_d_model 256 \
    --mambasl_d_state 4 \
    --mambasl_d_conv 4 \
    --mambasl_expand 1 \
    --mambasl_kernel 5 \
    --tv_dt 1 \
    --tv_B 1 \
    --tv_C 0 \
    --use_D 0 \
    --mambasl_alpha_init 1 \
    --mambasl_alpha_learnable 1 \
    --mambasl_skip_out_proj 1 \
    --des 'TSCMambaSL_v2_PhonemeSpectra' \
    --lradj 'cosine' \
    --comment 'Variant 2: TSCMamba + MambaSL without MambaBlock out_proj' \
    --train_epochs 200 \
    --itr 1 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --devices 0 \
    >logs/classification/${model_name}_${model_id_name}.log
