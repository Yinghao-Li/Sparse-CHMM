#!/bin/bash

# Quit if there are any errors
set -e

BERT_MODEL="bert-base-uncased"
TRAIN_PATH="../data_constr/CoNLL/train.json"
VALID_PATH="../data_constr/CoNLL/valid.json"
TEST_PATH="../data_constr/CoNLL/test.json"

LM_BATCH_SIZE=256

NUM_LM_NN_PRETRAIN_EPOCHS=5
NUM_LM_TRAIN_EPOCHS=100
NUM_LM_VALID_TOLERANCE=20
NUM_LM_VALID_SMOOTHING=5

CONC_BASE=10
CONC_MAX=1000

WXOR_TEMPERATURE=1
NONDIAG_SPLIT_RATIO=0.5
NONDIAG_SPLIT_DECAY=0.1

DIAG_EXP_T1=1.5
DIAG_EXP_T2=1.2
NONDIAG_EXP=4

NN_LR=0.0002
S2_LR_DECAY=0.2
S3_LR_DECAY=1

for SEED in 0 1 2 3 4
do

OUTPUT_DIR="./output/Co03/${SEED}/"

CUDA_VISIBLE_DEVICES=$1 python sparse-chmm-train.py \
    --bert_model_name_or_path $BERT_MODEL \
    --train_path $TRAIN_PATH \
    --valid_path $VALID_PATH \
    --test_path $TEST_PATH \
    --output_dir $OUTPUT_DIR \
    --save_dataset \
    --save_dataset_to_data_dir \
    --load_preprocessed_dataset \
    --nn_lr $NN_LR \
    --s2_lr_decay $S2_LR_DECAY \
    --s3_lr_decay $S3_LR_DECAY \
    --lm_batch_size $LM_BATCH_SIZE \
    --num_lm_nn_pretrain_epochs $NUM_LM_NN_PRETRAIN_EPOCHS \
    --num_lm_train_epochs $NUM_LM_TRAIN_EPOCHS \
    --num_lm_valid_tolerance $NUM_LM_VALID_TOLERANCE \
    --num_lm_s3_valid_tolerance $NUM_LM_VALID_TOLERANCE \
    --num_lm_valid_smoothing $NUM_LM_VALID_SMOOTHING \
    --freeze_s2_base_emiss \
    --include_s2 \
    --include_s3 \
    --seed $SEED \
    --dirichlet_conc_base $CONC_BASE \
    --dirichlet_conc_max $CONC_MAX \
    --diag_exp_t1 $DIAG_EXP_T1 \
    --diag_exp_t2 $DIAG_EXP_T2 \
    --nondiag_exp $NONDIAG_EXP \
    --nondiag_split_ratio $NONDIAG_SPLIT_RATIO \
    --nondiag_split_decay $NONDIAG_SPLIT_DECAY \
    --wxor_temperature $WXOR_TEMPERATURE

done
