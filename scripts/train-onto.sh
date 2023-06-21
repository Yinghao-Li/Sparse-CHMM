#!/bin/bash

# Quit if there are any errors
set -e

BERT_MODEL="bert-base-uncased"
TRAIN_PATH="./data/OntoNotes/train.json"
VALID_PATH="./data/OntoNotes/valid.json"
TEST_PATH="./data/OntoNotes/test.json"

LM_BATCH_SIZE=32

NUM_LM_NN_PRETRAIN_EPOCHS=3
NUM_LM_TRAIN_EPOCHS=50
NUM_LM_VALID_TOLERANCE=6
NUM_LM_VALID_SMOOTHING=1

CONC_BASE=2
CONC_MAX=1000
TRAINING_RATIO=0.1

WXOR_TEMPERATURE=1
NONDIAG_SPLIT_RATIO=0.2
NONDIAG_SPLIT_DECAY=1

NN_LR=0.0001
PRETRAIN_LR=0.0001
S2_LR_DECAY=0.2
S3_LR_DECAY=1

NONDIAG_EXP=4

DIAG_EXP_T1=1
DIAG_EXP_T2=1.1

for SEED in 0 1 2 3 4
do

OUTPUT_DIR="./output/OntoNotes/${SEED}/"

PYTHONPATH="." CUDA_VISIBLE_DEVICES=$1 python ./run/train.py \
    --bert_model_name_or_path $BERT_MODEL \
    --train_path $TRAIN_PATH \
    --valid_path $VALID_PATH \
    --test_path $TEST_PATH \
    --output_dir $OUTPUT_DIR \
    --save_dataset \
    --save_dataset_to_data_dir \
    --load_preprocessed_dataset \
    --training_ratio_per_epoch $TRAINING_RATIO \
    --save_init_mat \
    --load_init_mat \
    --pretrain_lr $PRETRAIN_LR \
    --nn_lr $NN_LR \
    --s2_lr_decay $S2_LR_DECAY \
    --s3_lr_decay $S3_LR_DECAY \
    --batch_size $LM_BATCH_SIZE \
    --num_lm_nn_pretrain_epochs $NUM_LM_NN_PRETRAIN_EPOCHS \
    --num_lm_train_epochs $NUM_LM_TRAIN_EPOCHS \
    --num_lm_valid_tolerance $NUM_LM_VALID_TOLERANCE \
    --num_lm_s3_valid_tolerance $NUM_LM_VALID_TOLERANCE \
    --num_lm_valid_smoothing $NUM_LM_VALID_SMOOTHING \
    --include_s2 \
    --include_s3 \
    --freeze_s2_base_emiss \
    --seed $SEED \
    --dirichlet_conc_base $CONC_BASE \
    --dirichlet_conc_max $CONC_MAX \
    --diag_exp_t1 $DIAG_EXP_T1 \
    --diag_exp_t2 $DIAG_EXP_T2 \
    --nondiag_exp $NONDIAG_EXP \
    --nondiag_split_ratio $NONDIAG_SPLIT_RATIO \
    --nondiag_split_decay $NONDIAG_SPLIT_DECAY \
    --wxor_temperature $WXOR_TEMPERATURE \
    --calculate_wxor_on_valid

done
