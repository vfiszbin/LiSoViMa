#!/bin/bash
python train_dpo/train_dpo.py \
  --hf_token "..." \
  --hf_user "thdsofia" \
  --dataset_name "thdsofia/training_data_dpo" \
  --source_dataset "thdsofia/DPO_STEM_training" \
  --seed 42 \
  --base_model "thdsofia/general_sft" \
  --output_dir "./dposft-output" \
  --batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --bf16 \