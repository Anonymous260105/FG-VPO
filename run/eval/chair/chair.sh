#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$PWD:$PYTHONPATH"

MODEL_PATH="./output/llava_fgvpo/llava_fgvpo_merged"
IMG_DIR="/path/to/your/project/evaluation/MSCOCO/val2014"
OUT_JSON="./output/eval/chair/llava_fgvpo_merged.json"
CACHE_PATH="/path/to/your/project/evaluation/chair/chair.pkl"
SAVE_PATH="./output/eval/chair/metrics_llava_fgvpo_merged.json"

python run/eval/chair/llava_captions_generation.py \
  --model_path $MODEL_PATH \
  --img_dir $IMG_DIR \
  --out_json $OUT_JSON \
  --max_new_tokens 128 \
  --limit 500

python run/eval/chair/chair.py \
  --cap_file $OUT_JSON \
  --image_id_key image_id \
  --caption_key caption \
  --coco_path /home/lijingran/dataset/MSCOCO/annotations \
  --cache $CACHE_PATH \
  --save_path $SAVE_PATH
