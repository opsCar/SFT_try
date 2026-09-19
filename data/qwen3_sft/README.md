---
dataset_info:
  config_name: sft_Qwen_Qwen3-0.6B-Base
  features:
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: system
    dtype: string
  splits:
  - name: train
    num_bytes: 17937374
    num_examples: 7500
  download_size: 5093980
  dataset_size: 17937374
configs:
- config_name: sft_Qwen_Qwen3-0.6B-Base
  data_files:
  - split: train
    path: sft_Qwen_Qwen3-0.6B-Base/train-*
---
