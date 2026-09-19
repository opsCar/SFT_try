---
dataset_info:
  features:
  - name: source
    dtype: string
  - name: messages
    list:
    - name: content
      dtype: string
    - name: role
      dtype: string
  - name: num_turns
    dtype: int64
  splits:
  - name: train
    num_bytes: 71908734
    num_examples: 15806
  - name: test
    num_bytes: 929564
    num_examples: 200
  download_size: 37644679
  dataset_size: 72838298
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
---
