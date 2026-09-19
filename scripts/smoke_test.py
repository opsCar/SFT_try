import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 可选：若仍需从镜像下载，保留这一行
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 改为绝对路径，避免被当成 Hugging Face 仓库 ID
MODEL_PATH = "/home/peter/projects/animeprompt-align/models/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    dtype=dtype,
    device_map="auto",
)

messages = [
    {
        "role": "system",
        "content": (
            "You convert anime image requests into strict JSON. "
            "Return only JSON with positive_prompt, negative_prompt and control."
        ),
    },
    {
        "role": "user",
        "content": "画一个银色长发、红色眼睛的少女，在雨夜街头侧身回头。",
    },
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=False,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
    )

generated = outputs[0, inputs["input_ids"].shape[1] :]
text = tokenizer.decode(generated, skip_special_tokens=True)
print(text)

try:
    json.loads(text)
    print("\nJSON valid: yes")
except json.JSONDecodeError:
    print("\nJSON valid: no（这是训练前基线，不一定能输出严格 JSON）")
