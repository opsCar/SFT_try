import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "Qwen3-0.6B"
ADAPTER_PATH = PROJECT_ROOT / "outputs" / "qwen3-06b-animeprompt-qlora" / "final_adapter"

SYSTEM_MSG = (
    "You convert anime image requests into strict JSON. "
    "Return only JSON with positive_prompt, negative_prompt and control."
)

TEST_REQUESTS = [
    "画一个银色长发、红色眼睛的少女，在雨夜街头侧身回头。",
    "帮我画一张《原神》胡桃的图，棕色双马尾，戴黑色高礼帽，闭眼大笑，双手比爪爪。",
    "画一张《明日方舟》阿米娅的图，从背后看，兔耳朵，黑色外套，背景是海边灯塔。",
    "画一个原创动漫少女，蓝发蓝眼，穿水手服，站在樱花树下微笑。",
]


def load_model():
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        quantization_config=bnb_config,
        device_map="auto",
        dtype=compute_dtype,
    )
    model = PeftModel.from_pretrained(base, str(ADAPTER_PATH))
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, user_request):
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": user_request},
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

    generated = outputs[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def main():
    print("加载模型和 adapter...")
    model, tokenizer = load_model()

    for req in TEST_REQUESTS:
        print(f"\n{'='*60}")
        print(f"请求: {req}")
        text = generate(model, tokenizer, req)
        print(f"输出: {text}")

        try:
            parsed = json.loads(text)
            print(f"JSON 合法: yes | positive_prompt 数量: {len(parsed.get('positive_prompt', []))}")
        except json.JSONDecodeError:
            print("JSON 合法: no")


if __name__ == "__main__":
    main()
