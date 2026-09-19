import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer

# ========== 路径配置 ==========
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "Qwen3-0.6B"
DATA_DIR = PROJECT_ROOT / "data" / "anime-prompts"
TRAIN_FILE = DATA_DIR / "train_clean.jsonl"
EVAL_FILE = DATA_DIR / "eval_clean.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "qwen3-06b-animeprompt-qlora"
FINAL_ADAPTER_DIR = OUTPUT_DIR / "final_adapter"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        default=None,
        help="checkpoint 目录，例如 outputs/.../checkpoint-20",
    )
    return parser.parse_args()


def main():
    cli_args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；先完成 PyTorch GPU 验证")

    set_seed(42)
    use_bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print("GPU:", torch.cuda.get_device_name(0))
    print("compute dtype:", compute_dtype)
    print("model path:", MODEL_PATH)
    print("train file:", TRAIN_FILE)

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(TRAIN_FILE),
            "validation": str(EVAL_FILE),
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        model_init_kwargs={
            "dtype": compute_dtype,
            "quantization_config":quantization_config
        },
        max_length=512,
        completion_only_loss=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        learning_rate=1e-4,
        warmup_steps=0.05,
        lr_scheduler_type="cosine",
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_cache=False,
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=str(MODEL_PATH),
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.model.print_trainable_parameters()
    result = trainer.train(resume_from_checkpoint=cli_args.resume)
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()
    trainer.save_model(str(FINAL_ADAPTER_DIR))
    tokenizer.save_pretrained(str(FINAL_ADAPTER_DIR))
    print("adapter saved to:", FINAL_ADAPTER_DIR)


if __name__ == "__main__":
    main()
