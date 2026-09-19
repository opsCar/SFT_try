import json
from pathlib import Path

INPUT_FILE = Path("/home/peter/projects/animeprompt-align/data/anime-prompts/jev_audit_filtered.jsonl")
OUTPUT_TRAIN = INPUT_FILE.parent / "train.jsonl"
OUTPUT_EVAL = INPUT_FILE.parent / "eval.jsonl"
EVAL_RATIO = 0.02

SYSTEM_MSG = (
    "You convert anime image requests into strict JSON. "
    "Return only JSON with positive_prompt, negative_prompt and control."
)

NEGATIVE_PROMPT = ["bad_anatomy", "extra_fingers", "text", "watermark"]


def build_assistant_json(tags):
    positive = [t.strip() for t in tags.split(",") if t.strip()]
    return json.dumps(
        {
            "positive_prompt": positive,
            "negative_prompt": NEGATIVE_PROMPT,
            "control": {"recommended": False, "type": None},
        },
        ensure_ascii=False,
    )


def to_trl_sample(obj):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": obj["user_request"]},
        ],
        "completion": [
            {"role": "assistant", "content": build_assistant_json(obj["tags"])}
        ],
    }


def main():
    samples = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))

    trl_samples = [to_trl_sample(obj) for obj in samples]

    split_idx = int(len(trl_samples) * (1 - EVAL_RATIO))
    train_samples = trl_samples[:split_idx]
    eval_samples = trl_samples[split_idx:]

    with open(OUTPUT_TRAIN, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(OUTPUT_EVAL, "w", encoding="utf-8") as f:
        for s in eval_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"训练集: {len(train_samples)} 条 -> {OUTPUT_TRAIN}")
    print(f"评估集: {len(eval_samples)} 条 -> {OUTPUT_EVAL}")


if __name__ == "__main__":
    main()
