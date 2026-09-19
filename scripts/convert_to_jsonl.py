import json
from pathlib import Path

INPUT_FILE = Path("/home/peter/projects/animeprompt-align/data/anime-prompts/jev_audit_filtered.jsonl")
OUTPUT_TRAIN = INPUT_FILE.parent / "train_clean.jsonl"
OUTPUT_EVAL = INPUT_FILE.parent / "eval_clean.jsonl"
EVAL_RATIO = 0.02

SYSTEM_MSG = (
    "You convert anime image requests into strict JSON. "
    "Return only JSON with positive_prompt, negative_prompt and control."
)

NEGATIVE_PROMPT = ["bad_anatomy", "extra_fingers", "text", "watermark"]

# 元数据标签黑名单：图站标注、来源、状态类，不是画面内容描述
META_TAGS = {
    # 分辨率/质量
    "highres", "absurdres", "lowres", "animated",
    # 图站标注
    "commentary", "commentary request", "english commentary", "chinese commentary",
    "translation request", "source request", "artist request", "symbol-only commentary",
    "twitter username", "instagram username", "pixiv username",
    "bad id", "bad pixiv id", "bad twitter id", "bad link",
    "numbered", "duplicate", "pixel-perfect duplicate", "expressions",
    # 来源/委托
    "skeb commission", "commission", "artist name", "original",
}

# 单条数据最多保留的标签数，防止模型学到“输出超长数组”的模式
MAX_TAGS = 40


def build_assistant_json(tags):
    # 1) 切分 + 去空白
    positive = [t.strip() for t in tags.split(",") if t.strip()]

    # 2) 过滤元数据标签
    positive = [t for t in positive if t not in META_TAGS]

    # 3) 去重，保持原有顺序
    seen = set()
    deduped = []
    for t in positive:
        if t not in seen:
            seen.add(t)
            deduped.append(t)

    # 4) 截断过长的数组
    deduped = deduped[:MAX_TAGS]

    return json.dumps(
        {
            "positive_prompt": deduped,
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

    # 过滤后 positive_prompt 为空的样本直接丢弃
    trl_samples = []
    skipped = 0
    for obj in samples:
        sample = to_trl_sample(obj)
        completion = json.loads(sample["completion"][0]["content"])
        if not completion["positive_prompt"]:
            skipped += 1
            continue
        trl_samples.append(sample)

    split_idx = int(len(trl_samples) * (1 - EVAL_RATIO))
    train_samples = trl_samples[:split_idx]
    eval_samples = trl_samples[split_idx:]

    with open(OUTPUT_TRAIN, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(OUTPUT_EVAL, "w", encoding="utf-8") as f:
        for s in eval_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"原始样本: {len(samples)}")
    print(f"过滤后丢弃: {skipped} 条（positive_prompt 为空）")
    print(f"训练集: {len(train_samples)} 条 -> {OUTPUT_TRAIN}")
    print(f"评估集: {len(eval_samples)} 条 -> {OUTPUT_EVAL}")


if __name__ == "__main__":
    main()
