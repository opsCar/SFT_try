import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset
from openai import OpenAI

# ========== 配置 ==========
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "你的-API-Key")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

DATA_DIR = Path("/home/peter/projects/animeprompt-align/data/anime-prompts/")
CSV_FILE = DATA_DIR / "anime_prompts.csv"
CACHE_FILE = DATA_DIR / "user_requests_cache.jsonl"
OUTPUT_TRAIN = DATA_DIR / "train.jsonl"
OUTPUT_EVAL = DATA_DIR / "eval.jsonl"

SOURCE_COLUMN = "safebooru_clean"
EVAL_RATIO = 0.02

SYSTEM_MSG = (
    "You convert anime image requests into strict JSON. "
    "Return only JSON with positive_prompt, negative_prompt and control."
)

NEGATIVE_PROMPT = ["bad_anatomy", "extra_fingers", "text", "watermark"]

# 元数据标签黑名单
META_TAGS = {
    "highres", "absurdres", "lowres", "animated",
    "commentary", "commentary request", "english commentary", "chinese commentary",
    "translation request", "source request", "artist request", "symbol-only commentary",
    "twitter username", "instagram username", "pixiv username",
    "bad id", "bad pixiv id", "bad twitter id", "bad link",
    "numbered", "duplicate", "pixel-perfect duplicate", "expressions",
    "skeb commission", "commission", "artist name", "original",
}

# 高频标签，随机丢弃以降低默认出现概率
HIGH_FREQ_TAGS = {"1girl", "solo"}
DROP_PROB = 0.3

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并发处理数量（默认 1，即串行）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="生成数据上限（默认 None，即全量）",
    )
    return parser.parse_args()


# ========== 第一步：加载数据集 ==========
def load_source(limit=None):
    ds = load_dataset("csv", data_files=str(CSV_FILE), split="train")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


# ========== 第二步：调用 DeepSeek 反向生成中文请求 ==========
def load_cache():
    cache = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                cache[obj["tags"]] = obj["user_request"]
    return cache


def save_cache_entry(tags, user_request):
    """线程安全地追加缓存（用锁保护）。"""
    with cache_lock:
        with open(CACHE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"tags": tags, "user_request": user_request}, ensure_ascii=False) + "\n")


import threading
cache_lock = threading.Lock()


def reverse_generate(tags):
    """把英文标签反推成中文用户请求，返回字符串或 None。"""
    user_prompt = (
        f"以下是一组动漫图片的英文提示词标签：\n{tags}\n\n"
        "请反推出一个自然的中文用户请求，描述用户想要画什么样的图。"
        "要求：\n"
        "1. 只输出 JSON，格式为 {\"user_request\": \"...\"}；\n"
        "2. 中文请求要自然、口语化，像真实用户会说的话；\n"
        "3. 不要包含露骨、暴力或其他不适合训练数据的描述；\n"
        "4. 如果标签内容明显不适合作为训练数据，返回 {\"user_request\": null}。"
    )
    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": "你是一个动漫图片提示词专家。"},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=256,
            temperature=0.7,
        )
        result = json.loads(response.choices[0].message.content)
        return result.get("user_request")
    except Exception as e:
        print(f"[API 错误] {e}")
        return None


def process_one(tags, cache):
    """处理单条 tags：命中缓存则跳过，否则调 API 并写缓存。"""
    if tags in cache:
        return tags, cache[tags]
    user_request = reverse_generate(tags)
    cache[tags] = user_request
    save_cache_entry(tags, user_request)
    return tags, user_request


# ========== 第三步：组装 TRL 格式 ==========
def build_assistant_json(tags):
    positive = [
        t.strip() for t in tags.split(",")
        if t.strip() and t.strip() not in META_TAGS
    ]
    # 对高频标签做随机丢弃
    positive = [
        t for t in positive
        if t not in HIGH_FREQ_TAGS or random.random() > DROP_PROB
    ]
    # 去重，保持顺序
    seen = set()
    deduped = []
    for t in positive:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return json.dumps(
        {
            "positive_prompt": deduped,
            "negative_prompt": NEGATIVE_PROMPT,
            "control": {"recommended": False, "type": None},
        },
        ensure_ascii=False,
    )


def to_trl_sample(tags, user_request):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_request},
        ],
        "completion": [
            {"role": "assistant", "content": build_assistant_json(tags)}
        ],
    }


# ========== 主流程 ==========
def main():
    cli_args = parse_args()
    workers = max(1, cli_args.workers)
    limit = cli_args.limit

    ds = load_source(limit=limit)
    cache = load_cache()
    print(f"总样本数: {len(ds)}，已有缓存: {len(cache)}，并发数: {workers}，上限: {limit or '全量'}")

    tags_list = ds[SOURCE_COLUMN]

    # 过滤掉已在缓存里的，只处理需要新生成的
    pending = [tags for tags in tags_list if tags not in cache]
    print(f"需要新生成: {len(pending)} 条")

    if workers == 1:
        # 串行模式
        for i, tags in enumerate(pending):
            process_one(tags, cache)
            if (i + 1) % 20 == 0:
                print(f"已处理 {i + 1}/{len(pending)}")
            time.sleep(0.5)
    else:
        # 并发模式
        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_one, tags, cache): tags for tags in pending}
            for future in as_completed(futures):
                completed += 1
                if completed % 20 == 0:
                    print(f"已处理 {completed}/{len(pending)}")

    # 组装并写出
    samples = []
    for tags in tags_list:
        user_request = cache.get(tags)
        if not user_request:
            continue
        samples.append(to_trl_sample(tags, user_request))

    split_idx = int(len(samples) * (1 - EVAL_RATIO))
    train_samples = samples[:split_idx]
    eval_samples = samples[split_idx:]

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
