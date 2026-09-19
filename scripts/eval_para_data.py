import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from typesafe_sdk import Noul, TypeSafeClient

# ========== 配置 ==========
DATA_FILE = Path("/home/peter/projects/animeprompt-align/data/anime-prompts/user_requests_cache.jsonl")
OUTPUT_FILE = DATA_FILE.parent / "jev_audit_results.jsonl"
MAX_WORKERS = 100  # 最多 100 个请求同时进行


# ========== 读取数据 ==========
def load_samples(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("user_request"):
                obj["_line"] = i
                samples.append(obj)
    return samples


# ========== 单条评估 ==========
def evaluate_one(client, sample):
    """返回 (sample, score, error)，score 为 None 表示失败。"""
    try:
        response = client.system_one(
            state={
                "tags": sample["tags"],
                "user_request": sample["user_request"],
            },
            questions={
                "is_match": Noul(
                    instructions=(
                        "Does the Chinese user_request accurately describe the same "
                        "visual content as the English tags?"
                    ),
                )
            },
        )
        score = response.answers["is_match"].noul
        return sample, round(score, 4), None
    except Exception as e:
        return sample, None, str(e)


# ========== 主流程 ==========
def main():
    samples = load_samples(DATA_FILE)
    print(f"加载了 {len(samples)} 条有效样本，并发数: {MAX_WORKERS}")

    results = []
    errors = 0
    completed = 0

    # 每个线程用独立的 client，避免共享 httpx client 的线程安全问题
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for sample in samples:
            client = TypeSafeClient()  # 每个任务一个 client
            futures[executor.submit(evaluate_one, client, sample)] = sample

        for future in as_completed(futures):
            sample, score, error = future.result()
            completed += 1
            if score is not None:
                results.append({
                    "tags": sample["tags"],
                    "user_request": sample["user_request"],
                    "jev_match": score,
                })
            else:
                errors += 1
                if errors <= 10:  # 只打印前 10 个错误，避免刷屏
                    print(f"[错误] {error[:120]}")
            if completed % 50 == 0:
                print(f"已完成 {completed}/{len(samples)}，成功 {len(results)}，失败 {errors}")

    # 写出结果
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n完成。成功 {len(results)} 条，失败 {errors} 条 -> {OUTPUT_FILE}")

    if not results:
        print("没有成功的结果，跳过统计。")
        return

    # 统计摘要
    scores = [r["jev_match"] for r in results]
    scores_sorted = sorted(scores)
    n = len(scores_sorted)
    print(f"\n=== Jev 匹配度统计 ===")
    print(f"样本数:   {n}")
    print(f"均值:     {sum(scores) / n:.4f}")
    print(f"中位数:   {scores_sorted[n // 2]:.4f}")
    print(f"最小值:   {scores_sorted[0]:.4f}")
    print(f"最大值:   {scores_sorted[-1]:.4f}")
    print(f"P10:      {scores_sorted[int(n * 0.10)]:.4f}")
    print(f"P25:      {scores_sorted[int(n * 0.25)]:.4f}")
    print(f"P75:      {scores_sorted[int(n * 0.75)]:.4f}")
    print(f"P90:      {scores_sorted[int(n * 0.90)]:.4f}")

    worst = sorted(results, key=lambda x: x["jev_match"])[:5]
    print(f"\n=== 匹配度最低的 5 条 ===")
    for r in worst:
        print(f"\nScore: {r['jev_match']:.4f}")
        print(f"Tags:  {r['tags'][:120]}...")
        print(f"请求:  {r['user_request'][:120]}...")

    best = sorted(results, key=lambda x: x["jev_match"], reverse=True)[:5]
    print(f"\n=== 匹配度最高的 5 条 ===")
    for r in best:
        print(f"\nScore: {r['jev_match']:.4f}")
        print(f"Tags:  {r['tags'][:120]}...")
        print(f"请求:  {r['user_request'][:120]}...")


if __name__ == "__main__":
    main()
