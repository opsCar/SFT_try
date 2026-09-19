import json
from pathlib import Path

# ========== 配置 ==========
INPUT_FILE = Path("/home/peter/projects/animeprompt-align/data/anime-prompts/jev_audit_results.jsonl")
OUTPUT_FILE = INPUT_FILE.parent / "jev_audit_filtered.jsonl"
THRESHOLD = 0.90


def main():
    kept = []
    dropped = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj["jev_match"] >= THRESHOLD:
                kept.append(obj)
            else:
                dropped.append(obj)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for obj in kept:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    total = len(kept) + len(dropped)
    print(f"总计:     {total}")
    print(f"保留:     {len(kept)}  ({len(kept) / total:.1%})")
    print(f"丢弃:     {len(dropped)}  ({len(dropped) / total:.1%})")
    print(f"输出文件: {OUTPUT_FILE}")

    # 打印被丢弃的样本，方便复查
    if dropped:
        print(f"\n=== 被丢弃的 {len(dropped)} 条 ===")
        for obj in sorted(dropped, key=lambda x: x["jev_match"]):
            print(f"\nScore: {obj['jev_match']:.4f}")
            print(f"Tags:  {obj['tags'][:120]}...")
            print(f"请求:  {obj['user_request'][:120]}...")


if __name__ == "__main__":
    main()
