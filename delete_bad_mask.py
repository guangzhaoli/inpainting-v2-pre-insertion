"""删除 report_bad_mask_definite.jsonl 里的所有 case 的完整四件套。

按 (category, name) 去重得到样本组，每组删除 ref_image/ref_mask/tar_image/tar_mask
四个文件（若存在）；并从该类目的 labels.json / label.json 中移除对应 key。

用法：
  python delete_bad_mask.py --report <path> --dry-run   # 只打印计划
  python delete_bad_mask.py --report <path>             # 实际删除
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PAIR_DIRS = ("ref_image", "ref_mask", "tar_image", "tar_mask")
LABEL_NAMES = ("labels.json", "label.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--report", required=True)
    p.add_argument("--root", default="/inspire/hdd/global_user/250041225002/project/qwen-inpainting/a2edit/data")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)

    # 读取报告，按 (category, name) 去重
    groups: dict[tuple[str, str], set[str]] = {}
    with open(args.report) as f:
        for line in f:
            r = json.loads(line)
            key = (r["category"], r["name"])
            groups.setdefault(key, set()).add(r.get("mask_dir", ""))

    # 按类目统计
    from collections import Counter
    per_cat = Counter(cat for cat, _ in groups)
    print(f"待删除样本组数：{len(groups)}")
    for cat, n in sorted(per_cat.items()):
        print(f"  {cat:<22} {n}")

    total_files = 0
    deleted_groups = 0
    label_updates: list[Path] = []

    for (cat, name), _bad_dirs in sorted(groups.items()):
        cdir = root / cat
        # 删除四件套
        any_exists = False
        for s in PAIR_DIRS:
            p = cdir / s / name
            if p.exists():
                print(f"{'Would delete' if args.dry_run else 'Delete'}: {p}")
                if not args.dry_run:
                    p.unlink()
                total_files += 1
                any_exists = True
        if any_exists:
            deleted_groups += 1

        # 同步 labels.json / label.json
        for ln in LABEL_NAMES:
            lp = cdir / ln
            if not lp.is_file():
                continue
            with lp.open() as f:
                data = json.load(f)
            if not isinstance(data, dict) or name not in data:
                continue
            print(f"{'Would remove key' if args.dry_run else 'Remove key'} {name} from {lp}")
            if not args.dry_run:
                del data[name]
                with lp.open("w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                    f.write("\n")
                if lp not in label_updates:
                    label_updates.append(lp)

    print(f"\n{'[DRY-RUN] ' if args.dry_run else ''}删除文件数：{total_files}，样本组数：{deleted_groups}，更新标签文件：{len(label_updates)}")
    if args.dry_run:
        print("（dry-run，未实际删除；去掉 --dry-run 执行）")


if __name__ == "__main__":
    main()
