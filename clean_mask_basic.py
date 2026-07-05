"""基础 mask 清洗：只筛纯空/纯满的错误 mask，只记录不删除。

对每个类目的 ref_mask / tar_mask 计算阈值化后的前景占比 ratio：
  - ratio == 0       -> 纯空 mask（empty）
  - ratio >= FULL    -> 纯满 mask（full）
注意：部分类目的 mask 是带抗锯齿的软 mask（0~255 多灰阶），
所以用 >127 阈值二值化后再算占比，而非数 unique 值。

用法：
  python clean_mask_basic.py                       # 扫默认数据目录下全部类目
  python clean_mask_basic.py --root <data_dir>     # 指定数据目录
  python clean_mask_basic.py --categories clothes dog   # 只扫指定类目
  python clean_mask_basic.py --workers 16          # 并行进程数
  python clean_mask_basic.py --full 0.999          # 调整纯满阈值
  python clean_mask_basic.py --soft 0.001          # 调整软 mask 阈值

输出：
  - <root>/report_empty_full_mask.jsonl   每行一个坏样本记录
  - stdout 汇总每个类目的命中数
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

PAIR_DIRS = ("ref_image", "ref_mask", "tar_image", "tar_mask")
MASK_DIRS = ("ref_mask", "tar_mask")
DEFAULT_ROOT = "/inspire/hdd/global_user/250041225002/project/qwen-inpainting/a2edit/data"
DEFAULT_FULL = 0.999          # 前景占比 >= 此值视为纯满
BIN_THRESHOLD = 127           # >127 视为前景
DEFAULT_SOFT = 0.001          # 非{0,255}像素占比 >= 此值视为软 mask（抗锯齿等）


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="只读筛选纯空/纯满 mask")
    p.add_argument("--root", default=DEFAULT_ROOT, help="数据根目录")
    p.add_argument("--categories", nargs="*", help="只扫这些类目；默认扫全部")
    p.add_argument("--workers", type=int, default=8, help="并行进程数")
    p.add_argument("--full", type=float, default=DEFAULT_FULL, help="纯满阈值")
    p.add_argument("--soft", type=float, default=DEFAULT_SOFT, help="软 mask 阈值：非{0,255}像素占比")
    p.add_argument("--out", default=None, help="报告输出路径；默认 <root>/report_empty_full_mask.jsonl")
    return p.parse_args()


def list_categories(root: Path, only: list[str] | None) -> list[str]:
    cats = [d.name for d in root.iterdir()
            if d.is_dir() and all((d / s).is_dir() for s in PAIR_DIRS)]
    if only:
        cats = [c for c in cats if c in set(only)]
    return sorted(cats)


def check_one_mask(path: Path, full_thresh: float, soft_thresh: float) -> tuple[float, str | None]:
    """返回 (前景占比, 问题类型 or None)。读图失败记为 read_error。

    问题类型：empty（纯空）/ full（纯满）/ soft（非{0,255}像素过多，抗锯齿等）。
    判定顺序：read_error -> empty -> full -> soft，避免全 0/全 255 被误判为 soft。
    """
    try:
        arr = np.asarray(Image.open(path).convert("L"))
    except Exception:
        return -1.0, "read_error"
    if arr.size == 0:
        return -1.0, "read_error"
    fg = float((arr > BIN_THRESHOLD).mean())
    if fg == 0.0:
        return 0.0, "empty"
    if fg >= full_thresh:
        return fg, "full"
    # 非{0,255}像素占比，检测软 mask
    non_binary = float(((arr != 0) & (arr != 255)).mean())
    if non_binary >= soft_thresh:
        return fg, "soft"
    return fg, None


def check_sample(args: tuple[str, str, str, float, float]) -> list[dict]:
    cat, name, root, full_thresh, soft_thresh = args
    recs: list[dict] = []
    for mdir in MASK_DIRS:
        path = Path(root) / cat / mdir / name
        ratio, issue = check_one_mask(path, full_thresh, soft_thresh)
        if issue is not None:
            recs.append({
                "category": cat,
                "name": name,
                "mask_dir": mdir,
                "issue": issue,
                "fg_ratio": round(ratio, 6),
                "path": str(path),
            })
    return recs


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    cats = list_categories(root, args.categories)
    if not cats:
        raise SystemExit(f"未找到符合条件的类目目录于 {root}")

    tasks: list[tuple[str, str, str, float, float]] = []
    for cat in cats:
        ref_dir = root / cat / "ref_mask"
        for name in sorted(os.listdir(ref_dir)):
            if name.startswith("."):
                continue
            tasks.append((cat, name, str(root), args.full, args.soft))

    out_path = Path(args.out) if args.out else root / "report_empty_full_mask.jsonl"
    per_cat: dict[str, dict[str, int]] = {c: {"empty": 0, "full": 0, "soft": 0, "read_error": 0} for c in cats}
    total_hits = 0

    with out_path.open("w", encoding="utf-8") as fout, ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(check_sample, t): t for t in tasks}
        for fut in as_completed(futures):
            for rec in fut.result():
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                per_cat[rec["category"]][rec["issue"]] += 1
                total_hits += 1

    print(f"\n扫描完成：{len(tasks)} 个样本，命中 {total_hits} 条")
    print(f"报告写入：{out_path}\n")
    print(f"{'category':<22}{'empty':>8}{'full':>8}{'soft':>8}{'read_err':>10}")
    print("-" * 56)
    for cat in cats:
        c = per_cat[cat]
        print(f"{cat:<22}{c['empty']:>8}{c['full']:>8}{c['soft']:>8}{c['read_error']:>10}")


if __name__ == "__main__":
    main()
