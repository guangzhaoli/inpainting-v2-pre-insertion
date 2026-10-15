"""Persistently track two Inspire teachers, publish status, and notify via lark-cli.

Reads the launch manifest on every poll, so a reviewed replacement job can be
tracked without restarting the watcher. No jobs are created by this program.
"""

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import time

from tensorboard.compat.proto.event_pb2 import Event


def atomic_json(path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


class EventTail:
    def __init__(self):
        self.offsets = {}
        self.values = collections.defaultdict(lambda: collections.deque(maxlen=200))

    def read(self, directory):
        for path in sorted(directory.glob("events.out.tfevents.*")):
            offset = self.offsets.get(str(path), 0)
            with path.open("rb") as stream:
                stream.seek(offset)
                while True:
                    start = stream.tell()
                    header = stream.read(12)
                    if len(header) != 12:
                        break
                    length = struct.unpack("<Q", header[:8])[0]
                    if length > 100_000_000:
                        raise ValueError(f"Invalid event length at {path}:{start}")
                    payload = stream.read(length)
                    crc = stream.read(4)
                    if len(payload) != length or len(crc) != 4:
                        break
                    event = Event.FromString(payload)
                    for value in event.summary.value:
                        if value.HasField("simple_value"):
                            self.values[value.tag].append(
                                (event.step, value.simple_value, event.wall_time)
                            )
                    self.offsets[str(path)] = stream.tell()
        result = {}
        for tag, values in self.values.items():
            last = values[-1]
            recent = [x[1] for x in list(values)[-50:]]
            result[tag] = {
                "step": last[0],
                "value": last[1],
                "mean_last_50": sum(recent) / len(recent),
                "wall_time": last[2],
            }
        return result


def cli_status(manifest):
    names = [j["name"] for j in manifest["jobs"].values()]
    command = [
        "inspire",
        "--account",
        "main",
        "--json",
        "job",
        "status",
        *names,
        "--workspace",
        manifest["workspace"],
    ]
    result = subprocess.run(command, text=True, capture_output=True, timeout=55)
    if result.returncode:
        raise RuntimeError("Inspire status unavailable: " + result.stderr[-400:])
    data = json.loads(result.stdout)
    if not data.get("success"):
        raise RuntimeError("Inspire status returned unsuccessful response")
    return {j["name"]: j for j in data["data"]["items"]}


def notify(manifest, key, text, state, state_path):
    if key in state.get("notified", []):
        return
    env = os.environ | {
        "LARKSUITE_CLI_NO_UPDATE_NOTIFIER": "1",
        "LARKSUITE_CLI_NO_SKILLS_NOTIFIER": "1",
    }
    response = subprocess.run(
        [
            "lark-cli",
            "im",
            "+messages-send",
            "--as",
            "bot",
            "--user-id",
            manifest["notification_user"],
            "--text",
            text,
            "--idempotency-key",
            "ifv2-" + hashlib.sha256(key.encode()).hexdigest()[:40],
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=50,
    )
    if response.returncode != 0:
        print("notification failed; retained for retry", key, flush=True)
        return
    payload = json.loads(response.stdout)
    if payload.get("ok"):
        state.setdefault("notified", []).append(key)
        atomic_json(state_path, state)


def summarize_validation(directory):
    results = {}
    for path in sorted(
        directory.parent.glob("v*/validation/step-*/metrics.json"),
        key=lambda p: p.stat().st_mtime,
    ):
        # metrics.json precedes metadata/grid publication. Wait for both.
        if (
            not (path.parent / "metadata.json").exists()
            or not (path.parent / "all_rows.png").exists()
        ):
            continue
        data = json.loads(path.read_text())
        data["directory"] = str(path.parent)
        results[int(data["step"])] = data
    return [results[step] for step in sorted(results)]


def historical_comparison(root, domain, validation):
    baseline_path = root / "historical_baselines.json"
    if not baseline_path.exists():
        return None
    baseline = json.loads(baseline_path.read_text())["domains"].get(domain, {})
    metadata = json.loads((Path(validation["directory"]) / "metadata.json").read_text())
    pairs = []
    for sample in metadata["samples"]:
        key = f"{sample['category']}/{sample['stem']}"
        if key in baseline and sample.get("metrics"):
            pairs.append((sample["metrics"], baseline[key]["metrics"]))
    if not pairs:
        return None
    result = {"matched_cases": len(pairs), "metrics": {}}
    for key in ["reference_dino_cosine", "raw_background_mae"]:
        values = [(a[key], b[key]) for a, b in pairs if key in a and key in b]
        if values:
            current = sum(a for a, b in values) / len(values)
            old = sum(b for a, b in values) / len(values)
            result["metrics"][key] = {
                "current": current,
                "historical_v1": old,
                "delta": current - old,
            }
    result["caveat"] = (
        "Same cases, but different seeds/CFG/output modes; not a controlled architecture-only ablation."
    )
    return result


def process(manifest, state, tails, state_path):
    root = Path(manifest["run_root"])
    viewer_url = (
        (root / "public_url.txt").read_text().strip()
        if (root / "public_url.txt").exists()
        else ""
    )
    try:
        platform = cli_status(manifest)
        state["platform"] = platform
        state["platform_checked_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    except Exception as error:
        print(str(error), flush=True)
        platform = state.get("platform", {})
    snapshot = {
        "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "public_url": viewer_url,
        "workspace": manifest["workspace"],
        "gpu_per_job": 8,
        "jobs": {},
        "platform_checked_utc": state.get("platform_checked_utc"),
    }
    terminal_count = 0
    for domain, job in manifest["jobs"].items():
        name = job["name"]
        live = platform.get(name, {"name": name, "status": "UNKNOWN"})
        versions = sorted(
            Path(job["expected_run_parent"]).glob("v*"), key=lambda p: p.stat().st_mtime
        )
        directory = versions[-1] if versions else None
        scalars = (
            tails.setdefault(str(directory), EventTail()).read(directory)
            if directory
            else {}
        )
        step = (
            int(
                scalars.get("train/global_step", scalars.get("loss", {"step": -1}))[
                    "step"
                ]
            )
            + 1
        )
        if not scalars:
            step = int(job.get("resume_step", 0))
        records = summarize_validation(directory) if directory else []
        seconds = scalars.get("perf/optimizer_step_time_sec", {}).get("mean_last_50")
        estimated_steps = job["estimated_steps"]
        status = live["status"]
        exit_path = root / f"{domain}.exit.json"
        exit_data = json.loads(exit_path.read_text()) if exit_path.exists() else None
        latest_checkpoint = None
        if directory:
            candidates = []
            for p in directory.glob("checkpoint-*/state.json"):
                try:
                    st = json.loads(p.read_text())
                    candidates.append((st["global_step"], p.parent, st))
                except (OSError, ValueError):
                    continue
            if candidates:
                _, p, st = max(candidates, key=lambda v: v[0])
                latest_checkpoint = {"path": str(p), "state": st}
        if status in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            terminal_count += 1
        entry = {
            "name": name,
            "domain": domain,
            "status": status,
            "phase": "training" if scalars else "loading_or_initial_validation",
            "resource": live.get("resource"),
            "step": step,
            "estimated_steps": estimated_steps,
            "seconds_per_step": seconds,
            "eta_hours_excluding_future_validation": max(0, estimated_steps - step)
            * seconds
            / 3600
            if seconds
            else None,
            "training_directory": str(directory) if directory else None,
            "scalars": scalars,
            "validation": records,
            "checkpoint": latest_checkpoint,
            "exit": exit_data,
        }
        snapshot["jobs"][domain] = entry
        label = "首饰" if domain == "accessory_jewelry" else "宠物"
        base = f"InsertFuse v2 {label}\n任务：{name}\n状态：{status}；训练进度 {step}/{estimated_steps} 步"
        if (
            scalars
            and seconds
            and step >= max(10, int(job.get("resume_step", 0)) + 1)
            and status == "RUNNING"
        ):
            notify(
                manifest,
                name + ":healthy",
                base
                + f"\n已进入有效训练；近 50 步平均 {seconds:.2f} 秒/步。\n"
                + viewer_url
                if seconds
                else base,
                state,
                state_path,
            )
        if records:
            recent = records[-1]
            metrics = recent["metrics"]
            first = records[0]["metrics"]
            delta = {k: metrics[k] - first[k] for k in metrics if k in first}
            entry["validation_change_from_first"] = delta
            report = {
                "domain": domain,
                "step": recent["step"],
                "metrics": metrics,
                "change_from_first": delta,
                "interpretation": "DINO cosine is identity evidence; raw background MAE measures change, not naturalness. Inspect images for geometry, identity detail, seams and contact shadows.",
                "quality_improvement_confirmed": False,
                "historical_v1_comparison": historical_comparison(root, domain, recent),
            }
            entry["historical_v1_comparison"] = report["historical_v1_comparison"]
            atomic_json(root / f"{domain}.analysis.json", report)
            if len(records) == 1 or recent["step"] % 1000 == 0:
                detail = "；".join(f"{k}={v:.4f}" for k, v in metrics.items())
                baseline_detail = ""
                comparison = report["historical_v1_comparison"]
                if comparison:
                    identity = comparison["metrics"].get("reference_dino_cosine")
                    if identity:
                        baseline_detail = f"\n同样本历史 v1 DINO={identity['historical_v1']:.4f}；当前差值 {identity['delta']:+.4f}（设置不同，仅作参考）。"
                notify(
                    manifest,
                    name + ":validation:" + str(recent["step"]),
                    base
                    + f"\n验证 step {recent['step']}：{detail}"
                    + baseline_detail
                    + f"\n查看验证图：{viewer_url}\n指标不能单独证明自然度改善。",
                    state,
                    state_path,
                )
        if status in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            detail = ""
            if status == "SUCCEEDED" and latest_checkpoint:
                detail = "\n最终 checkpoint：" + latest_checkpoint["path"]
                completed_step = int(latest_checkpoint["state"].get("global_step", 0))
                if completed_step < estimated_steps:
                    detail = (
                        f"\n进程正常退出，但只保存到 {completed_step}/{estimated_steps} 步，完整训练尚未完成。"
                        + "\n可恢复 checkpoint："
                        + latest_checkpoint["path"]
                    )
            elif status == "SUCCEEDED":
                detail = "\n平台报告成功，但未找到 checkpoint；需核验产物，不能视为训练完成。"
            if status == "FAILED":
                log = Path(job["log"])
                if log.exists():
                    with log.open("rb") as stream:
                        stream.seek(max(0, log.stat().st_size - 20000))
                        tail = stream.read().decode(errors="replace")
                    errors = [
                        line
                        for line in tail.splitlines()
                        if any(
                            t in line
                            for t in ["Error:", "CUDA out of memory", "Exception:"]
                        )
                    ]
                    detail = "\n错误：" + (
                        "\n".join(errors[-2:])[:900] if errors else "请查看训练日志。"
                    )
            notify(
                manifest,
                name + ":terminal:" + status,
                base + detail + "\n" + viewer_url,
                state,
                state_path,
            )
        latest_wall = scalars.get("loss", {}).get("wall_time")
        if status == "RUNNING" and latest_wall and time.time() - latest_wall > 1800:
            notify(
                manifest,
                name + ":stalled:" + str(step),
                base
                + "\n超过 30 分钟未写入新训练步，请检查验证或训练是否阻塞。\n"
                + viewer_url,
                state,
                state_path,
            )
    atomic_json(root / "status.json", snapshot)
    atomic_json(state_path, state)
    print(
        snapshot["updated_utc"],
        [(j["name"], j["status"], j["step"]) for j in snapshot["jobs"].values()],
        flush=True,
    )
    return terminal_count == len(manifest["jobs"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    state_path = args.manifest.parent / "watcher_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    tails = {}
    while True:
        manifest = json.loads(args.manifest.read_text())
        try:
            process(manifest, state, tails, state_path)
        except Exception as error:
            print(type(error).__name__, str(error), flush=True)
        if args.once:
            break
        # Keep watching replacements after an initial failed startup. No user
        # notifications are sent for unchanged/non-actionable states.
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
