"""Serve real training scalars and allowlisted validation images, without GPU work."""

import csv
import datetime as dt
import hashlib
import io
import json
import math
import statistics
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

from PIL import Image
from tensorboard.compat.proto.event_pb2 import Event


STATIC = Path(__file__).parent
NAMES = {"fashion_tryon": "服饰试穿", "general_object": "通用物体", "human": "人物 / 头发", "animal": "宠物", "accessory_jewelry": "首饰"}
COLORS = {"fashion_tryon": "#d8783e", "general_object": "#0c9488", "human": "#7564d9"}
METRICS = {
    "loss": ("总训练 loss", "loss"),
    "train/latent_mse": ("Latent MSE", "loss"),
    "train/region_fg_lpips": ("前景 LPIPS", "loss"),
    "train/region_ref_dino": ("参考身份 DINO loss", "loss"),
    "train/region_face_dino": ("人脸 DINO loss", "loss"),
    "train/region_detail_lpips": ("局部细节 LPIPS", "loss"),
    "train/region_boundary_l1": ("边界 L1", "loss"),
    "train/region_bg_l1": ("背景 L1", "loss"),
    "perf/optimizer_step_time_sec": ("每步耗时", "秒"),
    "optim/lr": ("学习率", "lr"),
    "optim/grad_norm": ("梯度范数", "norm"),
    "system/cuda_max_memory_allocated_gib": ("GPU 峰值已分配显存", "GiB"),
    "train/face_pair_valid_fraction": ("有效人脸配对占比", "fraction"),
}
ROLES = {"reference", "reference_mask_overlay", "source", "source_mask_overlay", "output", "output_restored_raw", "composite"}


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path):
    return json.loads(path.read_text())


def finite(value):
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


class ScalarReader:
    """Incremental TFRecord reader; duplicate steps keep the latest wall time."""

    def __init__(self):
        self.offsets = {}
        self.values = {tag: {} for tag in METRICS}

    def read(self, directories):
        files = sorted(p for d in directories for p in d.glob("events.out.tfevents.*"))
        if any(p.stat().st_size < self.offsets.get(str(p), 0) for p in files):
            self.offsets.clear()
            self.values = {tag: {} for tag in METRICS}
        for path in files:
            with path.open("rb") as stream:
                stream.seek(self.offsets.get(str(path), 0))
                while True:
                    header = stream.read(12)
                    if len(header) != 12:
                        break
                    length = struct.unpack("<Q", header[:8])[0]
                    if length > 64_000_000:
                        break
                    payload, crc = stream.read(length), stream.read(4)
                    if len(payload) != length or len(crc) != 4:
                        break
                    event = Event.FromString(payload)
                    for value in event.summary.value:
                        if value.tag in self.values and value.HasField("simple_value"):
                            # The trainer logs before incrementing global_step.
                            point = [int(event.step) + 1, finite(value.simple_value), event.wall_time]
                            previous = self.values[value.tag].get(point[0])
                            if previous is None or point[2] >= previous[2]:
                                self.values[value.tag][point[0]] = point
                    self.offsets[str(path)] = stream.tell()
        return {tag: sorted(points.values(), key=lambda p: p[0]) for tag, points in self.values.items() if points}


class Dashboard:
    def __init__(self, manifest):
        self.manifest_path = Path(manifest).resolve()
        self.root = self.manifest_path.parent
        self.readers = {}
        self.lock = threading.RLock()
        self.refreshed = 0
        self.snapshot = {}
        self.metadata = {}
        self.image_lock = threading.Lock()

    def refresh(self):
        with self.lock:
            if self.snapshot and time.monotonic() - self.refreshed < 5:
                return self.snapshot
            manifest = read_json(self.manifest_path)
            observer = read_json(self.root / "status.json") if (self.root / "status.json").exists() else {}
            jobs = []
            for domain, job in manifest["jobs"].items():
                live = observer.get("jobs", {}).get(domain, {})
                versions = sorted(Path(job["expected_run_parent"]).glob("v[0-9]*"), key=lambda p: p.stat().st_mtime)
                curves = self.readers.setdefault(domain, ScalarReader()).read(versions)
                loss = curves.get("loss", [])
                step = max(loss[-1][0] if loss else 0, int(live.get("step", 0)))
                target = int(job.get("estimated_steps", 1500))
                times = curves.get("perf/optimizer_step_time_sec", [])
                steady = [p[1] for p in times if p[0] > 5 and p[1] is not None and p[1] > 0][-30:]
                seconds = statistics.mean(steady) if steady else None
                last_write = max((p[-1][2] for p in curves.values() if p), default=None)
                validations = {}
                for directory in versions:
                    for path in sorted(directory.glob("validation/step-*/metadata.json")):
                        metrics_path = path.parent / "metrics.json"
                        if not metrics_path.is_file() or not (path.parent / "all_rows.png").is_file():
                            continue
                        try:
                            metrics = read_json(metrics_path)
                            metadata = read_json(path)
                            key = int(metrics["step"])
                            self.metadata[(domain, key)] = (path, metadata)
                            validations[key] = {"step": key, "samples": metrics["samples"], "metrics": metrics["metrics"], "published_at": dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc).isoformat()}
                        except (OSError, ValueError, KeyError):
                            continue  # A writer may be publishing this checkpoint now.
                status = live.get("status", "UNKNOWN")
                phase = "training" if loss else "loading"
                if status in {"QUEUING", "PENDING"}:
                    phase = "queued"
                elif status == "RUNNING" and step and step % 500 == 0 and step not in validations:
                    phase = "evaluating"
                elif status == "RUNNING" and step >= target:
                    phase = "finishing"
                elif status == "SUCCEEDED":
                    phase = "completed" if step >= target else "incomplete"
                elif status in {"FAILED", "CANCELLED"}:
                    phase = "failed" if status == "FAILED" else "cancelled"
                checkpoints = []
                for directory in versions:
                    for p in directory.glob("checkpoint-*/state.json"):
                        try:
                            state = read_json(p)
                            checkpoints.append({"step": int(state["global_step"]), "name": p.parent.name})
                        except (OSError, ValueError, KeyError):
                            continue
                jobs.append({
                    "domain": domain, "label": NAMES.get(domain, domain), "color": COLORS.get(domain, "#66707a"),
                    "name": job["name"], "gpu_type": job.get("gpu_type", "GPU"), "gpu_count": int(job.get("gpus", 8)),
                    "status": status, "phase": phase, "step": step, "target": target, "loss": loss[-1][1] if loss else None,
                    "seconds_per_step": seconds, "speed_samples": len(steady),
                    "eta_hours": max(0, target - step) * seconds / 3600 if seconds and phase in {"training", "evaluating", "finishing"} else None,
                    "last_scalar_at": dt.datetime.fromtimestamp(last_write, dt.timezone.utc).isoformat() if last_write else None,
                    "scalar_age_sec": time.time() - last_write if last_write else None,
                    "curves": curves, "validations": [validations[k] for k in sorted(validations)],
                    "checkpoint": max(checkpoints, key=lambda p: p["step"]) if checkpoints else None,
                    "validation_interval": int(manifest.get("monitoring", {}).get("validation_steps", 500)),
                })
            platform_time = observer.get("platform_checked_utc")
            platform_age = time.time() - dt.datetime.fromisoformat(platform_time).timestamp() if platform_time else None
            self.snapshot = {
                "updated_at": utc_now(), "platform_checked_at": platform_time, "platform_age_sec": platform_age,
                "run": self.root.name, "project": manifest.get("project"), "image": manifest.get("image"),
                "workspace": manifest.get("workspace"), "jobs": jobs,
                "metrics": [{"key": k, "label": v[0], "unit": v[1]} for k, v in METRICS.items()],
                "source": {"status": "Inspire job status → status.json", "curves": "TensorBoard events.out.tfevents.*", "cases": "validation/step-*/metadata.json + metrics.json", "step": "已完成的 optimizer step；训练标量原始索引 + 1。", "eta": "剩余 step × 最近 30 步平均耗时；忽略前 5 步，不含后续验证、保存和排队。", "smoothing": "曲线平滑使用向后移动平均，保留原始曲线与原始数据导出。"},
            }
            self.refreshed = time.monotonic()
            return self.snapshot

    def get_metadata(self, domain, step):
        self.refresh()
        if (domain, step) not in self.metadata:
            raise FileNotFoundError("Validation is not published yet")
        return self.metadata[(domain, step)]

    @staticmethod
    def key(sample):
        return f"{sample['category']}/{sample['stem']}"

    def image_url(self, domain, step, sample, role, size=720):
        return "/api/image?" + urlencode({"domain": domain, "step": step, "case": self.key(sample), "role": role, "size": size})

    def cases(self, domain, step):
        _, data = self.get_metadata(domain, step)
        samples = []
        for sample in data["samples"]:
            samples.append({"key": self.key(sample), "category": sample["category"], "stem": sample["stem"], "seed": sample.get("seed"), "metrics": sample.get("metrics", {}), "images": {role: self.image_url(domain, step, sample, role) for role in ROLES if role in sample["paths"]}, "thumbnail": self.image_url(domain, step, sample, "output", 320)})
        return {"domain": domain, "step": step, "samples": samples, "sampling": data.get("requested_sampling", {}), "seed": data.get("seed"), "output_semantics": data.get("output_semantics", {})}

    def image(self, domain, step, key, role, size):
        if role not in ROLES or size not in {0, 320, 720, 1440}:
            raise ValueError("Unsupported image option")
        meta_path, data = self.get_metadata(domain, step)
        sample = next((s for s in data["samples"] if self.key(s) == key), None)
        if sample is None:
            raise FileNotFoundError("Case does not exist")
        path = Path(sample["paths"][role]).resolve()
        if not path.is_relative_to(meta_path.parent.resolve()) or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError("Image is outside the selected validation")
        if size == 0:
            kind = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/" + path.suffix.lower()[1:]
            return path.read_bytes(), kind
        signature = hashlib.sha256(f"{path}:{path.stat().st_mtime_ns}:{size}".encode()).hexdigest()
        cache = self.root / "dashboard_cache" / f"{signature}.webp"
        with self.image_lock:
            if not cache.exists():
                cache.parent.mkdir(exist_ok=True)
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    image.thumbnail((size, size), Image.Resampling.LANCZOS)
                    image.save(cache, "WEBP", quality=90, method=3)
        return cache.read_bytes(), "image/webp"


def serve(manifest, port):
    dashboard = Dashboard(manifest)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = urlsplit(self.path)
            query = parse_qs(route.query)
            one = lambda key, default=None: query.get(key, [default])[0]
            try:
                cache = "no-store"
                if route.path in {"/", "/dashboard.css", "/dashboard.js"}:
                    name = {"/": "index.html", "/dashboard.css": "dashboard.css", "/dashboard.js": "dashboard.js"}[route.path]
                    content = (STATIC / name).read_bytes()
                    kind = {"index.html": "text/html", "dashboard.css": "text/css", "dashboard.js": "text/javascript"}[name] + "; charset=utf-8"
                elif route.path in {"/api/status", "/status.json"}:
                    content = json.dumps(dashboard.refresh(), ensure_ascii=False, allow_nan=False).encode()
                    kind = "application/json; charset=utf-8"
                elif route.path == "/api/cases":
                    content = json.dumps(dashboard.cases(one("domain"), int(one("step"))), ensure_ascii=False, allow_nan=False).encode()
                    kind = "application/json; charset=utf-8"
                elif route.path == "/api/image":
                    content, kind = dashboard.image(one("domain"), int(one("step")), one("case"), one("role"), int(one("size", "720")))
                    cache = "public, max-age=3600"
                elif route.path == "/api/export":
                    metric = one("metric", "loss")
                    if metric not in METRICS:
                        raise ValueError("Unknown metric")
                    domains = set(query.get("domain", []))
                    buf = io.StringIO()
                    writer = csv.writer(buf)
                    writer.writerow(["domain", "optimizer_step", metric, "wall_time_utc"])
                    for job in dashboard.refresh()["jobs"]:
                        if domains and job["domain"] not in domains:
                            continue
                        for step, value, wall in job["curves"].get(metric, []):
                            writer.writerow([job["domain"], step, value, dt.datetime.fromtimestamp(wall, dt.timezone.utc).isoformat()])
                    content, kind = ("\ufeff" + buf.getvalue()).encode(), "text/csv; charset=utf-8"
                elif route.path == "/healthz":
                    content, kind = b'{"ok":true}', "application/json"
                elif route.path == "/favicon.ico":
                    self.send_response(204)
                    self.end_headers()
                    return
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", cache)
                self.send_header("X-Content-Type-Options", "nosniff")
                if route.path == "/api/export":
                    self.send_header("Content-Disposition", 'attachment; filename="teacher-metrics.csv"')
                self.end_headers()
                self.wfile.write(content)
            except (KeyError, ValueError, TypeError, OSError):
                self.send_error(404, "Requested data is unavailable")
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Teacher dashboard: http://127.0.0.1:{port}", flush=True)
    server.serve_forever()
