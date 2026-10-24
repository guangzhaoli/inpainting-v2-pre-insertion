#!/usr/bin/env python3
"""High-performance multi-method insertion result review gallery.

Serves ref / tar / masks / mask-overlays / baseline+ours outputs with:
  - on-disk thumbnail cache (JPEG)
  - keyboard-driven pick / shortlist / save
  - JSONL decision log

Usage:
  conda activate qwen-inpainting-pre-insertion
  python scripts/review_gallery/server.py --port 8765
  # open http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import mimetypes
import re
import threading
import time
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[2]
EVAL = Path(
    "/inspire/hdd/global_user/250041225002/project/qwen-inpainting/a2edit/eval"
)
PI_ROOT = EVAL / "paper_insertion_benchmark_v1" / "paperinsert_900"
AI_ROOT = EVAL / "paper_insertion_benchmark_v1" / "anyinsertion_official"
TS_ROOT = EVAL / "testset_20260120"

PI_BASELINES = EVAL / "paper_insertion_benchmark_v1" / "baselines_infer"
AI_BASELINES = EVAL / "anyinsertion_official_baselines_infer"
TS_BASELINES = EVAL / "testset_20260120_baselines_infer"

OURS_ROOT = (
    REPO
    / "output/inference/ours_ckpt_two_testsets"
)
OURS_PI = (
    OURS_ROOT
    / "paperinsert_900/final_architecture_maskmix_geometry_region_perceptual/checkpoint-0-1000"
)
OURS_AI = (
    OURS_ROOT
    / "anyinsertion_official/final_architecture_maskmix_geometry_region_perceptual/checkpoint-0-1000"
)
OURS_TS = (
    OURS_ROOT
    / "testset_20260120/final_architecture_maskmix_geometry_region_perceptual/checkpoint-0-1000"
)

DEFAULT_CACHE = REPO / "output/review_gallery_cache"
DEFAULT_DECISIONS = REPO / "output/review_gallery_decisions"

METHOD_ORDER = [
    "ours",
    "anydoor",
    "insert_anything",
    "qwen_edit_2511",
    "qwen_image_2_0",
    "flux_kontext",
    "a2edit_moe",
]

METHOD_LABELS = {
    "ours": "Ours (restored_raw)",
    "anydoor": "AnyDoor",
    "insert_anything": "InsertAnything",
    "qwen_edit_2511": "Qwen-Edit-2511",
    "qwen_image_2_0": "Qwen-Image-2.0*",
    "flux_kontext": "FLUX-Kontext",
    "a2edit_moe": "A2Edit-MoE",
}

SAFE_ID = re.compile(r"^[A-Za-z0-9_./\-]+$")


@dataclass
class Case:
    case_id: str
    suite: str
    major: str
    minor: str
    ref_image: str
    ref_mask: str
    tar_image: str
    tar_mask: str
    results: dict[str, str]  # method -> abs path

    def to_public(self) -> dict[str, Any]:
        d = asdict(self)
        d["methods"] = [m for m in METHOD_ORDER if m in self.results]
        d["method_labels"] = {m: METHOD_LABELS.get(m, m) for m in d["methods"]}
        # Short, proxy-friendly asset URLs (avoid putting absolute FS paths in <img src>).
        # Append ?v=<mtime_ns> so browser cache invalidates when the underlying
        # file changes (e.g. ours switched from output.png -> output_restored_raw.png).
        cid = _enc_id(self.case_id)

        def _v(path: str) -> str:
            try:
                return str(Path(path).stat().st_mtime_ns)
            except OSError:
                return "0"

        d["assets"] = {
            "ref_image": f"/asset/{self.suite}/{cid}/ref_image?v={_v(self.ref_image)}",
            "ref_mask": f"/asset/{self.suite}/{cid}/ref_mask?v={_v(self.ref_mask)}",
            "tar_image": f"/asset/{self.suite}/{cid}/tar_image?v={_v(self.tar_image)}",
            "tar_mask": f"/asset/{self.suite}/{cid}/tar_mask?v={_v(self.tar_mask)}",
            "tar_overlay": (
                f"/asset/{self.suite}/{cid}/tar_overlay"
                f"?v={_v(self.tar_image)}-{_v(self.tar_mask)}"
            ),
            "results": {
                m: (
                    f"/asset/{self.suite}/{cid}/result/{m}"
                    f"?v={_v(self.results[m])}&src={Path(self.results[m]).name}"
                )
                for m in d["methods"]
            },
        }
        # Debug: expose which on-disk file ours is serving.
        if "ours" in self.results:
            d["ours_file"] = Path(self.results["ours"]).name
        return d


def _enc_id(case_id: str) -> str:
    # Keep path separators out of URL path; decode on server.
    return case_id.replace("/", "__")


def _dec_id(token: str) -> str:
    return token.replace("__", "/")


class GalleryIndex:
    def __init__(self) -> None:
        self.cases: dict[str, dict[str, Case]] = {}  # suite -> case_id -> Case
        self.order: dict[str, list[str]] = {}
        self.allowed_roots: list[Path] = []
        self._lock = threading.Lock()

    def build(self) -> None:
        t0 = time.time()
        self.cases = {}
        self.order = {}
        self.allowed_roots = [
            PI_ROOT,
            AI_ROOT,
            TS_ROOT,
            PI_BASELINES,
            AI_BASELINES,
            TS_BASELINES,
            OURS_ROOT,
        ]
        self._index_manifest_suite(
            suite="paperinsert_900",
            data_root=PI_ROOT,
            manifest=PI_ROOT / "manifests/paperinsert_900.jsonl",
            baseline_root=PI_BASELINES,
            ours_root=OURS_PI,
            ours_layout="indexed",
        )
        self._index_manifest_suite(
            suite="anyinsertion",
            data_root=AI_ROOT,
            manifest=AI_ROOT / "manifests/anyinsertion_official.jsonl",
            baseline_root=AI_BASELINES,
            ours_root=OURS_AI,
            ours_layout="indexed",
        )
        self._index_folder_suite(
            suite="testset_20260120",
            data_root=TS_ROOT,
            baseline_root=TS_BASELINES,
            ours_root=OURS_TS,
        )
        dt = time.time() - t0
        counts = {s: len(v) for s, v in self.cases.items()}
        print(f"[gallery] indexed {counts} in {dt:.2f}s", flush=True)

    def _register(
        self,
        suite: str,
        case: Case,
    ) -> None:
        self.cases.setdefault(suite, {})[case.case_id] = case
        self.order.setdefault(suite, []).append(case.case_id)

    def _find_result(
        self,
        baseline_root: Path,
        method: str,
        case_id: str,
        major: str,
    ) -> Path | None:
        # baselines: {method}/{case_id}.png  or nested category
        candidates = [
            baseline_root / method / f"{case_id}.png",
            baseline_root / method / major / f"{case_id}.png",
            baseline_root / method / f"{case_id}.jpg",
        ]
        # nested case_id like pets/cats/1
        if "/" in case_id:
            candidates.insert(0, baseline_root / method / f"{case_id}.png")
            candidates.insert(1, baseline_root / method / f"{case_id}.jpg")
        for p in candidates:
            if p.is_file() and p.stat().st_size > 0:
                return p
        return None

    # Prefer restored full-res raw over composited preview for "ours".
    _OURS_OUTPUT_NAMES = (
        "output_restored_raw.png",
        "output.png",
        "output_local.png",
    )

    def _pick_ours_file(self, case_dir: Path) -> Path | None:
        for name in self._OURS_OUTPUT_NAMES:
            p = case_dir / name
            if p.is_file() and p.stat().st_size > 0:
                return p
        return None

    def _find_ours_indexed(self, ours_root: Path, major: str, case_id: str) -> Path | None:
        if not ours_root.is_dir():
            return None
        major_dir = ours_root / major
        if not major_dir.is_dir():
            return None
        # folder names: 001_obj_xxx or just contain case_id
        suffix = f"_{case_id}"
        for p in major_dir.iterdir():
            if not p.is_dir():
                continue
            if p.name == case_id or p.name.endswith(suffix) or p.name.endswith(case_id):
                out = self._pick_ours_file(p)
                if out is not None:
                    return out
        # fallback linear (rare)
        for name in self._OURS_OUTPUT_NAMES:
            for p in major_dir.rglob(name):
                if case_id in p.parent.name and p.is_file() and p.stat().st_size > 0:
                    return p
        return None

    def _find_ours_testset(self, ours_root: Path, case_id: str) -> Path | None:
        # case_id: accessory/1 -> accessory/001_1/output_restored_raw.png
        if not ours_root.is_dir():
            return None
        parts = case_id.split("/")
        stem = parts[-1]
        cat = "/".join(parts[:-1]) if len(parts) > 1 else ""
        cat_dir = ours_root / cat if cat else ours_root
        if not cat_dir.is_dir():
            return None
        # try padded index folders
        for p in cat_dir.iterdir():
            if not p.is_dir():
                continue
            # 001_1, 010_10
            if p.name.endswith(f"_{stem}") or p.name == stem:
                out = self._pick_ours_file(p)
                if out is not None:
                    return out
        return None

    def _index_manifest_suite(
        self,
        suite: str,
        data_root: Path,
        manifest: Path,
        baseline_root: Path,
        ours_root: Path,
        ours_layout: str,
    ) -> None:
        if not manifest.is_file():
            print(f"[gallery] skip {suite}: no manifest {manifest}", flush=True)
            return
        order: list[str] = []
        cases: dict[str, Case] = {}
        with manifest.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                case_id = str(row["case_id"])
                major = str(row.get("major", ""))
                minor = str(row.get("minor", ""))
                ref_image = str((data_root / row["src_image"]).resolve())
                ref_mask = str((data_root / row["src_mask"]).resolve())
                tar_image = str((data_root / row["tar_image"]).resolve())
                tar_mask = str((data_root / row["tar_mask"]).resolve())
                results: dict[str, str] = {}
                for method in METHOD_ORDER:
                    if method == "ours":
                        if ours_layout == "indexed":
                            p = self._find_ours_indexed(ours_root, major, case_id)
                        else:
                            p = None
                    else:
                        p = self._find_result(baseline_root, method, case_id, major)
                    if p is not None:
                        results[method] = str(p.resolve())
                cases[case_id] = Case(
                    case_id=case_id,
                    suite=suite,
                    major=major,
                    minor=minor,
                    ref_image=ref_image,
                    ref_mask=ref_mask,
                    tar_image=tar_image,
                    tar_mask=tar_mask,
                    results=results,
                )
                order.append(case_id)
        self.cases[suite] = cases
        self.order[suite] = order

    def _index_folder_suite(
        self,
        suite: str,
        data_root: Path,
        baseline_root: Path,
        ours_root: Path,
    ) -> None:
        if not data_root.is_dir():
            return
        # Reuse paperinsert_infer folder discovery lightly
        canonical = ("src_image", "src_mask", "tar_image", "tar_mask")
        legacy = ("ref_image", "ref_mask", "tar_image", "tar_mask")
        cases: dict[str, Case] = {}
        order: list[str] = []

        def stem_map(d: Path) -> dict[str, Path]:
            out: dict[str, Path] = {}
            if not d.is_dir():
                return out
            for p in d.iterdir():
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                    out[p.stem] = p
            return out

        for leaf in sorted(data_root.rglob("*")):
            if not leaf.is_dir():
                continue
            rel_parts = leaf.relative_to(data_root).parts
            if any(x.startswith(("results", "resuls", "qc", "reports")) for x in rel_parts):
                continue
            if all((leaf / n).is_dir() for n in canonical):
                schema = "src"
            elif all((leaf / n).is_dir() for n in legacy):
                schema = "ref"
            else:
                continue
            rel_s = str(leaf.relative_to(data_root)).replace("\\", "/")
            top = rel_parts[0] if rel_parts else rel_s
            if schema == "src":
                ref_m = stem_map(leaf / "src_image")
                ref_mask_m = stem_map(leaf / "src_mask")
            else:
                ref_m = stem_map(leaf / "ref_image")
                ref_mask_m = stem_map(leaf / "ref_mask")
            tar_m = stem_map(leaf / "tar_image")
            tar_mask_m = stem_map(leaf / "tar_mask")
            stems = sorted(set(ref_m) & set(ref_mask_m) & set(tar_m) & set(tar_mask_m))
            for stem in stems:
                case_id = f"{rel_s}/{stem}"
                results: dict[str, str] = {}
                for method in METHOD_ORDER:
                    if method == "ours":
                        p = self._find_ours_testset(ours_root, case_id)
                    else:
                        p = self._find_result(baseline_root, method, case_id, top)
                        if p is None:
                            # sometimes stored as method/category/stem.png
                            p = self._find_result(baseline_root, method, stem, rel_s)
                    if p is not None:
                        results[method] = str(p.resolve())
                cases[case_id] = Case(
                    case_id=case_id,
                    suite=suite,
                    major=top,
                    minor="/".join(rel_parts[1:]) if len(rel_parts) > 1 else "",
                    ref_image=str(ref_m[stem].resolve()),
                    ref_mask=str(ref_mask_m[stem].resolve()),
                    tar_image=str(tar_m[stem].resolve()),
                    tar_mask=str(tar_mask_m[stem].resolve()),
                    results=results,
                )
                order.append(case_id)
        self.cases[suite] = cases
        self.order[suite] = order

    def list_cases(
        self,
        suite: str,
        major: str | None = None,
        q: str | None = None,
        method: str | None = None,
        only_missing: str | None = None,
    ) -> list[dict[str, Any]]:
        ids = self.order.get(suite, [])
        out: list[dict[str, Any]] = []
        for cid in ids:
            c = self.cases[suite][cid]
            if major and c.major != major:
                continue
            if q and q.lower() not in c.case_id.lower():
                continue
            if method and method not in c.results:
                continue
            if only_missing and only_missing in c.results:
                continue
            out.append(

                {
                    "case_id": c.case_id,
                    "major": c.major,
                    "minor": c.minor,
                    "n_methods": len(c.results),
                    "methods": [m for m in METHOD_ORDER if m in c.results],
                }
            )
        return out

    def get(self, suite: str, case_id: str) -> Case | None:
        return self.cases.get(suite, {}).get(case_id)

    def is_allowed(self, path: Path) -> bool:
        try:
            rp = path.resolve()
        except Exception:
            return False
        for root in self.allowed_roots:
            try:
                rp.relative_to(root.resolve())
                return rp.is_file()
            except ValueError:
                continue
        return False


class ThumbCache:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def key(
        self,
        path: Path,
        width: int,
        kind: str,
        mask_path: Path | None = None,
    ) -> Path:
        try:
            mtime = path.stat().st_mtime_ns
        except FileNotFoundError:
            mtime = 0
        mask_part = ""
        if mask_path is not None:
            try:
                mask_part = f"|{mask_path.resolve()}|{mask_path.stat().st_mtime_ns}"
            except FileNotFoundError:
                mask_part = f"|{mask_path}"
        raw = f"{path.resolve()}|{mtime}|{width}|{kind}{mask_part}"
        h = hashlib.sha1(raw.encode()).hexdigest()
        return self.root / kind / f"{width}" / f"{h}.jpg"

    def get_or_make(
        self,
        path: Path,
        width: int,
        kind: str = "rgb",
        mask_path: Path | None = None,
        overlay_alpha: float = 0.45,
    ) -> Path:
        width = max(32, min(int(width), 2048))
        out = self.key(path, width, kind, mask_path=mask_path)
        if out.is_file() and out.stat().st_size > 0:
            return out
        out.parent.mkdir(parents=True, exist_ok=True)
        # Build outside the global lock when possible; only serialize the final replace.
        tmp = out.with_suffix(f".{threading.get_ident()}.tmp.jpg")
        try:
            if kind == "mask":
                im = Image.open(path).convert("L")
                rgb = Image.new("RGB", im.size, (20, 20, 20))
                green = Image.new("RGB", im.size, (40, 220, 120))
                rgb.paste(green, mask=im.point(lambda v: 255 if v > 128 else 0))
                im = rgb
            elif kind == "overlay":
                if mask_path is None:
                    raise ValueError("mask_path required for overlay")
                from PIL import ImageChops, ImageFilter

                base = Image.open(path).convert("RGB")
                mask = Image.open(mask_path).convert("L")
                if mask.size != base.size:
                    mask = mask.resize(base.size, Image.NEAREST)
                edge = mask.point(lambda v: 255 if v > 128 else 0)
                dil = edge.filter(ImageFilter.MaxFilter(5))
                border = ImageChops.subtract(dil, edge)
                tinted = Image.blend(base, Image.new("RGB", base.size, (255, 40, 40)), overlay_alpha * 0.35)
                tinted.paste(Image.new("RGB", base.size, (255, 50, 50)), mask=border)
                base_rgba = tinted.convert("RGBA")
                fill = Image.new("RGBA", base.size, (255, 60, 60, int(180 * overlay_alpha)))
                base_rgba.paste(fill, mask=edge)
                im = base_rgba.convert("RGB")
            else:
                im = Image.open(path).convert("RGB")
            if width and im.width > width:
                h = int(im.height * (width / im.width))
                im = im.resize((width, max(1, h)), Image.BILINEAR)
            im.save(tmp, format="JPEG", quality=85, optimize=True)
            with self._lock:
                if out.is_file() and out.stat().st_size > 0:
                    tmp.unlink(missing_ok=True)
                    return out
                tmp.replace(out)
            return out
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass


class DecisionStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def path_for(self, suite: str) -> Path:
        return self.root / f"{suite}.jsonl"

    def load(self, suite: str) -> dict[str, dict[str, Any]]:
        p = self.path_for(suite)
        out: dict[str, dict[str, Any]] = {}
        if not p.is_file():
            return out
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                out[str(row["case_id"])] = row
        return out

    def upsert(self, suite: str, row: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            data = self.load(suite)
            row = dict(row)
            row["suite"] = suite
            row["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            data[str(row["case_id"])] = row
            p = self.path_for(suite)
            tmp = p.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for cid in sorted(data):
                    f.write(json.dumps(data[cid], ensure_ascii=False) + "\n")
            tmp.replace(p)
            return row

    def export_shortlist(self, suite: str, decision: str = "keep") -> list[dict[str, Any]]:
        data = self.load(suite)
        return [v for v in data.values() if v.get("decision") == decision]


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

INDEX = GalleryIndex()
THUMBS: ThumbCache
DECISIONS: DecisionStore
STATIC_DIR = Path(__file__).resolve().parent / "static"


class Handler(BaseHTTPRequestHandler):
    # HTTP/1.0 avoids keep-alive edge cases with ThreadingHTTPServer that can
    # leave browsers spinning on pending image requests.
    protocol_version = "HTTP/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        super().log_message(fmt, *args)

    def _send(
        self,
        code: int,
        body: bytes,
        content_type: str,
        cache: str | None = None,
        extra: dict[str, str] | None = None,
    ) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        if cache:
            self.send_header("Cache-Control", cache)
        if extra:
            for k, v in extra.items():
                self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
            self.wfile.flush()
        except BrokenPipeError:
            pass

    def _json(self, obj: Any, code: int = 200, cache: str | None = "no-store") -> None:
        body = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8", cache=cache)

    def _read_json(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(n) if n else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        path = unquote(u.path)
        qs = parse_qs(u.query)

        if path in {"/", "/index.html"}:
            html = (STATIC_DIR / "index.html").read_bytes()
            self._send(200, html, "text/html; charset=utf-8", cache="no-cache")
            return
        if path.startswith("/static/"):
            rel = path[len("/static/") :]
            fp = (STATIC_DIR / rel).resolve()
            if not str(fp).startswith(str(STATIC_DIR.resolve())) or not fp.is_file():
                self._json({"error": "not found"}, 404)
                return
            data = fp.read_bytes()
            ctype = mimetypes.guess_type(str(fp))[0] or "application/octet-stream"
            self._send(200, data, ctype, cache="public, max-age=3600")
            return

        if path == "/api/meta":
            suites = []
            for s, order in INDEX.order.items():
                majors = sorted({INDEX.cases[s][c].major for c in order})
                suites.append(
                    {
                        "id": s,
                        "n": len(order),
                        "majors": majors,
                        "methods": METHOD_ORDER,
                        "method_labels": METHOD_LABELS,
                    }
                )
            self._json({"suites": suites})
            return

        if path == "/api/cases":
            suite = (qs.get("suite") or ["paperinsert_900"])[0]
            major = (qs.get("major") or [""])[0] or None
            q = (qs.get("q") or [""])[0] or None
            method = (qs.get("method") or [""])[0] or None
            items = INDEX.list_cases(suite, major=major, q=q, method=method)
            dec = DECISIONS.load(suite)
            for it in items:
                d = dec.get(it["case_id"])
                if d:
                    it["decision"] = d.get("decision")
                    it["note"] = d.get("note", "")
                    it["stars"] = d.get("stars", 0)
            self._json({"suite": suite, "items": items, "total": len(items)})
            return

        if path.startswith("/api/case/"):
            # /api/case/{suite}/{case_id...}
            rest = path[len("/api/case/") :]
            parts = rest.split("/", 1)
            if len(parts) != 2:
                self._json({"error": "bad path"}, 400)
                return
            suite, case_id = parts[0], parts[1]
            case_id = unquote(case_id)
            c = INDEX.get(suite, case_id)
            if not c:
                self._json({"error": "not found"}, 404)
                return
            dec = DECISIONS.load(suite).get(case_id, {})
            pub = c.to_public()
            pub["decision"] = dec.get("decision")
            pub["note"] = dec.get("note", "")
            pub["stars"] = dec.get("stars", 0)
            # ordered neighbor ids
            order = INDEX.order.get(suite, [])
            try:
                i = order.index(case_id)
            except ValueError:
                i = -1
            pub["index"] = i
            pub["total"] = len(order)
            pub["prev_id"] = order[i - 1] if i > 0 else None
            pub["next_id"] = order[i + 1] if 0 <= i < len(order) - 1 else None
            self._json(pub)
            return

        if path == "/api/decisions":
            suite = (qs.get("suite") or ["paperinsert_900"])[0]
            self._json({"suite": suite, "items": list(DECISIONS.load(suite).values())})
            return

        if path == "/api/export":
            suite = (qs.get("suite") or ["paperinsert_900"])[0]
            decision = (qs.get("decision") or ["keep"])[0]
            rows = DECISIONS.export_shortlist(suite, decision=decision)
            body = ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else "")).encode()
            self._send(
                200,
                body,
                "application/x-ndjson; charset=utf-8",
                cache="no-store",
                extra={
                    "Content-Disposition": f'attachment; filename="{suite}_{decision}.jsonl"'
                },
            )
            return

        if path == "/img":
            self._serve_img(qs)
            return

        if path.startswith("/asset/"):
            self._serve_asset(path, qs)
            return

        self._json({"error": "not found", "path": path}, 404)

    def do_POST(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        path = unquote(u.path)
        if path == "/api/decision":
            row = self._read_json()
            suite = str(row.get("suite") or "")
            case_id = str(row.get("case_id") or "")
            if not suite or not case_id:
                self._json({"error": "suite/case_id required"}, 400)
                return
            if INDEX.get(suite, case_id) is None:
                self._json({"error": "unknown case"}, 404)
                return
            saved = DECISIONS.upsert(
                suite,
                {
                    "case_id": case_id,
                    "decision": row.get("decision") or "none",
                    "note": row.get("note") or "",
                    "stars": int(row.get("stars") or 0),
                    "methods_shown": row.get("methods_shown") or [],
                },
            )
            self._json({"ok": True, "item": saved})
            return
        if path == "/api/reload":
            INDEX.build()
            self._json({"ok": True, "counts": {s: len(v) for s, v in INDEX.cases.items()}})
            return
        self._json({"error": "not found"}, 404)

    def _serve_bytes_image(
        self,
        path: Path,
        *,
        kind: str = "rgb",
        mask_path: Path | None = None,
        width: int = 0,
    ) -> None:
        if not INDEX.is_allowed(path):
            self._json({"error": f"path not allowed: {path}"}, 403)
            return
        if mask_path is not None and not INDEX.is_allowed(mask_path):
            self._json({"error": f"mask not allowed: {mask_path}"}, 403)
            return
        try:
            if kind in {"overlay", "mask"} or (width and width > 0):
                w = width if width > 0 else min(path_stat_w(path), 1600)
                thumb = THUMBS.get_or_make(path, w, kind=kind, mask_path=mask_path)
                data = thumb.read_bytes()
                ctype = "image/jpeg"
            else:
                data = path.read_bytes()
                ctype = mimetypes.guess_type(str(path))[0] or "image/png"
            # Short cache: result files may switch (e.g. output.png -> restored_raw).
            self._send(200, data, ctype, cache="public, max-age=60")
        except Exception as e:
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def _serve_img(self, qs: dict[str, list[str]]) -> None:
        kind = (qs.get("kind") or ["rgb"])[0]  # rgb|mask|overlay
        raw_path = (qs.get("path") or [""])[0]
        mask_raw = (qs.get("mask") or [""])[0]
        width = int((qs.get("w") or ["0"])[0] or 0)
        if not raw_path:
            self._json({"error": "path required"}, 400)
            return
        path = Path(raw_path)
        mp = Path(mask_raw) if mask_raw else None
        self._serve_bytes_image(path, kind=kind, mask_path=mp, width=width)

    def _serve_asset(self, path: str, qs: dict[str, list[str]]) -> None:
        """GET /asset/{suite}/{case_token}/{slot}[/method]?w=320"""
        # path: /asset/suite/case/slot or /asset/suite/case/result/method
        parts = [p for p in path.strip("/").split("/") if p]
        # ["asset", suite, case_token, slot, (method)]
        if len(parts) < 4:
            self._json({"error": "bad asset path", "parts": parts}, 400)
            return
        suite = parts[1]
        case_id = _dec_id(unquote(parts[2]))
        slot = parts[3]
        method = parts[4] if len(parts) >= 5 else ""
        width = int((qs.get("w") or ["0"])[0] or 0)
        case = INDEX.get(suite, case_id)
        if case is None:
            self._json({"error": "unknown case", "suite": suite, "case_id": case_id}, 404)
            return
        if slot == "ref_image":
            self._serve_bytes_image(Path(case.ref_image), kind="rgb", width=width)
        elif slot == "ref_mask":
            self._serve_bytes_image(Path(case.ref_mask), kind="mask", width=width or 512)
        elif slot == "tar_image":
            self._serve_bytes_image(Path(case.tar_image), kind="rgb", width=width)
        elif slot == "tar_mask":
            self._serve_bytes_image(Path(case.tar_mask), kind="mask", width=width or 512)
        elif slot == "tar_overlay":
            self._serve_bytes_image(
                Path(case.tar_image),
                kind="overlay",
                mask_path=Path(case.tar_mask),
                width=width or 512,
            )
        elif slot == "result":
            if not method or method not in case.results:
                self._json({"error": "missing result", "method": method}, 404)
                return
            self._serve_bytes_image(Path(case.results[method]), kind="rgb", width=width)
        else:
            self._json({"error": f"unknown slot {slot}"}, 400)


def path_stat_w(path: Path) -> int:
    try:
        with Image.open(path) as im:
            return im.width
    except Exception:
        return 1024


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--decisions-dir", type=Path, default=DEFAULT_DECISIONS)
    return p.parse_args()


def main() -> int:
    global THUMBS, DECISIONS
    args = parse_args()
    THUMBS = ThumbCache(args.cache_dir)
    DECISIONS = DecisionStore(args.decisions_dir)
    INDEX.build()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"[gallery] http://{args.host}:{args.port}  "
        f"cache={args.cache_dir} decisions={args.decisions_dir}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[gallery] stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
