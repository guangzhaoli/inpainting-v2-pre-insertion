#!/usr/bin/env python3
"""Resumable parallel downloader for a public Google Drive folder.

`gdown --folder` aborts the whole folder when one Google content host times out.
This helper uses gdown only to resolve the official folder tree, then downloads
independent files in parallel with atomic `.part` files and image/JSON checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

import gdown
import requests
from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
CONTROL_FILENAMES = {"google_drive_files.json", "download_report.json"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(rows: list[tuple[str, str]]) -> str:
    payload = json.dumps(sorted(rows), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_download(path: Path, *, expected_suffix: str | None = None) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    suffix = (expected_suffix or path.suffix).lower()
    try:
        if suffix in IMAGE_EXTS:
            with Image.open(path) as image:
                image.verify()
        elif suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return True


def download_one(
    file_info,
    attempts: int,
    previous_row: dict | None,
) -> tuple[str, str]:
    destination = Path(file_info.local_path)
    can_skip = (
        previous_row is not None
        and previous_row.get("id") == file_info.id
        and previous_row.get("remote_path") == file_info.path
        and isinstance(previous_row.get("sha256"), str)
        and validate_download(destination)
        and file_sha256(destination) == previous_row["sha256"]
    )
    if can_skip:
        return "skipped", str(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    url = f"https://drive.usercontent.google.com/download?id={file_info.id}&export=download&confirm=t"
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            offset = part.stat().st_size if part.is_file() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with requests.get(
                url,
                stream=True,
                timeout=(20, 180),
                allow_redirects=True,
                headers=headers,
            ) as response:
                if response.status_code == 416 and offset:
                    if validate_download(part, expected_suffix=destination.suffix):
                        os.replace(part, destination)
                        return "downloaded", str(destination)
                    raise ValueError(f"Server rejected invalid partial range: {part}")
                response.raise_for_status()
                # Google may ignore Range and return 200; restart safely in
                # that case.  A real 206 appends to the retained partial file.
                mode = "ab" if offset and response.status_code == 206 else "wb"
                with part.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            # `.part` hides the real extension, so validate against the final
            # filename.  Without this, a 200 HTML quota/error page could be
            # accepted as a PNG and poison a resumable download permanently.
            if not validate_download(part, expected_suffix=destination.suffix):
                raise ValueError(f"Downloaded file failed validation: {part}")
            os.replace(part, destination)
            return "downloaded", str(destination)
        except ValueError as error:
            # A fully received but invalid payload (usually a quota HTML page)
            # must not be resumed as though it were media.
            last_error = error
            part.unlink(missing_ok=True)
            if attempt < attempts:
                time.sleep(min(2 * attempt, 20))
        except Exception as error:
            last_error = error
            # Preserve a genuine interrupted transfer for the next Range
            # request.  The final destination remains untouched until a full
            # validation succeeds.
            if attempt < attempts:
                time.sleep(min(2 * attempt, 20))
    raise RuntimeError(f"Failed {destination} after {attempts} attempts: {last_error}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument("--expected-tree-sha256", default=None)
    parser.add_argument("--expected-content-sha256", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers <= 0 or args.attempts <= 0:
        raise ValueError("--workers and --attempts must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "google_drive_files.json"
    previous_rows: dict[str, dict] = {}
    if manifest_path.is_file():
        try:
            previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(previous_manifest, list):
                previous_rows = {
                    str(row["remote_path"]): row
                    for row in previous_manifest
                    if isinstance(row, dict) and "remote_path" in row
                }
        except (OSError, ValueError, TypeError):
            previous_rows = {}
    files = gdown.download_folder(
        url=args.url,
        output=str(args.output),
        quiet=False,
        # A public Drive folder is limited to 50 children per directory by
        # gdown.  Never accept a silently truncated tree if upstream changes.
        remaining_ok=False,
        skip_download=True,
        resume=True,
    )
    if not files:
        raise RuntimeError(f"Could not resolve Google Drive folder: {args.url}")
    output_resolved = args.output.resolve()
    remote_paths = [str(item.path) for item in files]
    local_paths = [str(Path(item.local_path).resolve()) for item in files]
    file_ids = [str(item.id) for item in files]
    if len(set(remote_paths)) != len(files) or len(set(local_paths)) != len(files):
        raise ValueError("Google Drive tree contains duplicate remote or local paths")
    if len(set(file_ids)) != len(files):
        raise ValueError("Google Drive tree contains duplicate file IDs")
    for item in files:
        local_path = Path(item.local_path).resolve()
        try:
            local_path.relative_to(output_resolved)
        except ValueError as error:
            raise ValueError(f"Resolved Drive path escapes output root: {local_path}") from error
        if local_path.parent == output_resolved and local_path.name in CONTROL_FILENAMES:
            raise ValueError(f"Remote file collides with downloader control file: {item.path}")
    tree_sha256 = canonical_digest([(str(item.path), str(item.id)) for item in files])
    if args.expected_count is not None and len(files) != args.expected_count:
        raise ValueError(
            f"Google Drive tree count mismatch: actual={len(files)} expected={args.expected_count}"
        )
    if args.expected_tree_sha256 and tree_sha256 != args.expected_tree_sha256:
        raise ValueError(
            f"Google Drive tree digest mismatch: actual={tree_sha256} "
            f"expected={args.expected_tree_sha256}"
        )

    counts = {"downloaded": 0, "skipped": 0}
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_item = {
            executor.submit(
                download_one,
                item,
                args.attempts,
                previous_rows.get(str(item.path)),
            ): item
            for item in files
        }
        completed = 0
        for future in as_completed(future_to_item):
            completed += 1
            item = future_to_item[future]
            try:
                status, _ = future.result()
                counts[status] += 1
            except Exception as error:
                failures.append(f"{item.path}: {error}")
            if completed % 25 == 0 or completed == len(files):
                print(
                    f"completed={completed}/{len(files)} downloaded={counts['downloaded']} "
                    f"skipped={counts['skipped']} failures={len(failures)}",
                    flush=True,
                )
    manifest = []
    if not failures:
        for item in files:
            local_path = Path(item.local_path)
            if not validate_download(local_path):
                failures.append(f"{item.path}: final file failed validation")
                continue
            manifest.append(
                {
                    "id": item.id,
                    "remote_path": item.path,
                    "local_path": str(local_path),
                    "bytes": local_path.stat().st_size,
                    "sha256": file_sha256(local_path),
                }
            )
    content_sha256 = (
        canonical_digest([(str(row["remote_path"]), str(row["sha256"])) for row in manifest])
        if not failures
        else None
    )
    if (
        not failures
        and args.expected_content_sha256
        and content_sha256 != args.expected_content_sha256
    ):
        failures.append(
            f"content digest mismatch: actual={content_sha256} "
            f"expected={args.expected_content_sha256}"
        )
    report = {
        "file_count": len(files),
        "tree_sha256": tree_sha256,
        "content_sha256": content_sha256,
        **counts,
        "failures": failures,
    }
    (args.output / "download_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if failures:
        raise RuntimeError("Google Drive downloads failed:\n- " + "\n- ".join(failures[:30]))
    temporary_manifest: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=manifest_path.parent,
            prefix=f".{manifest_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_manifest = Path(handle.name)
            handle.write(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_manifest, manifest_path)
        temporary_manifest = None
    finally:
        if temporary_manifest is not None:
            temporary_manifest.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
