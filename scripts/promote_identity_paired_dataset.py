#!/usr/bin/env python3
"""Atomically promote an identity-paired benchmark candidate.

The command deliberately has no portable rename fallback.  It requires Linux
``renameat2(2)`` support for both ``RENAME_EXCHANGE`` and
``RENAME_NOREPLACE`` and proves those operations on the destination filesystem
before touching either release directory.

Promotion has two namespace mutations::

    stable <RENAME_EXCHANGE> candidate
    candidate (the old stable tree) <RENAME_NOREPLACE> backup

Both the old and new manifest hashes are required on the command line.  Before
the first rename, every file authenticated by each ``CHECKSUMS.sha256`` is
re-hashed and fsynced along with its directories.  A failed post-mutation
verification triggers an inode-guided rollback to the original
``stable``/``candidate`` layout.  The backup name is never replaced.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import re
import stat
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence


MANIFEST_RELATIVE_PATH = Path("manifests/paperinsert_900.jsonl")
CHECKSUMS_RELATIVE_PATH = Path("CHECKSUMS.sha256")

RENAME_NOREPLACE = 1
RENAME_EXCHANGE = 2

_DIRECTORY_OPEN_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
_FILE_OPEN_FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_CHECKSUM_ROW_RE = re.compile(r"^([0-9a-f]{64})  ([^\x00]+)$")
_MAX_CHECKSUM_MANIFEST_BYTES = 64 * 1024 * 1024


class PromotionError(RuntimeError):
    """Base class for a refused or failed promotion."""


class PreconditionsError(PromotionError):
    """The requested paths or expected hashes are not safe to promote."""


class RollbackError(PromotionError):
    """The promotion failed and the original layout could not be proven."""


@dataclass(frozen=True)
class ReleaseHashes:
    manifest_sha256: str
    checksums_sha256: str


@dataclass(frozen=True)
class EntryIdentity:
    device: int
    inode: int

    @classmethod
    def from_stat(cls, result: os.stat_result) -> "EntryIdentity":
        return cls(device=result.st_dev, inode=result.st_ino)


@dataclass(frozen=True)
class AuthoritativeFileSnapshot:
    relative_path: str
    sha256: str
    identity: EntryIdentity
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(
        cls,
        *,
        relative_path: str,
        sha256: str,
        result: os.stat_result,
    ) -> "AuthoritativeFileSnapshot":
        return cls(
            relative_path=relative_path,
            sha256=sha256,
            identity=EntryIdentity.from_stat(result),
            size=result.st_size,
            mtime_ns=result.st_mtime_ns,
            ctime_ns=result.st_ctime_ns,
        )


@dataclass
class Slot:
    """A directory entry addressed relative to a pinned parent directory fd."""

    label: str
    path: Path
    parent_fd: int

    @property
    def name(self) -> str:
        return self.path.name

    def close(self) -> None:
        if self.parent_fd >= 0:
            os.close(self.parent_fd)
            self.parent_fd = -1


class RenameAt2:
    """Small checked wrapper around glibc's ``renameat2`` symbol."""

    def __init__(self) -> None:
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            function = libc.renameat2
        except AttributeError as exc:
            raise PreconditionsError(
                "libc does not expose renameat2; refusing promotion without an atomic fallback"
            ) from exc
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        self._function = function

    def rename(
        self,
        old_dir_fd: int,
        old_name: str,
        new_dir_fd: int,
        new_name: str,
        flags: int,
    ) -> None:
        result = self._function(
            old_dir_fd,
            os.fsencode(old_name),
            new_dir_fd,
            os.fsencode(new_name),
            flags,
        )
        if result != 0:
            error_number = ctypes.get_errno()
            raise OSError(
                error_number,
                os.strerror(error_number),
                f"{old_name} -> {new_name} (flags={flags})",
            )


def _normalize_digest(value: str, *, label: str) -> str:
    if not _SHA256_RE.fullmatch(value):
        raise PreconditionsError(f"{label} must be exactly 64 hexadecimal characters")
    return value.lower()


def _argparse_digest(value: str) -> str:
    try:
        return _normalize_digest(value, label="SHA-256")
    except PreconditionsError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _validate_absolute_path(value: Path | str, *, label: str) -> Path:
    raw = os.fspath(value)
    path = Path(raw)
    if not path.is_absolute():
        raise PreconditionsError(f"{label} must be an absolute path: {raw!r}")
    normalized = os.path.normpath(raw)
    if raw != normalized:
        raise PreconditionsError(
            f"{label} must be lexically normalized (no '.', '..', or trailing slash): {raw!r}"
        )
    if path == path.parent:
        raise PreconditionsError(f"{label} cannot be a filesystem root")
    if path.name in {"", ".", ".."}:
        raise PreconditionsError(f"{label} has an unsafe final component: {raw!r}")
    return path


def _open_absolute_directory_nofollow(path: Path, *, label: str) -> int:
    """Open an absolute directory one component at a time without symlinks."""

    if not path.is_absolute():
        raise PreconditionsError(f"{label} is not absolute: {path}")
    current_fd = os.open(path.anchor, _DIRECTORY_OPEN_FLAGS)
    try:
        for component in path.parts[1:]:
            try:
                next_fd = os.open(component, _DIRECTORY_OPEN_FLAGS, dir_fd=current_fd)
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    detail = "contains a symbolic link"
                else:
                    detail = str(exc)
                raise PreconditionsError(
                    f"cannot safely open {label} {path}: {detail}"
                ) from exc
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _open_slot(path: Path, *, label: str) -> Slot:
    parent_fd = _open_absolute_directory_nofollow(path.parent, label=f"{label} parent")
    return Slot(label=label, path=path, parent_fd=parent_fd)


def _entry_stat(slot: Slot) -> os.stat_result | None:
    try:
        return os.stat(slot.name, dir_fd=slot.parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _directory_identity(slot: Slot, *, required: bool = True) -> EntryIdentity | None:
    result = _entry_stat(slot)
    if result is None:
        if required:
            raise PreconditionsError(f"{slot.label} does not exist: {slot.path}")
        return None
    if stat.S_ISLNK(result.st_mode):
        raise PreconditionsError(
            f"{slot.label} must not be a symbolic link: {slot.path}"
        )
    if not stat.S_ISDIR(result.st_mode):
        raise PreconditionsError(f"{slot.label} must be a directory: {slot.path}")
    return EntryIdentity.from_stat(result)


def _open_root_fd(slot: Slot) -> int:
    try:
        return os.open(slot.name, _DIRECTORY_OPEN_FLAGS, dir_fd=slot.parent_fd)
    except OSError as exc:
        raise PreconditionsError(
            f"cannot safely open {slot.label} {slot.path}: {exc}"
        ) from exc


def _open_relative_regular_file(
    root_fd: int, relative: Path, *, label: str
) -> tuple[int, list[int]]:
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"unsafe internal relative path: {relative}")

    directory_fds: list[int] = []
    current_fd = root_fd
    try:
        for component in relative.parts[:-1]:
            next_fd = os.open(component, _DIRECTORY_OPEN_FLAGS, dir_fd=current_fd)
            directory_fds.append(next_fd)
            current_fd = next_fd
        file_fd = os.open(relative.parts[-1], _FILE_OPEN_FLAGS, dir_fd=current_fd)
        result = os.fstat(file_fd)
        if not stat.S_ISREG(result.st_mode):
            os.close(file_fd)
            raise PreconditionsError(f"{label} is not a regular file: {relative}")
        return file_fd, directory_fds
    except OSError as exc:
        for directory_fd in reversed(directory_fds):
            os.close(directory_fd)
        raise PreconditionsError(
            f"cannot safely open {label} {relative}: {exc}"
        ) from exc
    except BaseException:
        for directory_fd in reversed(directory_fds):
            os.close(directory_fd)
        raise


def _sha256_fd(file_fd: int) -> str:
    digest = hashlib.sha256()
    while True:
        chunk = os.read(file_fd, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _read_fd(file_fd: int, *, byte_limit: int, label: str) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(file_fd, min(1024 * 1024, byte_limit + 1 - total))
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        total += len(chunk)
        if total > byte_limit:
            raise PreconditionsError(
                f"{label} exceeds the {byte_limit}-byte safety limit"
            )


def _hash_relative_file(slot: Slot, relative: Path, *, label: str) -> str:
    root_fd = _open_root_fd(slot)
    file_fd = -1
    directory_fds: list[int] = []
    try:
        file_fd, directory_fds = _open_relative_regular_file(
            root_fd, relative, label=label
        )
        before = os.fstat(file_fd)
        digest = _sha256_fd(file_fd)
        after = os.fstat(file_fd)
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(
            getattr(before, field) != getattr(after, field) for field in stable_fields
        ):
            raise PreconditionsError(
                f"{label} changed while it was being hashed: {slot.path / relative}"
            )
        return digest
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        for directory_fd in reversed(directory_fds):
            os.close(directory_fd)
        os.close(root_fd)


def _release_hashes(slot: Slot) -> ReleaseHashes:
    return ReleaseHashes(
        manifest_sha256=_hash_relative_file(
            slot,
            MANIFEST_RELATIVE_PATH,
            label=f"{slot.label} canonical manifest",
        ),
        checksums_sha256=_hash_relative_file(
            slot,
            CHECKSUMS_RELATIVE_PATH,
            label=f"{slot.label} checksum manifest",
        ),
    )


def _assert_release_hashes(slot: Slot, expected: ReleaseHashes) -> None:
    actual = _release_hashes(slot)
    if actual != expected:
        raise PreconditionsError(
            f"{slot.label} release hash mismatch at {slot.path}: "
            f"expected manifest={expected.manifest_sha256}, CHECKSUMS={expected.checksums_sha256}; "
            f"found manifest={actual.manifest_sha256}, CHECKSUMS={actual.checksums_sha256}"
        )


def _authoritative_checksum_rows(
    slot: Slot,
    *,
    expected_manifest_sha256: str,
    expected_checksums_sha256: str,
) -> list[tuple[Path, str]]:
    root_fd = _open_root_fd(slot)
    file_fd = -1
    directory_fds: list[int] = []
    try:
        file_fd, directory_fds = _open_relative_regular_file(
            root_fd,
            CHECKSUMS_RELATIVE_PATH,
            label=f"{slot.label} checksum manifest",
        )
        before = os.fstat(file_fd)
        payload = _read_fd(
            file_fd,
            byte_limit=_MAX_CHECKSUM_MANIFEST_BYTES,
            label=f"{slot.label} checksum manifest",
        )
        after = os.fstat(file_fd)
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        for directory_fd in reversed(directory_fds):
            os.close(directory_fd)
        os.close(root_fd)

    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise PreconditionsError(
            f"{slot.label} checksum manifest changed while it was being read"
        )
    actual_checksums_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_checksums_sha256 != expected_checksums_sha256:
        raise PreconditionsError(
            f"{slot.label} parsed checksum manifest SHA-256 mismatch: "
            f"expected {expected_checksums_sha256}, found {actual_checksums_sha256}"
        )

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PreconditionsError(
            f"{slot.label} checksum manifest is not valid UTF-8"
        ) from exc
    if not text or not text.endswith("\n"):
        raise PreconditionsError(
            f"{slot.label} checksum manifest must be non-empty and newline-terminated"
        )

    rows: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = _CHECKSUM_ROW_RE.fullmatch(line)
        if match is None:
            raise PreconditionsError(
                f"{slot.label} checksum manifest has an invalid row at line {line_number}"
            )
        expected_sha256, raw_relative = match.groups()
        pure_path = PurePosixPath(raw_relative)
        if (
            pure_path.is_absolute()
            or not pure_path.parts
            or any(part in {"", ".", ".."} for part in pure_path.parts)
            or pure_path.as_posix() != raw_relative
        ):
            raise PreconditionsError(
                f"{slot.label} checksum manifest has an unsafe path at line {line_number}: "
                f"{raw_relative!r}"
            )
        if raw_relative in seen:
            raise PreconditionsError(
                f"{slot.label} checksum manifest repeats path {raw_relative!r}"
            )
        seen.add(raw_relative)
        rows.append((Path(*pure_path.parts), expected_sha256))

    canonical_manifest = MANIFEST_RELATIVE_PATH.as_posix()
    checksums_by_path = {path.as_posix(): digest for path, digest in rows}
    if checksums_by_path.get(canonical_manifest) != expected_manifest_sha256:
        raise PreconditionsError(
            f"{slot.label} checksum manifest does not authenticate {canonical_manifest} "
            "with the required manifest SHA-256"
        )
    return rows


def _verify_and_fsync_authoritative_release(
    slot: Slot, expected: ReleaseHashes
) -> tuple[AuthoritativeFileSnapshot, ...]:
    """Verify and flush every file authenticated by ``CHECKSUMS.sha256``."""

    rows = _authoritative_checksum_rows(
        slot,
        expected_manifest_sha256=expected.manifest_sha256,
        expected_checksums_sha256=expected.checksums_sha256,
    )
    root_fd = _open_root_fd(slot)
    seen_directories: set[tuple[int, int]] = set()
    snapshots: list[AuthoritativeFileSnapshot] = []
    try:
        for relative, expected_sha256 in rows:
            file_fd = -1
            directory_fds: list[int] = []
            try:
                file_fd, directory_fds = _open_relative_regular_file(
                    root_fd,
                    relative,
                    label=f"{slot.label} authoritative release file",
                )
                before = os.fstat(file_fd)
                actual_sha256 = _sha256_fd(file_fd)
                after = os.fstat(file_fd)
                stable_fields = (
                    "st_dev",
                    "st_ino",
                    "st_size",
                    "st_mtime_ns",
                    "st_ctime_ns",
                )
                if any(
                    getattr(before, field) != getattr(after, field)
                    for field in stable_fields
                ):
                    raise PreconditionsError(
                        f"{slot.label} file changed while being verified: "
                        f"{relative.as_posix()}"
                    )
                if actual_sha256 != expected_sha256:
                    raise PreconditionsError(
                        f"{slot.label} checksum mismatch for {relative.as_posix()}: "
                        f"expected {expected_sha256}, found {actual_sha256}"
                    )
                snapshots.append(
                    AuthoritativeFileSnapshot.from_stat(
                        relative_path=relative.as_posix(),
                        sha256=expected_sha256,
                        result=after,
                    )
                )
                os.fsync(file_fd)
                for directory_fd in reversed(directory_fds):
                    result = os.fstat(directory_fd)
                    identity = (result.st_dev, result.st_ino)
                    if identity not in seen_directories:
                        os.fsync(directory_fd)
                        seen_directories.add(identity)
            except OSError as exc:
                raise PreconditionsError(
                    f"could not verify and fsync {slot.label} file "
                    f"{relative.as_posix()}: {exc}"
                ) from exc
            finally:
                if file_fd >= 0:
                    os.close(file_fd)
                for directory_fd in reversed(directory_fds):
                    os.close(directory_fd)

        checksum_fd = -1
        checksum_directory_fds: list[int] = []
        try:
            checksum_fd, checksum_directory_fds = _open_relative_regular_file(
                root_fd,
                CHECKSUMS_RELATIVE_PATH,
                label=f"{slot.label} checksum manifest",
            )
            os.fsync(checksum_fd)
        finally:
            if checksum_fd >= 0:
                os.close(checksum_fd)
            for directory_fd in reversed(checksum_directory_fds):
                os.close(directory_fd)
        os.fsync(root_fd)
    except OSError as exc:
        raise PreconditionsError(
            f"could not fsync {slot.label} release tree: {exc}"
        ) from exc
    finally:
        os.close(root_fd)
    return tuple(snapshots)


def _assert_authoritative_snapshot(
    slot: Slot,
    expected: ReleaseHashes,
    snapshots: tuple[AuthoritativeFileSnapshot, ...],
) -> None:
    rows = _authoritative_checksum_rows(
        slot,
        expected_manifest_sha256=expected.manifest_sha256,
        expected_checksums_sha256=expected.checksums_sha256,
    )
    authenticated_rows = tuple((path.as_posix(), digest) for path, digest in rows)
    snapshot_rows = tuple(
        (snapshot.relative_path, snapshot.sha256) for snapshot in snapshots
    )
    if authenticated_rows != snapshot_rows:
        raise PreconditionsError(
            f"{slot.label} authoritative checksum path set changed after full verification"
        )

    root_fd = _open_root_fd(slot)
    try:
        for snapshot in snapshots:
            file_fd = -1
            directory_fds: list[int] = []
            try:
                file_fd, directory_fds = _open_relative_regular_file(
                    root_fd,
                    Path(*PurePosixPath(snapshot.relative_path).parts),
                    label=f"{slot.label} authoritative release file",
                )
                result = os.fstat(file_fd)
                current = AuthoritativeFileSnapshot.from_stat(
                    relative_path=snapshot.relative_path,
                    sha256=snapshot.sha256,
                    result=result,
                )
                if current != snapshot:
                    raise PreconditionsError(
                        f"{slot.label} authoritative file changed after full verification: "
                        f"{snapshot.relative_path}"
                    )
            finally:
                if file_fd >= 0:
                    os.close(file_fd)
                for directory_fd in reversed(directory_fds):
                    os.close(directory_fd)
    finally:
        os.close(root_fd)


def _fsync_parent_directories(*slots: Slot) -> None:
    seen: set[tuple[int, int]] = set()
    for slot in slots:
        result = os.fstat(slot.parent_fd)
        identity = (result.st_dev, result.st_ino)
        if identity in seen:
            continue
        os.fsync(slot.parent_fd)
        seen.add(identity)


def _paths_are_nested(left: Path, right: Path) -> bool:
    return left != right and (left.is_relative_to(right) or right.is_relative_to(left))


def _assert_distinct_non_nested_paths(
    stable: Slot, candidate: Slot, backup: Slot
) -> None:
    slots = (stable, candidate, backup)
    for index, left in enumerate(slots):
        for right in slots[index + 1 :]:
            if left.path == right.path:
                raise PreconditionsError(
                    f"{left.label} and {right.label} resolve to the same explicit path"
                )
            if _paths_are_nested(left.path, right.path):
                raise PreconditionsError(
                    f"{left.label} and {right.label} must not contain one another: "
                    f"{left.path}, {right.path}"
                )
            left_parent = EntryIdentity.from_stat(os.fstat(left.parent_fd))
            right_parent = EntryIdentity.from_stat(os.fstat(right.parent_fd))
            if left_parent == right_parent and left.name == right.name:
                raise PreconditionsError(
                    f"{left.label} and {right.label} name the same directory entry"
                )


def _assert_same_filesystem(
    stable: Slot,
    candidate: Slot,
    backup: Slot,
    *,
    old_identity: EntryIdentity,
    new_identity: EntryIdentity,
) -> None:
    devices = {
        old_identity.device,
        new_identity.device,
        os.fstat(stable.parent_fd).st_dev,
        os.fstat(candidate.parent_fd).st_dev,
        os.fstat(backup.parent_fd).st_dev,
    }
    if len(devices) != 1:
        raise PreconditionsError(
            "stable, candidate, backup parent, and both release roots must be on one filesystem; "
            f"observed st_dev values {sorted(devices)}"
        )


def _probe_renameat2_capabilities(
    renamer: RenameAt2,
    stable_parent_fd: int,
    candidate_parent_fd: int,
    backup_parent_fd: int,
) -> None:
    """Exercise both flags across the exact parent-directory pairs in use."""

    prefix = f".identity-promotion-renameat2-probe-{os.getpid()}-{uuid.uuid4().hex}"
    exchange_left = f"{prefix}-exchange-left"
    exchange_right = f"{prefix}-exchange-right"
    noreplace_source = f"{prefix}-noreplace-source"
    noreplace_destination = f"{prefix}-noreplace-destination"
    owned_entries: dict[tuple[int, str], EntryIdentity] = {}
    all_created_identities: set[EntryIdentity] = set()

    def create_probe_file(parent_fd: int, name: str) -> EntryIdentity:
        descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        try:
            identity = EntryIdentity.from_stat(os.fstat(descriptor))
            owned_entries[(parent_fd, name)] = identity
            all_created_identities.add(identity)
            os.fsync(descriptor)
            return identity
        finally:
            os.close(descriptor)

    def unlink_owned_probe_file(parent_fd: int, name: str) -> None:
        key = (parent_fd, name)
        expected_identity = owned_entries.get(key)
        if expected_identity is None:
            raise PreconditionsError(f"probe entry ownership is unknown: {name}")
        actual_identity = EntryIdentity.from_stat(
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        )
        if actual_identity != expected_identity:
            raise PreconditionsError(
                f"probe entry changed ownership before cleanup: {name}"
            )
        os.unlink(name, dir_fd=parent_fd)
        del owned_entries[key]

    try:
        exchange_left_before = create_probe_file(stable_parent_fd, exchange_left)
        exchange_right_before = create_probe_file(candidate_parent_fd, exchange_right)
        renamer.rename(
            stable_parent_fd,
            exchange_left,
            candidate_parent_fd,
            exchange_right,
            RENAME_EXCHANGE,
        )
        owned_entries.update(
            {
                (stable_parent_fd, exchange_left): exchange_right_before,
                (candidate_parent_fd, exchange_right): exchange_left_before,
            }
        )
        if (
            EntryIdentity.from_stat(os.stat(exchange_left, dir_fd=stable_parent_fd))
            != exchange_right_before
        ):
            raise PreconditionsError(
                "RENAME_EXCHANGE probe did not exchange the first inode"
            )
        if (
            EntryIdentity.from_stat(os.stat(exchange_right, dir_fd=candidate_parent_fd))
            != exchange_left_before
        ):
            raise PreconditionsError(
                "RENAME_EXCHANGE probe did not exchange the second inode"
            )

        source_before = create_probe_file(candidate_parent_fd, noreplace_source)
        create_probe_file(backup_parent_fd, noreplace_destination)
        try:
            renamer.rename(
                candidate_parent_fd,
                noreplace_source,
                backup_parent_fd,
                noreplace_destination,
                RENAME_NOREPLACE,
            )
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise PreconditionsError(
                    f"RENAME_NOREPLACE collision probe failed unexpectedly: {exc}"
                ) from exc
        else:
            raise PreconditionsError(
                "RENAME_NOREPLACE probe overwrote an existing destination"
            )

        unlink_owned_probe_file(backup_parent_fd, noreplace_destination)
        # Register the only possible destination inode before the syscall so
        # signal delivery immediately after a successful rename cannot leave
        # an untracked probe entry.
        owned_entries[(backup_parent_fd, noreplace_destination)] = source_before
        renamer.rename(
            candidate_parent_fd,
            noreplace_source,
            backup_parent_fd,
            noreplace_destination,
            RENAME_NOREPLACE,
        )
        del owned_entries[(candidate_parent_fd, noreplace_source)]
        if (
            EntryIdentity.from_stat(
                os.stat(noreplace_destination, dir_fd=backup_parent_fd)
            )
            != source_before
        ):
            raise PreconditionsError("RENAME_NOREPLACE probe moved the wrong inode")
        try:
            os.stat(
                noreplace_source,
                dir_fd=candidate_parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise PreconditionsError(
                "RENAME_NOREPLACE probe left its source entry behind"
            )
    except PromotionError:
        raise
    except OSError as exc:
        raise PreconditionsError(
            "filesystem does not safely support the required renameat2 operations: "
            f"{exc}"
        ) from exc
    finally:
        cleanup_errors: list[str] = []
        for parent_fd, name in tuple(owned_entries):
            try:
                actual_identity = EntryIdentity.from_stat(
                    os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                )
            except FileNotFoundError:
                pass
            else:
                if actual_identity not in all_created_identities:
                    cleanup_errors.append(
                        f"refused to unlink probe entry with changed inode: {name}"
                    )
                    continue
                try:
                    os.unlink(name, dir_fd=parent_fd)
                except OSError as exc:
                    cleanup_errors.append(f"could not unlink probe entry {name}: {exc}")
        seen_parents: set[tuple[int, int]] = set()
        for parent_fd in (stable_parent_fd, candidate_parent_fd, backup_parent_fd):
            try:
                result = os.fstat(parent_fd)
                identity = (result.st_dev, result.st_ino)
                if identity not in seen_parents:
                    os.fsync(parent_fd)
                    seen_parents.add(identity)
            except OSError as exc:
                cleanup_errors.append(f"could not fsync probe parent: {exc}")
        if cleanup_errors:
            raise PreconditionsError("; ".join(cleanup_errors))


def _rename_exchange(renamer: RenameAt2, left: Slot, right: Slot) -> None:
    renamer.rename(
        left.parent_fd, left.name, right.parent_fd, right.name, RENAME_EXCHANGE
    )


def _rename_slot_noreplace(renamer: RenameAt2, source: Slot, destination: Slot) -> None:
    renamer.rename(
        source.parent_fd,
        source.name,
        destination.parent_fd,
        destination.name,
        RENAME_NOREPLACE,
    )


def _assert_identity(slot: Slot, expected: EntryIdentity | None) -> None:
    actual = _directory_identity(slot, required=False)
    if actual != expected:
        raise PreconditionsError(
            f"{slot.label} inode mismatch at {slot.path}: expected {expected}, found {actual}"
        )


def _verify_exchange_state(
    stable: Slot,
    candidate: Slot,
    *,
    old_identity: EntryIdentity,
    new_identity: EntryIdentity,
    old_hashes: ReleaseHashes,
    new_hashes: ReleaseHashes,
    old_snapshots: tuple[AuthoritativeFileSnapshot, ...],
    new_snapshots: tuple[AuthoritativeFileSnapshot, ...],
) -> None:
    _assert_identity(stable, new_identity)
    _assert_identity(candidate, old_identity)
    _assert_release_hashes(stable, new_hashes)
    _assert_release_hashes(candidate, old_hashes)
    _assert_authoritative_snapshot(stable, new_hashes, new_snapshots)
    _assert_authoritative_snapshot(candidate, old_hashes, old_snapshots)


def _verify_final_state(
    stable: Slot,
    candidate: Slot,
    backup: Slot,
    *,
    old_identity: EntryIdentity,
    new_identity: EntryIdentity,
    old_hashes: ReleaseHashes,
    new_hashes: ReleaseHashes,
    old_snapshots: tuple[AuthoritativeFileSnapshot, ...],
    new_snapshots: tuple[AuthoritativeFileSnapshot, ...],
) -> None:
    _assert_identity(stable, new_identity)
    _assert_identity(candidate, None)
    _assert_identity(backup, old_identity)
    _assert_release_hashes(stable, new_hashes)
    _assert_release_hashes(backup, old_hashes)
    _assert_authoritative_snapshot(stable, new_hashes, new_snapshots)
    _assert_authoritative_snapshot(backup, old_hashes, old_snapshots)


def _verify_original_primary_state(
    stable: Slot,
    candidate: Slot,
    *,
    old_identity: EntryIdentity,
    new_identity: EntryIdentity,
    old_hashes: ReleaseHashes,
    new_hashes: ReleaseHashes,
    old_snapshots: tuple[AuthoritativeFileSnapshot, ...],
    new_snapshots: tuple[AuthoritativeFileSnapshot, ...],
) -> None:
    _assert_identity(stable, old_identity)
    _assert_identity(candidate, new_identity)
    _assert_release_hashes(stable, old_hashes)
    _assert_release_hashes(candidate, new_hashes)
    _assert_authoritative_snapshot(stable, old_hashes, old_snapshots)
    _assert_authoritative_snapshot(candidate, new_hashes, new_snapshots)


def _restore_original_layout(
    renamer: RenameAt2,
    stable: Slot,
    candidate: Slot,
    backup: Slot,
    *,
    old_identity: EntryIdentity,
    new_identity: EntryIdentity,
    old_hashes: ReleaseHashes,
    new_hashes: ReleaseHashes,
    old_snapshots: tuple[AuthoritativeFileSnapshot, ...],
    new_snapshots: tuple[AuthoritativeFileSnapshot, ...],
) -> None:
    """Restore namespace layout using root inode identities, never path guesses."""

    stable_identity = _directory_identity(stable, required=False)
    candidate_identity = _directory_identity(candidate, required=False)
    backup_identity = _directory_identity(backup, required=False)

    if stable_identity == old_identity and candidate_identity == new_identity:
        if backup_identity in {old_identity, new_identity}:
            raise RollbackError(
                "cannot prove the original layout: backup aliases a release root inode "
                f"({backup_identity})"
            )
        _verify_original_primary_state(
            stable,
            candidate,
            old_identity=old_identity,
            new_identity=new_identity,
            old_hashes=old_hashes,
            new_hashes=new_hashes,
            old_snapshots=old_snapshots,
            new_snapshots=new_snapshots,
        )
        return

    if stable_identity != new_identity:
        raise RollbackError(
            "cannot roll back: stable does not contain either the proven promoted layout "
            f"or the original old inode (found {stable_identity})"
        )

    if candidate_identity is None and backup_identity == old_identity:
        _rename_slot_noreplace(renamer, backup, candidate)
        _fsync_parent_directories(candidate, backup)
    elif candidate_identity == old_identity and backup_identity not in {
        old_identity,
        new_identity,
    }:
        # A non-cooperating process may have occupied the previously empty
        # backup path after preflight.  Preserve that unrelated directory and
        # still restore the two release slots; RENAME_NOREPLACE ensured it was
        # never overwritten by this process.
        pass
    else:
        raise RollbackError(
            "cannot roll back: the old stable inode is not uniquely located at candidate or backup; "
            f"candidate={candidate_identity}, backup={backup_identity}"
        )

    _assert_identity(candidate, old_identity)
    _rename_exchange(renamer, stable, candidate)
    _fsync_parent_directories(stable, candidate)
    _verify_original_primary_state(
        stable,
        candidate,
        old_identity=old_identity,
        new_identity=new_identity,
        old_hashes=old_hashes,
        new_hashes=new_hashes,
        old_snapshots=old_snapshots,
        new_snapshots=new_snapshots,
    )


def promote_dataset(
    *,
    stable_root: Path | str,
    candidate_root: Path | str,
    backup_root: Path | str,
    expected_new_manifest_sha256: str,
    expected_new_checksums_sha256: str,
    expected_old_manifest_sha256: str,
    expected_old_checksums_sha256: str,
    renamer: RenameAt2 | None = None,
) -> dict[str, object]:
    """Promote ``candidate_root`` and preserve the previous stable tree.

    Paths must be absolute, normalized, distinct, non-nested, and symlink-free.
    The candidate path is absent after success; its former contents become the
    new stable tree and the former stable contents are at ``backup_root``.
    """

    stable_path = _validate_absolute_path(stable_root, label="stable root")
    candidate_path = _validate_absolute_path(candidate_root, label="candidate root")
    backup_path = _validate_absolute_path(backup_root, label="backup root")
    old_hashes = ReleaseHashes(
        manifest_sha256=_normalize_digest(
            expected_old_manifest_sha256,
            label="expected old manifest SHA-256",
        ),
        checksums_sha256=_normalize_digest(
            expected_old_checksums_sha256,
            label="expected old CHECKSUMS SHA-256",
        ),
    )
    new_hashes = ReleaseHashes(
        manifest_sha256=_normalize_digest(
            expected_new_manifest_sha256,
            label="expected new manifest SHA-256",
        ),
        checksums_sha256=_normalize_digest(
            expected_new_checksums_sha256,
            label="expected new CHECKSUMS SHA-256",
        ),
    )

    stable = _open_slot(stable_path, label="stable root")
    candidate: Slot | None = None
    backup: Slot | None = None
    try:
        candidate = _open_slot(candidate_path, label="candidate root")
        backup = _open_slot(backup_path, label="backup root")
        _assert_distinct_non_nested_paths(stable, candidate, backup)

        old_identity = _directory_identity(stable)
        new_identity = _directory_identity(candidate)
        if (
            old_identity is None or new_identity is None
        ):  # pragma: no cover - required=True raises
            raise PreconditionsError(
                "stable and candidate release roots must both exist"
            )
        if old_identity == new_identity:
            raise PreconditionsError(
                "stable and candidate unexpectedly refer to the same directory inode"
            )
        if _entry_stat(backup) is not None:
            raise PreconditionsError(
                f"backup root already exists and will not be replaced: {backup.path}"
            )
        _assert_same_filesystem(
            stable,
            candidate,
            backup,
            old_identity=old_identity,
            new_identity=new_identity,
        )

        _assert_release_hashes(stable, old_hashes)
        _assert_release_hashes(candidate, new_hashes)
        old_snapshots = _verify_and_fsync_authoritative_release(stable, old_hashes)
        new_snapshots = _verify_and_fsync_authoritative_release(candidate, new_hashes)
        _fsync_parent_directories(stable, candidate, backup)

        active_renamer = renamer if renamer is not None else RenameAt2()
        _probe_renameat2_capabilities(
            active_renamer,
            stable.parent_fd,
            candidate.parent_fd,
            backup.parent_fd,
        )

        # Re-check all mutable preconditions after the capability probe and as
        # close as possible to the first real namespace mutation.
        _assert_identity(stable, old_identity)
        _assert_identity(candidate, new_identity)
        _assert_identity(backup, None)
        _assert_release_hashes(stable, old_hashes)
        _assert_release_hashes(candidate, new_hashes)
        _assert_authoritative_snapshot(stable, old_hashes, old_snapshots)
        _assert_authoritative_snapshot(candidate, new_hashes, new_snapshots)

        try:
            _rename_exchange(active_renamer, stable, candidate)
            _fsync_parent_directories(stable, candidate)
            _verify_exchange_state(
                stable,
                candidate,
                old_identity=old_identity,
                new_identity=new_identity,
                old_hashes=old_hashes,
                new_hashes=new_hashes,
                old_snapshots=old_snapshots,
                new_snapshots=new_snapshots,
            )

            _rename_slot_noreplace(active_renamer, candidate, backup)
            _fsync_parent_directories(candidate, backup)
            _verify_final_state(
                stable,
                candidate,
                backup,
                old_identity=old_identity,
                new_identity=new_identity,
                old_hashes=old_hashes,
                new_hashes=new_hashes,
                old_snapshots=old_snapshots,
                new_snapshots=new_snapshots,
            )
        except BaseException as promotion_failure:
            try:
                _restore_original_layout(
                    active_renamer,
                    stable,
                    candidate,
                    backup,
                    old_identity=old_identity,
                    new_identity=new_identity,
                    old_hashes=old_hashes,
                    new_hashes=new_hashes,
                    old_snapshots=old_snapshots,
                    new_snapshots=new_snapshots,
                )
                _fsync_parent_directories(stable, candidate, backup)
            except BaseException as rollback_failure:
                raise RollbackError(
                    "promotion failed and rollback could not prove the original layout; "
                    f"promotion error: {promotion_failure}; rollback error: {rollback_failure}"
                ) from rollback_failure
            if isinstance(promotion_failure, (KeyboardInterrupt, SystemExit)):
                raise
            raise PromotionError(
                "promotion failed; the original stable/candidate layout was restored: "
                f"{promotion_failure}"
            ) from promotion_failure

        return {
            "schema_version": 1,
            "status": "promoted",
            "stable_root": str(stable.path),
            "candidate_root": str(candidate.path),
            "candidate_exists_after_promotion": False,
            "backup_root": str(backup.path),
            "new_manifest_sha256": new_hashes.manifest_sha256,
            "new_checksums_sha256": new_hashes.checksums_sha256,
            "old_manifest_sha256": old_hashes.manifest_sha256,
            "old_checksums_sha256": old_hashes.checksums_sha256,
            "atomic_operations": ["RENAME_EXCHANGE", "RENAME_NOREPLACE"],
        }
    finally:
        stable.close()
        if candidate is not None:
            candidate.close()
        if backup is not None:
            backup.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Atomically exchange an identity-paired candidate with the stable dataset, then "
            "move the old stable tree to a non-existing backup path. No rename fallback is used."
        )
    )
    parser.add_argument("--stable-root", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--backup-root", required=True, type=Path)
    parser.add_argument(
        "--expected-new-manifest-sha256", required=True, type=_argparse_digest
    )
    parser.add_argument(
        "--expected-new-checksums-sha256", required=True, type=_argparse_digest
    )
    parser.add_argument(
        "--expected-old-manifest-sha256", required=True, type=_argparse_digest
    )
    parser.add_argument(
        "--expected-old-checksums-sha256", required=True, type=_argparse_digest
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        report = promote_dataset(
            stable_root=arguments.stable_root,
            candidate_root=arguments.candidate_root,
            backup_root=arguments.backup_root,
            expected_new_manifest_sha256=arguments.expected_new_manifest_sha256,
            expected_new_checksums_sha256=arguments.expected_new_checksums_sha256,
            expected_old_manifest_sha256=arguments.expected_old_manifest_sha256,
            expected_old_checksums_sha256=arguments.expected_old_checksums_sha256,
        )
    except (PromotionError, OSError) as exc:
        print(f"promotion refused or failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
