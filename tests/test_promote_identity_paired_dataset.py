import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import scripts.promote_identity_paired_dataset as promotion


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_release(root: Path, marker: str) -> promotion.ReleaseHashes:
    manifest = root / promotion.MANIFEST_RELATIVE_PATH
    manifest.parent.mkdir(parents=True)
    manifest.write_text(f'{{"release":"{marker}"}}\n', encoding="utf-8")
    checksums = root / promotion.CHECKSUMS_RELATIVE_PATH
    checksums.write_text(
        f"{_sha256(manifest)}  {promotion.MANIFEST_RELATIVE_PATH.as_posix()}\n",
        encoding="utf-8",
    )
    return promotion.ReleaseHashes(
        manifest_sha256=_sha256(manifest),
        checksums_sha256=_sha256(checksums),
    )


class PromoteIdentityPairedDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.stable = self.root / "stable"
        self.candidate = self.root / "candidate"
        self.backup = self.root / "old-backup"
        self.old_hashes = _write_release(self.stable, "old")
        self.new_hashes = _write_release(self.candidate, "new")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _promote(self) -> dict[str, object]:
        return promotion.promote_dataset(
            stable_root=self.stable,
            candidate_root=self.candidate,
            backup_root=self.backup,
            expected_new_manifest_sha256=self.new_hashes.manifest_sha256,
            expected_new_checksums_sha256=self.new_hashes.checksums_sha256,
            expected_old_manifest_sha256=self.old_hashes.manifest_sha256,
            expected_old_checksums_sha256=self.old_hashes.checksums_sha256,
        )

    def _assert_original_layout(self) -> None:
        self.assertEqual(
            _sha256(self.stable / promotion.MANIFEST_RELATIVE_PATH),
            self.old_hashes.manifest_sha256,
        )
        self.assertEqual(
            _sha256(self.candidate / promotion.MANIFEST_RELATIVE_PATH),
            self.new_hashes.manifest_sha256,
        )
        self.assertFalse(self.backup.exists())

    def test_success_uses_exchange_then_preserves_old_release_at_backup(self) -> None:
        report = self._promote()

        self.assertEqual(report["status"], "promoted")
        self.assertEqual(
            report["atomic_operations"], ["RENAME_EXCHANGE", "RENAME_NOREPLACE"]
        )
        self.assertFalse(self.candidate.exists())
        self.assertEqual(
            _sha256(self.stable / promotion.MANIFEST_RELATIVE_PATH),
            self.new_hashes.manifest_sha256,
        )
        self.assertEqual(
            _sha256(self.backup / promotion.MANIFEST_RELATIVE_PATH),
            self.old_hashes.manifest_sha256,
        )

    def test_wrong_expected_hash_refuses_before_any_namespace_mutation(self) -> None:
        with self.assertRaisesRegex(
            promotion.PreconditionsError, "release hash mismatch"
        ):
            promotion.promote_dataset(
                stable_root=self.stable,
                candidate_root=self.candidate,
                backup_root=self.backup,
                expected_new_manifest_sha256="0" * 64,
                expected_new_checksums_sha256=self.new_hashes.checksums_sha256,
                expected_old_manifest_sha256=self.old_hashes.manifest_sha256,
                expected_old_checksums_sha256=self.old_hashes.checksums_sha256,
            )

        self._assert_original_layout()

    def test_authoritative_payload_checksum_is_verified_before_promotion(self) -> None:
        payload = self.candidate / "object" / "tar_image" / "case.png"
        payload.parent.mkdir(parents=True)
        payload.write_bytes(b"expected payload")
        manifest = self.candidate / promotion.MANIFEST_RELATIVE_PATH
        checksums = self.candidate / promotion.CHECKSUMS_RELATIVE_PATH
        checksums.write_text(
            f"{_sha256(manifest)}  {promotion.MANIFEST_RELATIVE_PATH.as_posix()}\n"
            f"{_sha256(payload)}  object/tar_image/case.png\n",
            encoding="utf-8",
        )
        self.new_hashes = promotion.ReleaseHashes(
            manifest_sha256=_sha256(manifest),
            checksums_sha256=_sha256(checksums),
        )
        payload.write_bytes(b"tampered payload")

        with self.assertRaisesRegex(promotion.PreconditionsError, "checksum mismatch"):
            self._promote()

        self._assert_original_layout()

    def test_exact_checksum_bytes_parsed_are_bound_to_expected_digest(self) -> None:
        payload = self.candidate / "object" / "src_image" / "case.png"
        payload.parent.mkdir(parents=True)
        payload.write_bytes(b"payload")
        manifest = self.candidate / promotion.MANIFEST_RELATIVE_PATH
        checksums = self.candidate / promotion.CHECKSUMS_RELATIVE_PATH
        checksums.write_text(
            f"{_sha256(manifest)}  {promotion.MANIFEST_RELATIVE_PATH.as_posix()}\n"
            f"{_sha256(payload)}  object/src_image/case.png\n",
            encoding="utf-8",
        )
        expected_checksums = _sha256(checksums)
        malicious_short_list = (
            f"{_sha256(manifest)}  {promotion.MANIFEST_RELATIVE_PATH.as_posix()}\n"
        ).encode("utf-8")
        slot = promotion._open_slot(self.candidate, label="candidate root")
        try:
            with patch.object(
                promotion,
                "_read_fd",
                return_value=malicious_short_list,
            ):
                with self.assertRaisesRegex(
                    promotion.PreconditionsError,
                    "parsed checksum manifest SHA-256 mismatch",
                ):
                    promotion._authoritative_checksum_rows(
                        slot,
                        expected_manifest_sha256=_sha256(manifest),
                        expected_checksums_sha256=expected_checksums,
                    )
        finally:
            slot.close()

    def test_payload_change_during_capability_probe_is_refused_before_exchange(
        self,
    ) -> None:
        payload = self.candidate / "object" / "src_mask" / "case.png"
        payload.parent.mkdir(parents=True)
        payload.write_bytes(b"verified payload")
        manifest = self.candidate / promotion.MANIFEST_RELATIVE_PATH
        checksums = self.candidate / promotion.CHECKSUMS_RELATIVE_PATH
        checksums.write_text(
            f"{_sha256(manifest)}  {promotion.MANIFEST_RELATIVE_PATH.as_posix()}\n"
            f"{_sha256(payload)}  object/src_mask/case.png\n",
            encoding="utf-8",
        )
        self.new_hashes = promotion.ReleaseHashes(
            manifest_sha256=_sha256(manifest),
            checksums_sha256=_sha256(checksums),
        )
        original_probe = promotion._probe_renameat2_capabilities

        def probe_then_mutate(*args, **kwargs) -> None:
            original_probe(*args, **kwargs)
            payload.write_bytes(b"changed during probe")

        with patch.object(
            promotion,
            "_probe_renameat2_capabilities",
            side_effect=probe_then_mutate,
        ):
            with self.assertRaisesRegex(
                promotion.PreconditionsError,
                "authoritative file changed after full verification",
            ):
                self._promote()

        self._assert_original_layout()

    def test_existing_backup_is_never_replaced(self) -> None:
        self.backup.mkdir()
        sentinel = self.backup / "do-not-overwrite"
        sentinel.write_text("preserve", encoding="utf-8")

        with self.assertRaisesRegex(promotion.PreconditionsError, "already exists"):
            self._promote()

        self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")
        self.assertTrue(self.stable.exists())
        self.assertTrue(self.candidate.exists())

    def test_symlink_release_root_is_refused(self) -> None:
        real_candidate = self.candidate
        linked_candidate = self.root / "candidate-link"
        linked_candidate.symlink_to(real_candidate, target_is_directory=True)

        with self.assertRaisesRegex(promotion.PreconditionsError, "symbolic link"):
            promotion.promote_dataset(
                stable_root=self.stable,
                candidate_root=linked_candidate,
                backup_root=self.backup,
                expected_new_manifest_sha256=self.new_hashes.manifest_sha256,
                expected_new_checksums_sha256=self.new_hashes.checksums_sha256,
                expected_old_manifest_sha256=self.old_hashes.manifest_sha256,
                expected_old_checksums_sha256=self.old_hashes.checksums_sha256,
            )

        self._assert_original_layout()

    def test_symlink_inside_release_marker_path_is_refused(self) -> None:
        outside = self.root / "outside-manifests"
        outside.mkdir()
        (outside / "paperinsert_900.jsonl").write_text("external\n", encoding="utf-8")
        manifest_directory = self.candidate / "manifests"
        original_manifest = manifest_directory / "paperinsert_900.jsonl"
        original_manifest.unlink()
        manifest_directory.rmdir()
        manifest_directory.symlink_to(outside, target_is_directory=True)

        with self.assertRaisesRegex(promotion.PreconditionsError, "cannot safely open"):
            self._promote()

        self.assertEqual(
            _sha256(self.stable / promotion.MANIFEST_RELATIVE_PATH),
            self.old_hashes.manifest_sha256,
        )
        self.assertTrue(self.candidate.is_dir())
        self.assertFalse(self.backup.exists())

    def test_capability_probe_failure_has_no_rename_fallback(self) -> None:
        with patch.object(
            promotion,
            "_probe_renameat2_capabilities",
            side_effect=promotion.PreconditionsError("probe intentionally failed"),
        ):
            with self.assertRaisesRegex(
                promotion.PreconditionsError, "probe intentionally failed"
            ):
                self._promote()

        self._assert_original_layout()

    def test_probe_collision_never_unlinks_preexisting_entry(self) -> None:
        fixed_hex = "0" * 32
        probe_name = (
            f".identity-promotion-renameat2-probe-{promotion.os.getpid()}-"
            f"{fixed_hex}-exchange-left"
        )
        preexisting = self.root / probe_name
        preexisting.write_text("not owned by probe", encoding="utf-8")
        original_inode = preexisting.stat().st_ino
        parent_fd = promotion.os.open(self.root, promotion._DIRECTORY_OPEN_FLAGS)
        fake_uuid = type("FixedUuid", (), {"hex": fixed_hex})()
        try:
            with patch.object(promotion.uuid, "uuid4", return_value=fake_uuid):
                with self.assertRaisesRegex(
                    promotion.PreconditionsError,
                    "required renameat2 operations",
                ):
                    promotion._probe_renameat2_capabilities(
                        promotion.RenameAt2(),
                        parent_fd,
                        parent_fd,
                        parent_fd,
                    )
        finally:
            promotion.os.close(parent_fd)

        self.assertEqual(preexisting.stat().st_ino, original_inode)
        self.assertEqual(preexisting.read_text(encoding="utf-8"), "not owned by probe")

    def test_probe_interrupt_after_exchange_cleans_all_created_inodes(self) -> None:
        fixed_hex = "1" * 32
        prefix = (
            f".identity-promotion-renameat2-probe-{promotion.os.getpid()}-{fixed_hex}"
        )
        fake_uuid = type("FixedUuid", (), {"hex": fixed_hex})()
        parent_fd = promotion.os.open(self.root, promotion._DIRECTORY_OPEN_FLAGS)

        class InterruptAfterSuccessfulExchange:
            def __init__(self) -> None:
                self.delegate = promotion.RenameAt2()

            def rename(self, *args) -> None:
                self.delegate.rename(*args)
                if args[-1] == promotion.RENAME_EXCHANGE:
                    raise KeyboardInterrupt("injected immediately after exchange")

        try:
            with patch.object(promotion.uuid, "uuid4", return_value=fake_uuid):
                with self.assertRaisesRegex(
                    KeyboardInterrupt, "immediately after exchange"
                ):
                    promotion._probe_renameat2_capabilities(
                        InterruptAfterSuccessfulExchange(),
                        parent_fd,
                        parent_fd,
                        parent_fd,
                    )
        finally:
            promotion.os.close(parent_fd)

        self.assertEqual(
            [path.name for path in self.root.iterdir() if path.name.startswith(prefix)],
            [],
        )

    def test_cross_parent_probe_exercises_both_required_flags(self) -> None:
        cross_root = self.root / "cross-parent"
        stable = cross_root / "stable-parent" / "stable"
        candidate = cross_root / "candidate-parent" / "candidate"
        backup = cross_root / "backup-parent" / "backup"
        old_hashes = _write_release(stable, "cross-old")
        new_hashes = _write_release(candidate, "cross-new")
        backup.parent.mkdir(parents=True)

        class RecordingRenamer:
            def __init__(self) -> None:
                self.delegate = promotion.RenameAt2()
                self.flags: list[int] = []

            def rename(self, *args) -> None:
                self.flags.append(args[-1])
                self.delegate.rename(*args)

        renamer = RecordingRenamer()
        promotion.promote_dataset(
            stable_root=stable,
            candidate_root=candidate,
            backup_root=backup,
            expected_new_manifest_sha256=new_hashes.manifest_sha256,
            expected_new_checksums_sha256=new_hashes.checksums_sha256,
            expected_old_manifest_sha256=old_hashes.manifest_sha256,
            expected_old_checksums_sha256=old_hashes.checksums_sha256,
            renamer=renamer,
        )

        self.assertEqual(renamer.flags.count(promotion.RENAME_EXCHANGE), 2)
        self.assertEqual(renamer.flags.count(promotion.RENAME_NOREPLACE), 3)
        self.assertFalse(candidate.exists())
        self.assertTrue(stable.is_dir())
        self.assertTrue(backup.is_dir())

    def test_failed_exchange_verification_rolls_back(self) -> None:
        with patch.object(
            promotion,
            "_verify_exchange_state",
            side_effect=promotion.PreconditionsError(
                "injected exchange verification failure"
            ),
        ):
            with self.assertRaisesRegex(
                promotion.PromotionError, "original.*layout was restored"
            ):
                self._promote()

        self._assert_original_layout()

    def test_noreplace_move_failure_rolls_back_exchange(self) -> None:
        with patch.object(
            promotion,
            "_rename_slot_noreplace",
            side_effect=OSError("injected no-replace move failure"),
        ):
            with self.assertRaisesRegex(
                promotion.PromotionError, "layout was restored"
            ):
                self._promote()

        self._assert_original_layout()

    def test_parent_fsync_failure_after_exchange_rolls_back(self) -> None:
        original_fsync_parents = promotion._fsync_parent_directories
        calls = 0

        def fail_second_call(*slots) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected parent fsync failure")
            original_fsync_parents(*slots)

        with patch.object(
            promotion,
            "_fsync_parent_directories",
            side_effect=fail_second_call,
        ):
            with self.assertRaisesRegex(
                promotion.PromotionError, "layout was restored"
            ):
                self._promote()

        self._assert_original_layout()

    def test_failed_final_verification_restores_backup_then_rolls_back_exchange(
        self,
    ) -> None:
        with patch.object(
            promotion,
            "_verify_final_state",
            side_effect=promotion.PreconditionsError(
                "injected final verification failure"
            ),
        ):
            with self.assertRaisesRegex(
                promotion.PromotionError, "original.*layout was restored"
            ):
                self._promote()

        self._assert_original_layout()

    def test_backup_race_is_preserved_and_primary_release_slots_roll_back(self) -> None:
        original_verify = promotion._verify_exchange_state
        sentinel = self.backup / "created-by-another-process"

        def verify_then_occupy_backup(*args, **kwargs) -> None:
            original_verify(*args, **kwargs)
            self.backup.mkdir()
            sentinel.write_text("preserve", encoding="utf-8")

        with patch.object(
            promotion,
            "_verify_exchange_state",
            side_effect=verify_then_occupy_backup,
        ):
            with self.assertRaisesRegex(
                promotion.PromotionError,
                "original stable/candidate layout was restored",
            ):
                self._promote()

        self.assertEqual(
            _sha256(self.stable / promotion.MANIFEST_RELATIVE_PATH),
            self.old_hashes.manifest_sha256,
        )
        self.assertEqual(
            _sha256(self.candidate / promotion.MANIFEST_RELATIVE_PATH),
            self.new_hashes.manifest_sha256,
        )
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")

    def test_relative_and_nested_paths_are_refused(self) -> None:
        with self.assertRaisesRegex(promotion.PreconditionsError, "absolute path"):
            promotion.promote_dataset(
                stable_root=Path("stable"),
                candidate_root=self.candidate,
                backup_root=self.backup,
                expected_new_manifest_sha256=self.new_hashes.manifest_sha256,
                expected_new_checksums_sha256=self.new_hashes.checksums_sha256,
                expected_old_manifest_sha256=self.old_hashes.manifest_sha256,
                expected_old_checksums_sha256=self.old_hashes.checksums_sha256,
            )

        nested_backup = self.candidate / "backup"
        with self.assertRaisesRegex(
            promotion.PreconditionsError, "must not contain one another"
        ):
            promotion.promote_dataset(
                stable_root=self.stable,
                candidate_root=self.candidate,
                backup_root=nested_backup,
                expected_new_manifest_sha256=self.new_hashes.manifest_sha256,
                expected_new_checksums_sha256=self.new_hashes.checksums_sha256,
                expected_old_manifest_sha256=self.old_hashes.manifest_sha256,
                expected_old_checksums_sha256=self.old_hashes.checksums_sha256,
            )

        self._assert_original_layout()


if __name__ == "__main__":
    unittest.main()
