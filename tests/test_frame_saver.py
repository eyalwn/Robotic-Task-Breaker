"""Hermetic tests for FrameSaver."""
from __future__ import annotations

from sim.frame_saver import FrameSaver

FAKE_JPEG = b"\xff\xd8fake-jpeg-data"


def test_latest_always_written(tmp_path):
    saver = FrameSaver(str(tmp_path), save_every_n=5)
    for i in range(3):
        saver.save(FAKE_JPEG, loop_index=i + 1)
    assert (tmp_path / "latest.jpg").read_bytes() == FAKE_JPEG


def test_numbered_frames_follow_cadence(tmp_path):
    saver = FrameSaver(str(tmp_path), save_every_n=3)
    results = [saver.save(FAKE_JPEG, loop_index=i) for i in range(10)]

    for i in range(10):
        if i % 3 == 0:
            assert results[i] is not None
            assert (tmp_path / f"frame_{i:04d}.jpg").exists()
        else:
            assert results[i] is None
            assert not (tmp_path / f"frame_{i:04d}.jpg").exists()


def test_save_success_writes_file(tmp_path):
    saver = FrameSaver(str(tmp_path))
    path = saver.save_success(FAKE_JPEG)
    assert path.endswith("pickcube_success.jpg")
    assert (tmp_path / "pickcube_success.jpg").read_bytes() == FAKE_JPEG


def test_creates_output_dir(tmp_path):
    nested = tmp_path / "a" / "b"
    saver = FrameSaver(str(nested))
    saver.save(FAKE_JPEG, loop_index=0)
    assert (nested / "latest.jpg").exists()
