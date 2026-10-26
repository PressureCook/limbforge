"""Capture-stage regression tests for review findings C4 and C5."""

import os

import cv2
import numpy as np
import pytest

from prosthetic_pipeline import capture


def _write_photos(tmp_path, n, size=(64, 48)):
    d = tmp_path / "photos"
    d.mkdir()
    for i in range(n):
        img = np.full((size[1], size[0], 3), i % 255, dtype=np.uint8)
        cv2.imwrite(str(d / f"img_{i:04d}.jpg"), img)
    return str(d)


def test_photo_timestamps_survive_subsampling(tmp_path, monkeypatch):
    """C4: 30 photos at 4/s span 7.25 s of real time; subsampling to 10 frames
    must NOT compress that to 2.25 s."""
    monkeypatch.setattr(capture, "MAX_FRAMES", 10)
    frames, meta = capture.load_frames(_write_photos(tmp_path, 30), photo_fps=4.0)
    assert len(frames) == 10
    ts = [t for t, _ in frames]
    assert ts[0] == 0.0
    assert ts[-1] == pytest.approx(27 / 4.0)          # original index 27, not 9
    assert ts[1] - ts[0] == pytest.approx(3 / 4.0)    # stride 3 spacing


def test_photo_timestamps_without_subsampling(tmp_path):
    frames, _ = capture.load_frames(_write_photos(tmp_path, 8), photo_fps=2.0)
    assert [t for t, _ in frames] == pytest.approx([i / 2.0 for i in range(8)])


def _write_video(tmp_path, n_frames, fps=30.0, size=(64, 48)):
    path = str(tmp_path / "clip.mp4")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert vw.isOpened()
    for i in range(n_frames):
        vw.write(np.full((size[1], size[0], 3), i % 255, dtype=np.uint8))
    vw.release()
    return path


def test_video_respects_max_frames(tmp_path, monkeypatch):
    monkeypatch.setattr(capture, "MAX_FRAMES", 30)
    frames, meta = capture.load_frames(_write_video(tmp_path, 90), photo_fps=4.0)
    assert len(frames) <= 30
    # stride 3 at 30 fps -> timestamps keep the true clip duration
    assert frames[-1][0] == pytest.approx(87 / 30.0, abs=0.2)


def test_video_capped_even_if_frame_count_unreadable(tmp_path, monkeypatch):
    """C5: when the container reports no frame count, the in-loop cap must hold."""
    monkeypatch.setattr(capture, "MAX_FRAMES", 25)
    path = _write_video(tmp_path, 120)

    real_capture = cv2.VideoCapture

    class NoCountCapture:
        def __init__(self, p):
            self._c = real_capture(p)

        def isOpened(self):
            return self._c.isOpened()

        def get(self, prop):
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 0.0  # decoder can't report length (webm/VFR case)
            return self._c.get(prop)

        def read(self):
            return self._c.read()

        def release(self):
            self._c.release()

    monkeypatch.setattr(capture.cv2, "VideoCapture", NoCountCapture)
    frames, meta = capture.load_frames(path, photo_fps=4.0)
    assert len(frames) == 25
    assert any("kept the first" in n for n in meta["notes"])


def test_frames_are_downscaled(tmp_path):
    d = tmp_path / "big"
    d.mkdir()
    cv2.imwrite(str(d / "a.jpg"), np.zeros((1500, 2400, 3), dtype=np.uint8))
    for i in range(3):
        cv2.imwrite(str(d / f"b{i}.jpg"), np.zeros((1500, 2400, 3), dtype=np.uint8))
    frames, meta = capture.load_frames(str(d), photo_fps=2.0)
    h, w = frames[0][1].shape[:2]
    assert max(h, w) <= capture.WORK_LONG_EDGE
    assert w / h == pytest.approx(2400 / 1500, rel=0.01)
    assert any("downscaled" in n for n in meta["notes"])
