"""Integration test of the pose-extraction stage with a stubbed detector.

Exercises the full extract_pose logic (detection loop, visibility gating, gap
interpolation, run trimming, auto joint selection, angle computation, overlay
drawing) without needing a human photo or the model file:
`python -m prosthetic_pipeline.selftest_pose`
"""

import os
import tempfile
from dataclasses import dataclass

import numpy as np

from . import pose as pose_mod
from .demo_data import make_demo_series


@dataclass
class _LM:
    x: float
    y: float
    z: float
    visibility: float


class _StubLandmarker:
    """Replays the demo stick-figure as if MediaPipe had detected it."""

    def __init__(self, series):
        self.series = series
        self.i = 0

    def _result(self, i):
        class R:
            pass

        r = R()
        lm = self.series.norm_landmarks[i]
        if i < 3 or i in (20, 21):  # undetected head/tail + a mid gap
            r.pose_landmarks, r.pose_world_landmarks = [], []
            return r
        norm, world = [], []
        for k in range(33):
            x, y, v = lm[k]
            if np.isnan(x):
                x, y, v = 0.5, 0.5, 0.0
            if i in (30, 31) :
                v = 0.1  # low-visibility gap to exercise interpolation
            norm.append(_LM(x, y, 0.0, v))
            world.append(_LM((x - 0.45) * 1.1, (y - 0.5) * 1.1, 0.0, v))
        r.pose_landmarks = [norm]
        r.pose_world_landmarks = [world]
        return r

    def detect(self, image):
        res = self._result(self.i)
        self.i += 1
        return res

    def detect_for_video(self, image, ts):
        return self.detect(image)

    def close(self):
        pass


def main():
    series = make_demo_series(seed=3)
    n = len(series.t)
    frames = [(series.t[i], np.zeros((480, 640, 3), dtype=np.uint8)) for i in range(n)]

    stub = _StubLandmarker(series)
    orig = pose_mod._make_landmarker
    pose_mod._make_landmarker = lambda mp_path, vm: (
        __import__("mediapipe"), stub)
    try:
        with tempfile.TemporaryDirectory() as out:
            got = pose_mod.extract_pose(frames, {"kind": "photos", "notes": []},
                                        joint="auto", side="auto", out_dir=out)
            assert got.joint == "elbow" and got.side == "right", \
                f"auto-select picked {got.side} {got.joint}"
            assert len(got.t) >= n - 8, f"kept only {len(got.t)}/{n} frames"
            # angles recovered from landmarks should match the generating curve
            gen = series.theta_deg[-len(got.t):]
            err = np.abs(got.theta_deg - gen)
            assert np.median(err) < 4.0, f"angle recovery off by {np.median(err):.1f} deg"
            assert 0.05 < got.Ld < 0.6, f"distal length implausible: {got.Ld}"
            assert len(got.overlay_paths) == 3
            assert all(os.path.exists(p) for p in got.overlay_paths)
            print(f"pose extraction: {got.side} {got.joint}, {len(got.t)} frames, "
                  f"median angle error {np.median(err):.2f} deg, Lp={got.Lp:.2f} m, Ld={got.Ld:.2f} m")
    finally:
        pose_mod._make_landmarker = orig
    print("POSE PIPELINE SELF-TEST PASSED")


if __name__ == "__main__":
    main()
