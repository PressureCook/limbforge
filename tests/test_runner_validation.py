"""Movement-verification tests for review finding H1a."""

from types import SimpleNamespace

import pytest

from prosthetic_pipeline.runner import RunConfig, _validate_movement_rom


def _series(lo, hi):
    return SimpleNamespace(rom_deg=(lo, hi))


def _cfg(movement):
    return RunConfig(movement=movement, demo=False)


def test_missing_required_joint_fails():
    """A knee task checked against an arm recording must NOT pass vacuously."""
    with pytest.raises(RuntimeError, match="requires the knee"):
        _validate_movement_rom(_cfg("knee_bends"), {"elbow": _series(5, 120)},
                               tee=lambda *a: None)


def test_insufficient_rom_fails_with_specific_advice():
    with pytest.raises(RuntimeError, match="recorded range 18 deg"):
        _validate_movement_rom(_cfg("knee_bends"), {"knee": _series(10, 28)},
                               tee=lambda *a: None)


def test_limb_task_requires_all_joints():
    with pytest.raises(RuntimeError, match="hip"):
        _validate_movement_rom(_cfg("walk_in_place"),
                               {"knee": _series(0, 60), "ankle": _series(80, 100)},
                               tee=lambda *a: None)


def test_valid_recording_passes():
    logs = []
    _validate_movement_rom(_cfg("knee_bends"), {"knee": _series(5, 85)},
                           tee=logs.append)
    assert any("requirements met" in l for l in logs)


def test_demo_and_no_movement_skip():
    _validate_movement_rom(RunConfig(demo=True, movement="knee_bends"), {}, tee=None)
    _validate_movement_rom(RunConfig(demo=False, movement=None), {}, tee=None)
