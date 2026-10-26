"""Custom-drive synthesis: the catalogue never hard-fails a run again.

The evolved class space (F 60-1500 N x v 20-350 mm/s) previously contained 35
cells with "no feasible screw/motor combination"; synthesis must cover every
cell, catalogue parts must stay preferred when they fit, and synthesized
drives must respect the physical caps they claim.
"""
import numpy as np

from prosthetic_pipeline.actuator_design import (
    CUSTOM_G_MAX, CUSTOM_N_BALL_RPM, MOTOR_D_MAX, N_MOTOR_MAX_RPM,
    _motor_diameter_mm, _pick_drive, _synthesize_drive)


def test_every_evolvable_class_cell_gets_a_drive():
    for F in np.geomspace(60, 1500, 10):
        for v in np.geomspace(0.02, 0.35, 10):
            Fd, vd, Frms = 1.5 * F, 1.25 * v, 0.75 * F
            drives = [(_pick_drive(k, Fd, Frms, vd)
                       or _synthesize_drive(k, Fd, Frms, vd))
                      for k in ("leadscrew", "ballscrew")]
            assert any(drives), f"no drive at F={F:.0f} N v={v*1000:.0f} mm/s"


def test_synthesis_respects_physical_caps():
    d = _synthesize_drive("ballscrew", 2250.0, 1125.0, 0.44)
    assert d is not None
    assert d.n_screw_rpm <= CUSTOM_N_BALL_RPM + 1e-6
    assert d.n_motor_rpm <= N_MOTOR_MAX_RPM + 1e-6
    assert d.gear_ratio <= CUSTOM_G_MAX + 1e-6
    assert _motor_diameter_mm(d.motor_class) <= MOTOR_D_MAX
    assert "custom" in d.motor_class


def test_catalogue_preferred_when_it_fits():
    d = _pick_drive("ballscrew", 300.0, 150.0, 0.08)
    assert d is not None and "custom" not in d.motor_class


def test_motor_diameter_parses_both_label_kinds():
    assert _motor_diameter_mm("28-32 mm BLDC class") == 28.0
    assert _motor_diameter_mm("47 mm BLDC (custom wind)") == 47.0
    assert _motor_diameter_mm(None) == 30.0
