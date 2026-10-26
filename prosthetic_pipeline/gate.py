"""Universal validity gate (SPEC-2 §3, E-4): applied by the ENGINE to every
candidate of every topology, so feasibility cannot be skipped by a topology
class. Table-level checks (V3, V5-V7) are topology-agnostic; geometric checks
(V8, V9) use the optional `pose_points` protocol hook where a topology can
report its link segments and body points.

Returns hard pass/fail plus soft margins (§3.2) so rejected candidates carry a
feasibility gradient instead of a cliff.
"""

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from .mechanism import (DEAD_LENGTH_PARALLEL, MIN_MOMENT_ARM, MIN_RETRACTED,
                        STROKE_MARGIN, STROKE_RATIO_LIMIT)

HOLD_MARGIN = 1.3
ENVELOPE_FACTOR = 1.15
N_GEOM_SAMPLES = 16      # theta samples for V8/V9 geometry checks


@dataclass
class GateResult:
    ok: bool
    soft: float                      # feasibility-gradient penalty (0 when ok)
    reasons: list = field(default_factory=list)
    margins: Dict = field(default_factory=dict)


def _relu(x):
    return max(0.0, float(x))


def _segments_cross(p1, p2, p3, p4, eps=1e-9):
    """Proper intersection of segments p1p2 and p3p4 (shared endpoints ignored)."""
    d1, d2 = np.asarray(p2) - p1, np.asarray(p4) - p3
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-12:
        return False
    b = np.asarray(p3) - p1
    t = (b[0] * d2[1] - b[1] * d2[0]) / den
    s = (b[0] * d1[1] - b[1] * d1[0]) / den
    return (eps < t < 1 - eps) and (eps < s < 1 - eps)


def apply_gate(table, genome, topo, ctx, th_lo, th_hi, F, v) -> GateResult:
    in_rom = (table.theta >= th_lo - 1e-12) & (table.theta <= th_hi + 1e-12)
    reasons, margins = [], {}
    soft = 0.0

    # V1/V4: builder-reported assembly validity over the required ROM
    coverage = float(np.mean(table.valid[in_rom]))
    if coverage < 1.0:
        reasons.append(f"assembly/branch invalid over {100*(1-coverage):.0f}% of ROM")
        soft += 5.0 * (1.0 - coverage)

    rom_ok = table.valid & in_rom
    use = rom_ok if rom_ok.any() else in_rom
    for i, kind in enumerate(table.actuator_kinds):
        r_i = np.abs(table.r[i][use])
        q_i = table.q[i][use]
        tag = f"a{i}"
        if kind in ("linear", "tendon"):
            if F[i] <= 1.0:
                continue                       # dormant secondary actuator
            # V3 - minimum transmission ratio (replaces per-dyad angle checks)
            r_min = float(r_i.min())
            margins[f"{tag}_r_min_mm"] = r_min * 1000
            if r_min < MIN_MOMENT_ARM:
                reasons.append(f"{tag}: min|r| {r_min*1000:.1f} mm < {MIN_MOMENT_ARM*1000:.0f} mm")
                soft += 2.0 * _relu(MIN_MOMENT_ARM - r_min) / MIN_MOMENT_ARM
            # V5 - stroke ratio, with the legacy end margins (mechanism.fit_stroke)
            q_lo = float(q_i.min()) - STROKE_MARGIN
            q_hi = float(q_i.max()) + STROKE_MARGIN
            ratio = q_hi / max(q_lo, 1e-9)
            margins[f"{tag}_stroke_ratio"] = ratio
            if ratio > STROKE_RATIO_LIMIT:
                reasons.append(f"{tag}: stroke ratio {ratio:.2f} > {STROKE_RATIO_LIMIT}")
                soft += 2.0 * _relu(ratio - STROKE_RATIO_LIMIT)
            # V5b - degenerate stroke (legacy tiny-stroke penalty, restored)
            stroke = q_hi - q_lo
            if stroke < 0.008:
                reasons.append(f"{tag}: stroke {stroke*1000:.1f} mm is degenerate")
                soft += 2.0 * _relu(0.008 - stroke) / 0.008
            # V6 - retracted length / packaging
            need = max(MIN_RETRACTED, stroke + DEAD_LENGTH_PARALLEL)
            margins[f"{tag}_packaging_mm"] = (q_lo - need) * 1000
            if q_lo < need:
                reasons.append(f"{tag}: retracted {q_lo*1000:.0f} mm < {need*1000:.0f} mm")
                soft += 2.0 * _relu(need - q_lo) / DEAD_LENGTH_PARALLEL

    # V7 - static holdability with margin, from total capacity vs G+S load
    load = np.abs(table.G[use] + table.S[use])
    cap = np.zeros_like(load)
    for i in range(table.n_act):
        cap += np.abs(table.r[i][use]) * F[i]
    deficit = HOLD_MARGIN * load - cap
    worst = float(deficit.max()) if len(deficit) else 0.0
    margins["hold_deficit_Nm"] = worst
    if worst > 0:
        reasons.append(f"cannot hold statically ({worst:.1f} N m short at {HOLD_MARGIN}x)")
        soft += 1.0 * worst / max(float(np.max(HOLD_MARGIN * load)), 1e-6)

    # V8/V9 - geometry: link/actuator/bone crossings and the limb envelope.
    # The interpreter records sampled world segments into EVERY table
    # (table.geometry), so this check cannot be skipped by a topology class
    # (REVIEW-3 H2: no more hasattr opt-out).
    geom = getattr(table, "geometry", None)
    if geom is not None:
        env = ENVELOPE_FACTOR * max(ctx.Lp, ctx.Ld)
        gth = geom["theta"]
        in_g = (gth >= th_lo - 1e-9) & (gth <= th_hi + 1e-9)
        for si in np.flatnonzero(in_g):
            segs = geom["segs"][si]          # (S, 2, 2)
            if np.linalg.norm(segs.reshape(-1, 2), axis=1).max() > env:
                reasons.append(
                    f"body outside limb envelope at {np.degrees(gth[si]):.0f} deg")
                soft += 1.0
                break
            hit = False
            for a in range(len(segs)):
                for b in range(a + 1, len(segs)):
                    if _segments_cross(segs[a][0], segs[a][1],
                                       segs[b][0], segs[b][1]):
                        reasons.append(
                            f"links/actuator cross the limb at "
                            f"{np.degrees(gth[si]):.0f} deg")
                        soft += 1.0
                        hit = True
                        break
                if hit:
                    break
            if hit:
                break

    return GateResult(ok=not reasons, soft=soft, reasons=reasons, margins=margins)
