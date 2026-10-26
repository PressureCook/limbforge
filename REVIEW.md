# Code review — prosthetic actuator-joint pipeline

Full review of the repository as of 2026-08-12 (~4,900 lines: 15 pipeline modules,
Flask app, 3 templates). Every claim below was verified by reading the code and,
where a number is quoted, by executing it. Findings are ranked by severity.

**Overall:** the engineering spine is real. The moment-arm derivation is correct and
unit-tested against finite differences, the screw-efficiency and self-locking
criteria are textbook, and the motor-sizing chain is dimensionally sound. The
disclaimers are honest and prominent. What follows are the places where the physics,
the timing, or the verification story does not actually hold up.

---

## Critical

### C1 — Gravity torque is sign-inverted for the shoulder, and meaningless for the ankle

`simulate.py:62-72`

```python
return -ctx.mass * G * ctx.com * np.sin(theta)
```

The docstring says "gravity always restores toward theta = 0". That is true for the
elbow, knee and hip, whose flexion convention (`joints.py:1-6`) puts the
gravity-stable pose at 0°. It is **false for the shoulder**, whose triplet
`(hip, shoulder, elbow)` reads ≈170° with the arm hanging — a fact `joints.py:62`
and the README both state explicitly. The ankle triplet `(knee, ankle, foot)` reads
≈90° standing.

Verified by releasing each joint from rest with no actuator and integrating to
steady state:

| joint | released from | model settles at | real resting pose | |
|---|---|---|---|---|
| elbow | 90° | 0° | 0° | OK |
| knee | 90° | 0° | 0° | OK |
| shoulder | 120° | **0°** | 170° | wrong |
| ankle | 110° | **0°** | 90° | wrong |

The model has the released arm swinging *up* to full forward raise and staying there.

Because `sin(180 − θ) = sin θ`, the magnitude happens to be right for the shoulder —
peak load still lands at θ = 90° (arm horizontal), which is correct. Only the
direction is inverted. Consequences:

- the PD controller's gravity feed-forward (`simulate.py:117`) adds instead of cancels;
- the GA optimizes shoulder anchor geometry against a load pointing the wrong way.
  A/B on the synthetic arm demo, same seed and settings: evolved force class
  **140 N as shipped vs 303 N with the sign corrected**;
- `plot_mechanism_curves`' "1.3 × gravity load" reference curve is drawn against the
  wrong load;
- `_design_penalties`' `hold_deficit` uses `abs()`, so the static-holdability penalty
  survives — which is why nothing visibly breaks.

**Fix.** Give the anthropometry table a neutral offset and a sign, e.g.
`GRAVITY_NEUTRAL = {"elbow": 0.0, "knee": 0.0, "hip": 0.0, "shoulder": 180.0, "ankle": 90.0}`
and return `-m·g·c·sin(θ − θ_neutral)`. For the shoulder that flips the sign
automatically. The ankle needs more than an offset: standing ankle torque is
dominated by ground reaction, not foot weight, and nothing in the model represents
it — either model the stance phase or stop claiming the ankle spec is load-sized.

### C2 — Mechanical power is sign-inverted throughout the energy and battery analysis

`actuator_design.py:165`

```python
P_mech = F_tr * v_tr  # >0: actuator does work on the limb
```

By the codebase's own convention (`mechanism.py:20`, τ = −T·dL/dθ for axial tension
T), power delivered to the limb is **−F·v**. Verified against the simulator's own
torque and velocity traces:

```
corr(true joint power, +F*v) = -0.9999
corr(true joint power, -F*v) = +0.9999
```

Every sample is therefore classified backwards in `_energy_analysis`: powered lifting
falls into the cheap `lower_factor` branch (0.15–0.25 × |P|) and regenerative
lowering is billed at full `1/η_total`. Measured effect on the demo elbow curl:

| | as shipped | sign-corrected |
|---|---|---|
| leadscrew energy | 28.5 J | 33.5 J |
| ballscrew energy | 20.0 J | 22.9 J |
| ballscrew battery | 9.7 h | 8.5 h |

15–25 % on the headline battery number, and the "% of the time doing work" narrative
is inverted. On asymmetric tasks (payload lifts, gait) the gap is larger and can go
either direction.

**Fix.** `P_mech = -F_tr * v_tr`. Add a regression test asserting
`sign(P_mech) == sign(tau_act * omega)` on a known trajectory.

### C3 — Whole-limb mass is placed at first-segment length for the shoulder and hip

`joints.py:33-39` + `simulate.py:44-46`

`SEGMENT_ANTHROPOMETRY["shoulder"]` carries Winter's **whole arm** figures
(`mass_frac 0.050`, `com_frac 0.53`, name `"whole arm"`), and `hip` carries the
**whole leg** (`0.161`, `0.45`). But `pose.py:238-239` measures `Ld` as the distance
from the joint to the *next* landmark — shoulder→elbow, hip→knee. So a whole-limb
mass is hung at an upper-arm / thigh moment arm:

| joint | model COM | anatomical COM | peak gravity torque | |
|---|---|---|---|---|
| shoulder | 16 cm | ~28 cm | 6.4 vs 10.8 N·m | 1.7× understated |
| hip | 19 cm | ~38 cm | 23.9 vs 48.3 N·m | 2.0× understated |

The two heaviest joints in the library are sized against roughly half their real
static load. `demo_data.SEG_LEN` confirms the intent (`"shoulder": (0.45, 0.30)` —
0.30 m is the upper arm).

**Fix.** Pick one and be consistent. Either measure `Ld` as the full limb
(shoulder→wrist, hip→ankle — both landmark pairs are already detected) and keep the
whole-limb mass fractions, or switch to single-segment anthropometry consistent with
the measured `Ld` (upper arm 0.028 / 0.436; thigh 0.100 / 0.433). The first is more
physically honest for a whole-limb prosthesis.

### C4 — Photo-folder timestamps are wrong whenever subsampling triggers

`capture.py:80-89`

```python
paths = paths[::stride]
...
for i, p in enumerate(paths):
    frames.append((i / photo_fps, _apply_rotation(img, rotate)))
```

`i` indexes the *subsampled* list, so the time base is compressed by `stride`.
900 photos at 4 fps → stride 3 → the pipeline sees a **75 s** recording of what was
actually **224 s** of movement: 3× too fast. Everything downstream scales with this —
`thdot_d`, the evolved `v_max` class, screw rpm, motor selection, peak electrical
power. And the run log cheerfully prints "photo timing assumed at 4.0 photos/sec".

**Fix.** `frames.append((i * stride / photo_fps, ...))`, or assign timestamps before
subsampling. Worth a test: `len(frames)` and `frames[-1][0]` for a 900-photo folder.

### C5 — `MAX_FRAMES` is silently not applied to video when the frame count is unreadable

`capture.py:104-114`

```python
n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
stride = max(1, -(-n_total // MAX_FRAMES)) if n_total else 1
```

When OpenCV cannot report a frame count it returns 0, `stride` falls back to 1, and
the `while True` loop appends **every decoded frame** to an in-memory list with no
cap. This is not a corner case: `.webm` — precisely what the guided recorder produces
(`index.html:407`) — is a common offender, as are variable-frame-rate phone clips and
anything remuxed.

- 60 s of 1080p30 held as BGR ≈ **11 GB**
- 60 s of 4K30 ≈ **45 GB**

Even the capped path is heavy: 300 frames of 4K is 7.5 GB, because frames are never
downscaled.

**Fix.** Enforce the cap inside the read loop regardless of `n_total`, and downscale
each frame to a working resolution (long edge ~720–1080 px) on read. MediaPipe gains
nothing from 4K and the whole pipeline gets faster.

---

## High

### H1 — The movement-verification story does not survive past the browser

Three independent gaps, all in the feature the README leads with
("only recordings of the correct movement get designed on"):

**(a) The server-side ROM check passes vacuously.** `runner.py:105-129` skips any
required joint that is absent from `series_by_joint`, then logs success. Executed
directly with `movement="knee_bends"` against an arm run:

```
movement check: 'Knee bends (knee)' requirements met knee ROM >= 55 deg
```

No knee was extracted or checked. Nothing at the API boundary (`app.py:100`) rejects
a movement whose target does not match the chosen joint/limb either.
*Fix:* raise if none of `min_rom`'s joints were checked, and validate
movement × target in `api_run`.

**(b) Skipped live checks never leave the browser.** `index.html:400-430` computes
`verified` and `skippedChecks` only to select a banner string; nothing is added to
the `FormData` and `/api/run` has no field for it, so `run.json` cannot distinguish a
fully verified recording from one where every step was skipped. The README's
"skipped checks flag the recording as unverified" is true of the banner only.

**(c) The Start button is only ever enabled inside the camera handler.**
`index.html:264` is the sole place `recBtn.disabled` is set to false, and no
`change` listener touches it. Default page state (target = single joint, joint =
auto) has no applicable movement, so enabling the camera disables the button; picking
a joint afterwards leaves it disabled forever. The only escape is clicking "Enable
camera" a second time.

### H2 — The camera stream and the coach poller are never torn down

`index.html:258-269`. There is no `clearInterval(coachTimer)` and no
`stream.getTracks().forEach(t => t.stop())` anywhere in the file; `stopRec()` clears
only `phaseTimer`. After recording finishes the page keeps POSTing a downscaled JPEG
to `/api/coach` roughly twice a second for the life of the tab — each request running
`PoseLandmarker.detect` under `_COACH["lock"]` inside the same threaded Flask process
that is running the GA, stealing CPU from the very job the user is waiting on. The
camera indicator also stays lit after the participant is done, which is a bad look for
a tool that films people's bodies. Clicking "Enable camera" twice orphans the first
`MediaStream` entirely.

### H3 — The webcam-derived side silently overrides the user's explicit choice

`index.html:404`: `if (coach.side) fd.set("side", coach.side);` runs in **every**
mode, and `coach.side` is never reset when the mode changes. Enable the camera, switch
to "Upload video", submit a clip of someone facing the other way — the form still
carries whatever side the webcam last saw (possibly of an empty room, per H2). The
server takes it verbatim and pins `extract_pose` to the wrong limb with no indication
anywhere in the output.

### H4 — `_SAFE_NAME` admits `..`

`app.py:38`, used at `:266` and `:309`. `^[A-Za-z0-9._-]+$` matches `..`, so
`os.path.join(RUNS_ROOT, "..")` resolves to the project root and
`send_from_directory` will serve anything beneath it. Mitigated by the 127.0.0.1
bind, but it is a one-line fix: reject names that are `.`/`..` or start with a dot.

---

## Medium

- **M1 — Payload inertia scaled linearly, not quadratically.** `runner.py:377`,
  `usage_demo.py:106`. Rescaling the payload mass by `hand_dist / Ld` makes the
  gravity moment exact (5.59 vs 5.59 N·m) but leaves `I = m·Ld²` understating the
  payload's rotational inertia by the same factor (154 vs 325 g·m², 2.1×). The
  pick-and-place demo's dynamic force demand is optimistic. Cleanest fix is to give
  `simulate_population` an explicit payload *distance* rather than folding it into the
  mass.

- **M2 — Distal inertia is internally inconsistent.** `simulate.py:45` uses the
  uniform-rod formula `m·Ld²/3` while `com` sits at 0.43–0.53·Ld, which a uniform rod
  cannot do. Winter's radius of gyration for forearm+hand about the elbow is 0.468 L
  → `I = 0.219·m·L²`, vs `0.333·m·L²` here: ~50 % high. Add a `k_frac` column and use
  `I = m·(k·Ld)²`.

- **M3 — Drawings are hardcoded to 45° flexion.** `actuator_design.py:491`
  (`th = np.deg2rad(0.5 * 90)`) and `printable.py:189`. For the shoulder, whose
  measured range is ~95–175°, the "installation drawing" and the assembled STL preview
  both show a pose the joint never reaches. Use the midpoint of the actual `th_lo,
  th_hi` — it is already computed and passed around.

- **M4 — `runs/index.json` is a lock-free read-modify-write.** `runner.py:79-95`.
  `os.replace` is atomic but two concurrent writers still lose entries. Latent today
  because `_busy()` serializes runs; it will bite the moment that constraint is
  relaxed.

- **M5 — `F_peak` is measured from an already-clipped signal.** `simulate.py:131`
  tracks `|T_ax|` *after* `np.clip(..., -F_max, F_max)`, so `final["F_peak"]` can
  never exceed the class and "% of capacity" (`runner.py:190`) tops out at exactly
  100 %. The report cannot say *how far short* a design falls. `saturation_frac`
  covers this, but only on limb runs. Record the unclipped demand alongside.

- **M6 — The packaging check can only just barely pass.** The GA penalty
  (`evolve.py:61`) is a hard wall at `L_ret ≥ stroke + DEAD_LENGTH_PARALLEL`, so the
  optimizer parks exactly on it: the demo run reports margin **0.0 mm** and "FITS". A
  check whose only two outcomes are "0 mm" and "catastrophe" carries little
  information — consider a soft margin target so the number means something.

- **M7 — Single-joint run pages never surface packaging failure or re-evolution.**
  `runner.py:230,241,337` put `actuator.fits` and `usage_check` at the top level for
  single runs, but the `packaging!` and `re-evolved` chips exist only inside the limb
  `joint_sections` loop (`run.html:68-70`). A single-joint run that logs
  `WARNING: packaging check failed` renders a page that looks entirely healthy; the
  warning survives only in `log.txt`, which `run_page` does not read once status is
  `done` (`app.py:296`).

- **M8 — Two writers fight over `#coachTip`.** `index.html:250` (every 450 ms) and
  `:351` (every 180 ms) both set text and visibility, so during a checked phase the
  tip visibly alternates between the phase hint and the server's generic tip several
  times a second. `checkHint` also skips the 1600 ms freshness guard that `checkMet`
  applies, so a stale angle can be quoted as live.

- **M9 — Polling retries forever with no ceiling.** `index.html:432`. Restart Flask
  mid-run (jobs are in-memory, so `/api/status` starts 404-ing HTML), or trip the
  800 MB `MAX_CONTENT_LENGTH`, and the UI sits at "running…" with the Run button
  disabled until the page is reloaded.

- **M10 — Hold-power model drops gear efficiency and uses a magic constant.**
  `actuator_design.py:176` computes `tau_m_hold` with a bare `0.8` where back-drive
  efficiency belongs and omits `eta_gear` entirely, while the moving branch uses the
  full `eta_tot`.

- **M11 — `.spec h2 { display: none; }`** (`base.html:74`) was aimed at the `<h2>`
  that `_mini_markdown` emits for the spec's `# ` line, but it also hides the
  template's own "Real-world actuator spec" heading (`run.html:104`), leaving an
  untitled wall of text that is also unlabelled in the accessibility tree.

---

## Low / hygiene

**Front end**

- `{{ movements_json | safe }}` inside `<script>` (`index.html:142`) — `json.dumps`
  does not escape `</script>`. Use `{{ MOVEMENTS | tojson }}`.
- `/api/runs` fields are interpolated raw into `innerHTML` (`index.html:455-470`).
  Not currently exploitable — names come from `secure_filename`/`_slug` — but
  `r.error` is free-form server text and `index.json` is trusted verbatim from disk.
- Recorded blob URLs are never revoked (`index.html:373`); each retake pins tens of MB
  for the life of the page.
- Container guessed by substring (`index.html:407`): if `pickMime()` returns `""` the
  UA may produce Matroska, which is then saved as `.webm` and fails later in
  `load_frames` rather than at upload.
- Re-recording via the primary button leaves `#playback` visible over the hidden
  `#camView`, so the participant follows the overlay on top of the previous take.
- An uncaught `new MediaRecorder(...)` failure (`index.html:311`) leaves
  `recording = true` with nothing shown to the user.
- Selecting "free movement" is silently reverted on any target/joint change
  (`index.html:160-163`), because its `value=""` is never in `keys`.
- Accessibility: 14 of 15 `<label>`s are unassociated (no `for=`, no `aria-*`
  anywhere in the templates); `#recState`, `#liveCoach`, `#verifyBanner`, `#console`
  have no `aria-live`, so no recording feedback is ever announced; validation uses
  `alert()`; the run list is `innerHTML`-rebuilt every 5 s, destroying keyboard focus.
- `/run/<name>` opened on a running job is a dead snapshot — 40 frozen log lines, no
  auto-refresh, no link back to the live console.

**Python**

- Dead code: `time` (`app.py:14`), `Optional` (`actuator_design.py:25`),
  `DEAD_LENGTH_PARALLEL` and unused `L_ret` (`printable.py:24,138`), `os`
  (`usage_demo.py:13`, `visualize.py:3`), `F_at` and `tr` (`usage_demo.py:156,242`),
  `tri_pts` (`visualize.py:109`), and the no-op branch in
  `_variant_dict` (`actuator_design.py:295` — `asdict` already recurses).
- MediaPipe video timestamps `int(round(t*1000))` (`pose.py:104`) can collide at high
  frame rates, which `detect_for_video` rejects.
- `make_gif`'s axis bounds call `xs.min()` on a possibly-empty array
  (`visualize.py:151`) when no landmark in the decimated set is valid.
- HEIC conversion is macOS-only (`capture.py:22-32`) and degrades silently elsewhere,
  though the README does say macOS.
- `_JOBS` is never pruned (`app.py:35`); every job's full log lines are retained for
  the process lifetime.
- `api_run` holds `_LOCK` while writing up to 800 MB of uploads.

**Project**

- No `pyproject.toml`, no pinned versions (`>=` only), no `.gitignore`, no CI.
- Tests are `assert`s under `if __name__ == "__main__"` rather than pytest, so there
  is no per-test isolation or reporting. Coverage gaps are the risky parts:
  `capture.py`, `runner.py`, `app.py` and `printable.py` have **no** tests at all —
  C4 and C5 would both have been caught by a ten-line capture test.
- `printable.py` builds the actuator mock at `body_len` (stroke + dead length) rather
  than the evolved `L_ret`, while the link plates use the true anchor positions. These
  coincide today only because the GA parks on the packaging wall (M6); if that
  constraint ever slackens the printed parts will not assemble.

---

## What's working well

- **`mechanism.length_and_derivative`** — the virtual-work derivation
  (`dL/dθ = -cross(A,B)/L`) is correct, and `selftest.test_geometry` checks it against
  finite differences *and* verifies `r == |dL/dθ|`. This is the part most likely to be
  quietly wrong in a project like this, and it isn't.
- **`_screw_efficiency`** — the Acme efficiency formula and the `tanλ ≤ μ` self-locking
  criterion are textbook-correct, and the selftest independently re-derives the
  self-locking flag rather than trusting the function.
- **The motor-sizing chain** — separating continuous from peak torque, selecting the
  class on continuous and bounding peak at 3.5×, and computing hold power from the
  motor constant as `(τ/k_m)²` are all dimensionally and practically right.
- **`pose.py`'s detect-once / derive-many split** — the right architecture for limb
  runs, and the visibility gating, short-gap interpolation and longest-run selection
  are careful work.
- **Operational hygiene** — atomic JSON writes via `os.replace`, one self-contained
  folder per run, `log.txt` tee'd to the UI, `pose_check_*.jpg` overlays surfaced first
  in the docs. This is a codebase you can debug from its own outputs.
- **The disclaimers** — accurate, specific, repeated at every level (module docstrings,
  spec sheet, report, README), and they name the actual limitations rather than
  hand-waving. The "verify against vendor datasheets" framing is exactly right.

---

## Suggested order of attack

1. **C2** (one character) and **C4** (one expression) — smallest fixes, immediate
   correctness gain.
2. **C5** — cap and downscale frames on read; removes an OOM class and speeds
   everything up.
3. **C1** and **C3** together — they are the same conversation about what the flexion
   convention and the anthropometry table actually mean. Land them with a test that
   releases each joint from rest and asserts where it settles.
4. **H1** — decide whether the verification claim is real. Right now the README
   promises more than the code delivers, which for a body-measurement tool is the kind
   of gap worth closing before anything else ships.
5. **H2/H3** — small, and they are the difference between the recorder feeling
   trustworthy and not.
6. Move the selftests to pytest and add coverage for `capture.py` and `runner.py`
   before touching anything else.
