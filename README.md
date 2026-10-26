# LimbForge

Most prosthetics are designed by people. I wanted to see what happens if you let evolution have a go.

Film a real person moving, and LimbForge tracks their joints, then evolves an actuator mechanism to reproduce that movement, simulates it, sizes real motors for it, and exports 3D printable parts. The engine that grows mechanisms from scratch lives in its own repo, [open-topology-engine](https://github.com/PressureCook/open-topology-engine).

## The pipeline

Movement footage of **a real person** goes in; out comes an **evolved, physically
simulated prosthetic actuator joint** (or a **whole limb** of them), a **real-world
actuator spec**, and **3D-printable STL parts** of the mechanism.

```
capture ─► pose ─► joint-angle ─► evolutionary design ─► physics    ─► real-world  ─► 3D-printable ─► animated demo,
(guided    (Media   trajectories    (GA over actuator      simulation    actuator      STL parts       per-joint
 webcam /   Pipe)   θ(t) + limb     geometry, force,       per joint     spec          (linkage +      breakdown,
 uploads)           lengths         speed class)                                       actuator mock)  report
```

> Exploration/demo tool only: **not** a medical device and not a substitute for
> professional prosthetics engineering. Actuator numbers are first-principles
> estimates; verify against vendor datasheets before building anything. Printed
> parts are form/kinematics demonstrators, not load-bearing prosthetics.


## Open-topology evolution + 3D anatomical viewer (new)

- `--engine graph` now searches an **open mechanism graph** ("mechgraph"): the
  population is seeded FROM SCRATCH with the minimal possible mechanism (two
  bones + one linear actuator) and structural mutation grows everything richer -
  Assur-dyad linkage growth, actuator re-attachment onto invented links,
  antagonists, springs - behind a universal feasibility gate (moment arm,
  stroke ratio, packaging, static holdability, self-intersection, envelope).
- Every single-joint run now exports **scene.glb + scene.json**: parametric
  anatomical bones scaled to YOUR measured segment lengths, with the evolved
  mechanism mounted; the run page shows an **interactive 3D viewer** (orbit,
  scrub, layer toggles, labelled landmarks, force-utilisation colouring).
  Anatomy is representative geometry, NOT imaging of the participant, and the
  fixed-revolute axis is an approximation - both stated in the viewer.
- The **guided recorder** now guides a full data-capture protocol even with no
  task selected (calibration hold, slow ROM sweeps, fast reps, mid-range hold)
  and saves the live coach angle timeline with the run.

## Two ways to run it

**Web UI (recommended):**

```bash
.venv/bin/python app.py
```

then open <http://127.0.0.1:5001>:

- **Movement tasks**: pick what the prosthetic must be able to do: reach &
  pick up, push, lift/carry (arm); walk in place, sit-to-stand (leg); plus
  single-joint range tasks. Each task defines timed instructions, live
  verification thresholds, and the minimum range of motion a recording must contain.
- **Record (guided + verified)**: the site watches the webcam while it guides
  the participant: each step only completes when the pose tracker confirms the
  required movement actually happened, with live corrective tips ("can't see
  the knee: turn side-on", "go further: elbow at 40°, needs 95°+"). Skipped
  checks flag the recording as unverified, and the server re-checks every
  upload against the task's required range of motion: only recordings of the
  correct movement get designed on.
- **Upload video / photos**: same task instructions shown, same server-side
  movement validation applied.
- **What to test**: a specific joint (left/right/auto) or a **full limb**;
  limb runs extract every joint from one recording, evolve each, and finish
  with a **real-world usage demo**: a walking cycle for the leg, or picking a
  1 kg object off a table onto a shelf for the arm. If a design evolved from
  the recording alone can't handle the task (force/speed saturation), it is
  automatically **re-evolved with the task in the fitness trajectory**.
- Watch the pipeline log live; every run's outputs live on its own results page.

**CLI:**

```bash
.venv/bin/python run_pipeline.py --demo
.venv/bin/python run_pipeline.py --demo --limb leg
.venv/bin/python run_pipeline.py --input curl.mov --joint elbow --side right --body-mass 75
.venv/bin/python run_pipeline.py --input squat.mov --limb leg --side right
```

## Runs and outputs

Every run gets its own folder: `runs/<timestamp>_<label>/` (tracked in
`runs/index.json`, shown as cards in the UI). Inside each run folder:

| file | what it is |
|---|---|
| `demo.gif` / `limb_demo.gif` | animated side-by-side: recorded movement vs the evolved prosthesis (full chain for limb runs) |
| `usage_demo.gif` | limb runs: the prosthesis performing the real-world task (walking / pick & place) with per-joint force utilization |
| `actuator_spec.md` / `.json` | **real-world actuator design** (screw, gearing, motor, electrical, energy/battery, structure, packaging) |
| `actuator_drawing.png` | datasheet-style spec sheet: dimensioned views, ratings table, installation drawing |
| `print_plate.stl`, `part_*.stl` | **3D-printable demonstrator**: linkage plates with the evolved anchor geometry, telescoping actuator mock at true dimensions, pins (`printable_preview.png` shows the plate + assembly) |
| `report.md` | human-readable summary (per-joint breakdown for limb runs) |
| `joints/<joint>/…` | limb runs: the complete single-joint output set per joint |
| `pose_check_*.jpg` | skeleton overlays: **check these first** on real footage |
| `tracking.png`, `mechanism_curves.png`, `convergence.png` | analysis plots |
| `design.json`, `trajectory.csv`, `run.json`, `log.txt` | machine-readable data + log |
| `input/` | the uploaded/recorded video or photos (UI runs) |

## Capturing good input

- Film/photograph **from the side** of the moving joint, whole limb visible, good light.
- Move **slowly through the full range**, 3+ repetitions; hold still at the ends briefly.
- Video beats photos (real timing). For photo folders, set "photos per second"
  (`--photo-fps`) to how fast you actually shot; iPhone HEIC converts automatically.

## How each stage works

1. **Capture** (`capture.py`): photo folder or video in, ordered frames out.
2. **Pose** (`pose.py`): MediaPipe landmarks per frame; flexion angle of the chosen
   joint from metric 3D world landmarks; visibility gating, gap interpolation; your
   segment lengths measured from the pose.
3. **Evolve** (`evolve.py`, `mechanism.py`): genetic algorithm over a 6-gene genome
   (anchor placement on both segments + actuator force/speed class). Fitness =
   simulated tracking error + size/force/speed cost + hard physical penalties:
   stroke ratio, minimum moment arm, static gravity holdability, and **packaging**
   (retracted length must fit stroke + real dead length).
4. **Simulate** (`simulate.py`): distal segment as a pendulum (anthropometric
   mass/inertia scaled to your body mass); PD servo with gravity feed-forward,
   force and speed saturation.
5. **Actuator engineering** (`actuator_design.py`): turns the evolved class +
   recorded force/velocity traces into a buildable spec: lead-screw vs ball-screw
   variants (efficiency from thread geometry, self-locking check), gear ratio +
   motor class sizing, bus voltage and peak current, **energy per recorded movement
   and battery estimate for both variants**, buckling/pin checks, mass estimate,
   dimensioned drawing, and a recommendation (fail-safe self-locking vs efficient
   ball screw + brake).
6. **3D print** (`printable.py`): trimesh/manifold CSG builds watertight STL parts:
   two linkage plates with the joint bore and the evolved anchor brackets at exact
   positions, a telescoping barrel+rod mock at the specced retracted length/stroke/
   diameters with the motor pod, and shouldered pins; auto-scales oversized parts.
7. **Usage demo** (`usage_demo.py`): limb runs simulate the evolved joints through
   a synthetic normative-style task (gait cycle / pick & place with payload at the
   hand). Designs that saturate on the task are re-evolved with the task appended
   to the training trajectory, so the final prosthesis is designed for the job,
   not just the recording.
8. **Visualize/report** (`visualize.py`, `runner.py`): plots, GIFs (recorded
   movement, full-limb chain, usage scene), report, run index. Limb runs
   (`--limb arm|leg`) do stages 2–7 per joint from one recording; joints are
   designed independently (inter-joint reaction forces are not modeled).
9. **Live coach** (`app.py /api/coach`): during guided recording the browser
   streams downscaled frames; the server runs pose detection and returns joint
   visibility + angles, which gate the instruction phases and generate tips.

Self-tests:

```bash
.venv/bin/python -m prosthetic_pipeline.selftest        # geometry, GA, sim, actuator design
.venv/bin/python -m prosthetic_pipeline.selftest_pose   # pose stage with a stubbed detector
```

## Options (CLI and UI advanced panel)

`--joint` elbow/knee/shoulder/hip/ankle/auto · `--side` left/right/auto ·
`--limb` arm/leg (full-limb mode) · `--movement` task key (enforces its ROM
requirements on the recording) · `--no-stl` · `--body-mass` kg · `--photo-fps` ·
`--rotate` 90/180/270 · `--pop`, `--generations`, `--seed` · `--label` ·
`--runs-root`, `--out` · `--model` (MediaPipe `.task` file, bundled at
`models/pose_landmarker_full.task`).

## Open-topology mechanism evolution (experimental engine)

`--engine graph` (or the "design engine" selector in the UI) replaces the fixed
linear-actuator architecture with an **open topology search** over planar
mechanism graphs (see `SPEC-open-topology-evolution.md`):

- registered topologies: `direct_linear` (the legacy design, extendable with
  torsional springs and tendon/linear antagonists), `four_bar` (crank+coupler
  linkage with Newton loop closure and a hard validity gate: assembly, branch
  flips, transmission angle, singularity, limb envelope), `rotary_direct`
  (geared drive at the joint). `--topology` restricts the search; `auto` runs
  the structural search.
- every candidate reduces to a single-DOF **transmission table** and is
  integrated by a vectorised reduced-EOM simulator including the
  `1/2 dI/dtheta thetadot^2` term; species share one batched table build and
  one simulation call per shape.
- the search is **speciated with fitness sharing**; structural mutations are
  behaviour-preserving at birth (springs decode to k=0, new actuators to zero
  capacity); fitness adds complexity, energy, mass, saturation and singularity
  terms; immigrants keep diversity alive.
- honesty checks ship with every run: a **hold-out split** (evolve on the first
  70% of the recording, report RMSE on the rest), a **robustness screen**
  (gene jitter ~ printed tolerances; designs that stop assembling are
  rejected), and a **greedy-vs-QP allocation gap** for multi-actuator designs.
- downstream is topology-agnostic: per-actuator requirement sizing (screw chain
  for linear/tendon, a gearbox+motor path for rotary), declarative STL
  primitives, transmission plots, a topology-drawn GIF, and `design.json`
  **schema_version 2** with the full genome (v1 files still read as the legacy
  design).

Tests: `tests/test_topology.py` (reduction contract, mandatory dI term,
dead-centre rejection) and `tests/test_graph_engine.py` (behaviour-preserving
mutations, QP vs greedy, batch equality, rotary sizing).

## Limitations / next steps

- 2D sagittal-plane model (flexion/extension only), gravity in the motion plane.
- Photo folders get assumed timing: use video for honest dynamics.
- Anthropometric mass properties are population averages (Winter tables).
- Actuator constants (friction, motor classes, efficiencies) are labeled estimates.
- Natural extensions: multi-joint chains, 3D mechanisms, vendor catalog matching,
  EMG-driven control, export of the drawing to CAD.
