# Neurogolf

Work toward the **2026 NeuroGolf Championship** (Kaggle / IJCAI-ECAI 2026): build the smallest
ONNX networks that correctly reproduce ARC-AGI grid transformations, one network per task
(`task001.onnx … task400.onnx`).

## Contents

- [`PLAN.md`](PLAN.md) — concrete end-to-end plan for building our own 400-network **baseline**
  from scratch (the prerequisite that the public golf notebooks assume already exists).
- [`solver/`](solver/) — **runnable Tier-A coverage engine + golf passes** (the start of that
  baseline). Symbolic program synthesis: detect a transformation from the train examples → emit
  a tiny ONNX graph → shrink it losslessly. Self-test passes 7/7 with no competition data needed.

## TL;DR of the plan

- This is **program synthesis with the answers in hand**, not blind ARC prediction.
- **Correctness is a hard gate** (wrong ≈ 0 points) ⇒ **coverage first, golf second**.
- **Symbolic fixed-weight graphs beat trained nets** on score ⇒ prefer hand-built / synthesized
  ONNX graphs; use trained CNNs only as a last-resort fallback.
- Tiers: **M0** always-valid 400-file fallback → **Tier A** geometry+color → **Tier B** DSL
  synthesis → **Tier C** hard tail → **M4** golf passes (incl. sparse-Conv→dilated-Conv).

See [`PLAN.md`](PLAN.md) for the full design, repo layout, milestones, and open questions.

## The `solver/` engine

| File | Role |
|------|------|
| `onnx_ops.py` | minimal **content-aware** ONNX graph builders (Gather / Conv / Slice / Transpose …) |
| `solvers.py`  | detector→builder cascade; `solve_task(train)` → `(model, method)` |
| `surgery.py`  | behavior-preserving golf passes (prune, dedup, identity, conv1x1→Gather) |
| `scorer.py`   | **official** scoring + correctness, wrapping `Dataset/neurogolf_utils.py` |
| `analyze.py`  | diagnostic: classify each task's transform on content grids; coverage ceiling |
| `selftest.py` | synthetic validation of the cascade + surgery (no data needed) |
| `run.py`      | end-to-end over the real 400 tasks → `submission.zip` + `ledger.csv` |

### Current baseline (official scorer, gate = train+test+arc-gen)
**11/400 solved, 208.53 points** — 2 color-permute, 2 conv1x1 color-map, 2 transpose,
2 rot180, 2 upscale, 1 rot90. (Up from 4/95 once the scorer/correctness were fixed.)

### Why only 11 — the coverage ceiling
`analyze.py` shows **only 17/400 tasks are whole-grid transforms**; the other 383 need
object-level reasoning / composition (the real ARC difficulty). Of those 17, the ones with
**variable grid sizes** (flips, tile) or **variable factors** (some upscales) are unsolvable
by a *fixed* graph — there are no dynamic ops (`NonZero`/`Loop` are banned) to discover the
size at runtime. ~11 are size-agnostic or size-constant, hence solvable. **Real progress past
this requires Tier-B DSL synthesis with unrolled neighborhood/object ops** (see `PLAN.md`).

```bash
python3.12 -m venv .venv && ./.venv/bin/pip install numpy onnx onnxruntime kaggle
./.venv/bin/python solver/selftest.py      # -> 7/7 cases passed
```

Maps onto `PLAN.md`'s planned `ng/` package: `onnx_ops.py` ≈ `onnx_build.py`+`compile.py`,
`solvers.py` ≈ Tier-A of `synth.py`, `surgery.py` ≈ `golf.py`, `run.py` ≈ `build_submission.py`.

**Bugs fixed vs the reference `conv1x1` notebook:** `rot90`/`rot270` used two independent Gathers
(impossible for a real rotation, which mixes rows/cols → now `Transpose`+flip); the rotation case
compared example-0's output against all examples (never matched with >1 example); `const` could
memorize a single example (now requires ≥2 identical outputs).

**Open blocker (Unknown #3):** content-region vs full-canvas semantics for non-30×30 grids — the
geometric solvers currently assume full-canvas. Resolve from `neurogolf_utils.py` once the
competition rules are accepted and the data downloads, then swap the local `runs_correct` /
`param_count` proxies for the official scorer.

## Reference Kaggle material

Competition: <https://www.kaggle.com/competitions/neurogolf-2026>

Notebooks studied / referenced:
- Graph surgeries (golf passes): <https://www.kaggle.com/code/seddiktrk/neurogolf-2026-all-graph-surgeries>
- conv1x1 + more solvers (coverage engine ported into `solver/`): <https://www.kaggle.com/code/badboyhalo1801/neurogolf-v254-conv1x1-more-solvers>
- Sparse-Conv → dilated-Conv golf (basis of PLAN §8/M4): <https://www.kaggle.com/code/hoangvux/neurogolf>
- Starter notebook: <https://www.kaggle.com/code/nihilisticneuralnet/neurogolf-championship-2026-starter-notebook>
- Rule-based ONNX solver: <https://www.kaggle.com/code/imaadmahmood/neurogolf-2026-rule-based-onnx-solver>
- NeuroGolf 2026 ONNX: <https://www.kaggle.com/code/mpwolke/neurogolf-2026-onnx>
