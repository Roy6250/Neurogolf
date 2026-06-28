# Neurogolf

Work toward the **2026 NeuroGolf Championship** (Kaggle / IJCAI-ECAI 2026): build the smallest
ONNX networks that correctly reproduce ARC-AGI grid transformations, one network per task
(`task001.onnx … task400.onnx`).

## Contents

- [`PLAN.md`](PLAN.md) — concrete end-to-end plan for building our own 400-network **baseline**
  from scratch (the prerequisite that the public golf notebooks assume already exists).

## TL;DR of the plan

- This is **program synthesis with the answers in hand**, not blind ARC prediction.
- **Correctness is a hard gate** (wrong ≈ 0 points) ⇒ **coverage first, golf second**.
- **Symbolic fixed-weight graphs beat trained nets** on score ⇒ prefer hand-built / synthesized
  ONNX graphs; use trained CNNs only as a last-resort fallback.
- Tiers: **M0** always-valid 400-file fallback → **Tier A** geometry+color → **Tier B** DSL
  synthesis → **Tier C** hard tail → **M4** golf passes (incl. sparse-Conv→dilated-Conv).

See [`PLAN.md`](PLAN.md) for the full design, repo layout, milestones, and open questions.
