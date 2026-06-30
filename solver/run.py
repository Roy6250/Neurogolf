"""End-to-end over the real 400 tasks: solve -> golf -> official-score -> pack.

Uses the competition scorer (solver/scorer.py wrapping Dataset/neurogolf_utils.py)
as ground truth for both correctness and points. Writes models/, submission.zip,
and ledger.csv (per-task: method, correct, points, memory, params).

  ./.venv/bin/python solver/run.py [N]      # N = limit #tasks (default 400)
"""
import csv
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx

import onnx_ops as ops
import solvers
import surgery
import scorer

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "Dataset"
OUT = REPO / "models"


def examples_one_hot(task):
    ti, to = [], []
    for eg in task.get("train", []):
        ia, oa = np.array(eg["input"], np.float32), np.array(eg["output"], np.float32)
        if max(ia.shape + oa.shape) > 30:
            continue
        ti.append(ops.one_hot(ia, *ia.shape))
        to.append(ops.one_hot(oa, *oa.shape))
    return ti, to


def main(limit=400):
    OUT.mkdir(exist_ok=True)
    rows = []
    total_points = solved = 0

    for i in range(1, limit + 1):
        tf = DATA / f"task{i:03d}.json"
        if not tf.exists():
            continue
        task = json.loads(tf.read_text())
        model, method = solvers.solve_task(task.get("train", []))

        frac = 0.0
        if model is not None:
            # golf, but keep only if it still passes the train examples
            try:
                opt, _ = surgery.apply_all(model)
                if solvers.runs_correct(opt, *examples_one_hot(task)):
                    model = opt
            except Exception:
                pass
            # FRACTIONAL scoring: the board awards base x (held-out fraction correct),
            # with no penalty for wrong. So we SHIP a train-passing net even if it misses
            # some arc-gen examples — it banks base x frac >= 0, never worse than the
            # placeholder. arc-gen is our held-out proxy; the figure is an estimate.
            ok, right, wrong = scorer.verify(model, task)
            frac = right / (right + wrong) if (right + wrong) else 0.0
            if not ok:
                method = f"{method}~{frac:.2f}"   # near-miss: shipped for partial credit
            solved += int(ok)

        if model is None:
            method = method or "unsolved"
            model = ops.m_identity()   # valid placeholder; ~0 board on a non-identity task

        pts, mem, par = scorer.score(model)
        # score_network ignores correctness; board points = base x held-out fraction.
        board_pts = pts * frac
        total_points += board_pts
        onnx.save(model, str(OUT / f"task{i:03d}.onnx"))
        rows.append((f"task{i:03d}", method, round(board_pts, 3), round(frac, 3), mem, par))
        print(f"  task{i:03d}  {method:26s} est={board_pts:6.2f} frac={frac:.2f} "
              f"mem={mem} par={par}", flush=True)

    with open(REPO / "ledger.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "method", "est_points", "frac", "memory", "params"])
        w.writerows(rows)

    with zipfile.ZipFile(REPO / "submission.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(OUT.glob("task*.onnx")):
            zf.write(p, arcname=p.name)

    near = sum(1 for r in rows if 0 < r[3] < 1.0)
    methods = Counter(r[1].split("~")[0].split("(")[0] for r in rows)
    print(f"\nFULLY SOLVED (frac=1) {solved}/{len(rows)}   NEAR-MISS (0<frac<1) {near}")
    print(f"ESTIMATED TOTAL POINTS {total_points:.2f}  (board = base x held-out fraction)")
    print("methods:", dict(methods))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 400)
