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

        if model is not None:
            # golf, but keep only if it still passes the train examples
            try:
                opt, _ = surgery.apply_all(model)
                if solvers.runs_correct(opt, *examples_one_hot(task)):
                    model = opt
            except Exception:
                pass
            # official gate: train + test + arc-gen must all pass
            ok, right, wrong = scorer.verify(model, task)
            if not ok:
                method, model = f"FAILED_GATE({method})", None

        if model is None:
            if method is None:
                method = "unsolved"
            model = ops.m_identity()  # valid placeholder, never leave a hole
            solved_this = False
        else:
            solved_this = True
            solved += 1

        pts, mem, par = scorer.score(model)
        # IMPORTANT: score_network does NOT check correctness. The leaderboard awards
        # points only if the network is ALSO correct, so an unsolved placeholder scores
        # 0 on the board even though its graph cost yields 25 in isolation.
        board_pts = pts if solved_this else 0.0
        total_points += board_pts
        onnx.save(model, str(OUT / f"task{i:03d}.onnx"))
        rows.append((f"task{i:03d}", method, round(board_pts, 3), mem, par))
        flag = "" if solved_this else "  (placeholder, board=0)"
        print(f"  task{i:03d}  {method:24s} pts={board_pts:6.3f} mem={mem} par={par}{flag}", flush=True)

    with open(REPO / "ledger.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "method", "points", "memory", "params"])
        w.writerows(rows)

    with zipfile.ZipFile(REPO / "submission.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(OUT.glob("task*.onnx")):
            zf.write(p, arcname=p.name)

    methods = Counter(r[1].split("(")[0] for r in rows)
    print(f"\nSOLVED {solved}/{len(rows)}   TOTAL POINTS {total_points:.2f}")
    print("methods:", dict(methods))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 400)
