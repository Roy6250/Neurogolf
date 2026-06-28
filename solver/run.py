"""End-to-end: solve every task, golf it, pack submission.zip.

Requires the competition data (after accepting the rules on Kaggle):
    export KAGGLE_API_TOKEN=...   # or ~/.kaggle/kaggle.json
    kaggle competitions download -c neurogolf-2026 -p data && unzip data/*.zip -d data

Once `data/neurogolf_utils/neurogolf_utils.py` exists we should swap the local
`runs_correct` / `param_count` proxies for the official scorer to match the
leaderboard exactly (see STRATEGY.md Phase 0).
"""
import json
import zipfile
from collections import Counter
from pathlib import Path

import onnx

import onnx_ops as ops
import solvers
import surgery

DATA = Path("data")
OUT = Path("models")


def main():
    OUT.mkdir(exist_ok=True)
    task_files = sorted(DATA.glob("task*.json"))
    if not task_files:
        print("No task*.json under data/. Accept the competition rules and download first.")
        return

    stats = Counter()
    total_params = 0
    for tf in task_files:
        tn = tf.stem
        task = json.loads(tf.read_text())
        model, method = solvers.solve_task(task.get("train", []))

        if model is None:
            stats["unsolved"] += 1
            onnx.save(ops.m_identity(), str(OUT / f"{tn}.onnx"))  # placeholder
            continue

        # golf (re-verify after, since some passes are only conditionally safe)
        try:
            opt, _ = surgery.apply_all(model)
            if solvers.runs_correct(opt, *_examples_tensors(task)):
                model = opt
        except Exception:
            pass

        stats[method] += 1
        total_params += surgery.param_count(model)
        onnx.save(model, str(OUT / f"{tn}.onnx"))

    with zipfile.ZipFile("submission.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(OUT.glob("task*.onnx")):
            zf.write(p, arcname=p.name)

    print(f"solved: {sum(v for k, v in stats.items() if k != 'unsolved')}/{len(task_files)}")
    print(f"unsolved: {stats['unsolved']}  total params: {total_params}")
    print("methods:", dict(stats))


def _examples_tensors(task):
    import numpy as np
    ti, to = [], []
    for eg in task.get("train", []):
        ia, oa = np.array(eg["input"], np.float32), np.array(eg["output"], np.float32)
        ti.append(ops.one_hot(ia, *ia.shape))
        to.append(ops.one_hot(oa, *oa.shape))
    return ti, to


if __name__ == "__main__":
    main()
