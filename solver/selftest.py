"""Synthetic self-test: validates the solver cascade + surgery without competition data.

Builds toy tasks with known transformations, checks solve_task picks a correct (and
cheap) graph, and checks surgery preserves behavior while not increasing params.
Run: ./.venv/bin/python selftest.py
"""
import numpy as np
import onnxruntime as ort

import onnx_ops as ops
from onnx_ops import to_labels
import solvers
import surgery


def grid_examples(fn, shapes, seed=0):
    rng = np.random.default_rng(seed)
    egs = []
    for (h, w) in shapes:
        g = rng.integers(0, 10, size=(h, w))
        egs.append({"input": g.tolist(), "output": fn(g).tolist()})
    return egs


# NOTE: the geometric solvers operate on the FULL 30x30 canvas (padding = color 0),
# so flip/rotate are only canvas-consistent when content fills the canvas. We test
# those on 30x30 grids. Whether the real harness uses content-region or canvas
# semantics is Unknown #3 (resolved once neurogolf_utils downloads). identity and
# color_permute are canvas-agnostic, so we test them on small grids too.
FULL = [(30, 30), (30, 30)]  # distinct random fills (different seeds per case below)
CASES = {
    "identity":      (lambda g: g,              [(5, 5), (7, 4), (3, 8)]),
    "color_permute": (lambda g: (g + 3) % 10,   [(5, 5), (6, 4), (3, 7)]),
    "transpose":     (lambda g: g.T,            FULL),
    "flip_h":        (lambda g: g[:, ::-1],     FULL),
    "flip_v":        (lambda g: g[::-1, :],     FULL),
    "rot180":        (lambda g: np.rot90(g, 2), FULL),
    "rot90":         (lambda g: np.rot90(g, 1), FULL),
}


def run():
    passed, failed = 0, 0
    for name, (fn, shapes) in CASES.items():
        train = grid_examples(fn, shapes)
        model, method = solvers.solve_task(train)
        if model is None:
            print(f"  ✗ {name:14s} -> NO SOLVER FOUND")
            failed += 1
            continue

        # verify on a fresh (unseen) example -> generalization, not memorization
        held = grid_examples(fn, [shapes[0]], seed=99)[0]
        sess = ort.InferenceSession(model.SerializeToString(),
                                    providers=["CPUExecutionProvider"])
        ti = ops.one_hot(np.array(held["input"], np.float32),
                         *np.array(held["input"]).shape)
        got = to_labels(sess.run(None, {"input": ti})[0])
        exp = np.array(held["output"])
        gen_ok = np.array_equal(got[:exp.shape[0], :exp.shape[1]], exp)

        # surgery preserves behavior + does not grow params
        p0 = surgery.param_count(model)
        opt, rep = surgery.apply_all(model)
        sess2 = ort.InferenceSession(opt.SerializeToString(),
                                     providers=["CPUExecutionProvider"])
        got2 = to_labels(sess2.run(None, {"input": ti})[0])
        surg_ok = np.array_equal(got, got2)
        p1 = surgery.param_count(opt)

        ok = gen_ok and surg_ok and p1 <= p0
        print(f"  {'✓' if ok else '✗'} {name:14s} -> {method:18s} "
              f"params {p0}->{p1}  gen={gen_ok} surgery_ok={surg_ok}")
        passed += ok
        failed += (not ok)

    print(f"\n{passed}/{passed + failed} cases passed")
    return failed == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
