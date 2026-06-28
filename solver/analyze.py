"""Diagnostic: classify each task's transformation on the CONTENT grids (actual HxW,
not the 30x30 canvas), and report size-consistency. Tells us the real coverage ceiling
per template and how many tasks are size-constant (so a fixed graph can do content ops).

  ./.venv/bin/python solver/analyze.py
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "Dataset"


def classify(pairs):
    """pairs: list of (in_grid, out_grid) numpy arrays. Return a transform label that
    holds for ALL pairs on the content region, or None."""
    def all_(fn):
        return all(fn(a, b) for a, b in pairs)

    same_shape = all_(lambda a, b: a.shape == b.shape)
    if all_(lambda a, b: np.array_equal(a, b)):
        return "identity"
    if all_(lambda a, b: a.shape[::-1] == b.shape and np.array_equal(a.T, b)):
        return "transpose"
    if same_shape and all_(lambda a, b: np.array_equal(a[:, ::-1], b)):
        return "flip_h"
    if same_shape and all_(lambda a, b: np.array_equal(a[::-1, :], b)):
        return "flip_v"
    if same_shape and all_(lambda a, b: np.array_equal(np.rot90(a, 2), b)):
        return "rot180"
    if all_(lambda a, b: np.array_equal(np.rot90(a, 1), b)):
        return "rot90"
    if all_(lambda a, b: np.array_equal(np.rot90(a, 3), b)):
        return "rot270"

    # color map (same shape, consistent bijection-ish recolor across all pairs)
    if same_shape:
        mp = {}
        ok = True
        for a, b in pairs:
            for x, y in zip(a.ravel(), b.ravel()):
                if x in mp and mp[x] != y:
                    ok = False; break
                mp[x] = y
            if not ok: break
        if ok:
            return "color_map"

    # integer block upscale: out[i,j] = in[i//ry, j//rx]
    def is_upscale(a, b):
        if b.shape[0] % a.shape[0] or b.shape[1] % a.shape[1]:
            return False
        ry, rx = b.shape[0] // a.shape[0], b.shape[1] // a.shape[1]
        return (ry, rx) != (1, 1) and np.array_equal(np.repeat(np.repeat(a, ry, 0), rx, 1), b)
    if all_(is_upscale):
        return "upscale"

    # tile: out = np.tile(in, (ry,rx))
    def is_tile(a, b):
        if b.shape[0] % a.shape[0] or b.shape[1] % a.shape[1]:
            return False
        ry, rx = b.shape[0] // a.shape[0], b.shape[1] // a.shape[1]
        return (ry, rx) != (1, 1) and np.array_equal(np.tile(a, (ry, rx)), b)
    if all_(is_tile):
        return "tile"

    # constant output (all outputs identical grid)
    outs = [b for _, b in pairs]
    if len(outs) >= 2 and all(np.array_equal(outs[0], o) for o in outs[1:]):
        return "constant"

    return None


def main():
    label_counts = Counter()
    size_const_in = size_const_out = both_const = 0
    n = 0
    examples_by_label = {}
    for i in range(1, 401):
        tf = DATA / f"task{i:03d}.json"
        if not tf.exists():
            continue
        n += 1
        task = json.loads(tf.read_text())
        train = task.get("train", [])
        pairs = [(np.array(e["input"]), np.array(e["output"])) for e in train]
        if not pairs:
            continue
        in_shapes = {p[0].shape for p in pairs}
        out_shapes = {p[1].shape for p in pairs}
        size_const_in += len(in_shapes) == 1
        size_const_out += len(out_shapes) == 1
        both_const += len(in_shapes) == 1 and len(out_shapes) == 1

        label = classify(pairs) or "OTHER"
        label_counts[label] += 1
        examples_by_label.setdefault(label, []).append(f"task{i:03d}")

    print(f"tasks analyzed: {n}")
    print(f"size-constant inputs:  {size_const_in}/{n}")
    print(f"size-constant outputs: {size_const_out}/{n}")
    print(f"both constant:         {both_const}/{n}\n")
    print("transform (on content grids, holds across all train pairs):")
    for label, c in label_counts.most_common():
        ex = ", ".join(examples_by_label[label][:5])
        print(f"  {label:12s} {c:4d}   e.g. {ex}")
    covered = sum(c for l, c in label_counts.items() if l != "OTHER")
    print(f"\ndirectly classifiable: {covered}/{n}  (OTHER = needs synthesis/object reasoning)")


if __name__ == "__main__":
    main()
