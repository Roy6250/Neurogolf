"""Extended diagnostic: detect richer transformation families on the CONTENT grids,
to find which of the 383 "OTHER" tasks are reachable with composable primitives.
Analysis only (numpy) — guides which ONNX programs to build next.

  ./.venv/bin/python solver/analyze2.py
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "Dataset"


def cc_label(mask):
    """4-connected connected components of a boolean mask -> int label grid (0=bg)."""
    H, W = mask.shape
    lab = np.zeros((H, W), int)
    cur = 0
    for i in range(H):
        for j in range(W):
            if mask[i, j] and lab[i, j] == 0:
                cur += 1
                stack = [(i, j)]
                lab[i, j] = cur
                while stack:
                    y, x = stack.pop()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and lab[ny, nx] == 0:
                            lab[ny, nx] = cur
                            stack.append((ny, nx))
    return lab, cur


def bg_color(g):
    """Most common color = assumed background."""
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(cnts)])


def detect(pairs):
    def all_(fn):
        try:
            return all(fn(a, b) for a, b in pairs)
        except Exception:
            return False

    # --- crop to bounding box of non-background ---
    def bbox_crop(a, b):
        bg = bg_color(a)
        ys, xs = np.where(a != bg)
        if len(ys) == 0:
            return False
        sub = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        return sub.shape == b.shape and np.array_equal(sub, b)
    if all_(bbox_crop):
        return "bbox_crop"

    # --- constant output = most / least frequent color ---
    def const_freq(a, b, most):
        if len(set(b.ravel())) != 1:
            return False
        vals, cnts = np.unique(a, return_counts=True)
        pick = vals[np.argmax(cnts)] if most else vals[np.argmin(cnts)]
        return b.ravel()[0] == pick
    if all(len(set(b.ravel())) == 1 for _, b in pairs):
        if all_(lambda a, b: const_freq(a, b, True)):
            return "const_most_freq"
        if all_(lambda a, b: const_freq(a, b, False)):
            return "const_least_freq"

    # --- dedup adjacent duplicate rows / cols (compress) ---
    def dedup_rows(a, b):
        keep = [0] + [i for i in range(1, a.shape[0]) if not np.array_equal(a[i], a[i - 1])]
        return np.array_equal(a[keep], b)
    def dedup_cols(a, b):
        keep = [0] + [j for j in range(1, a.shape[1]) if not np.array_equal(a[:, j], a[:, j - 1])]
        return np.array_equal(a[:, keep], b)
    if all_(dedup_rows):
        return "dedup_rows"
    if all_(dedup_cols):
        return "dedup_cols"

    # --- symmetry completion (fill bg by mirror) ---
    def symm(a, b, axis):
        if a.shape != b.shape:
            return False
        bg = bg_color(a)
        m = a[:, ::-1] if axis == 1 else a[::-1, :]
        out = np.where(a != bg, a, m)
        return np.array_equal(out, b)
    if all_(lambda a, b: symm(a, b, 1)):
        return "symmetrize_h"
    if all_(lambda a, b: symm(a, b, 0)):
        return "symmetrize_v"

    # --- gravity: non-bg cells fall to the bottom of each column ---
    def gravity(a, b):
        if a.shape != b.shape:
            return False
        bg = bg_color(a)
        out = np.full_like(a, bg)
        for c in range(a.shape[1]):
            col = a[:, c]
            nz = col[col != bg]
            out[a.shape[0] - len(nz):, c] = nz
        return np.array_equal(out, b)
    if all_(gravity):
        return "gravity_down"

    # --- keep only the largest object (others -> bg) ---
    def keep_largest(a, b):
        if a.shape != b.shape:
            return False
        bg = bg_color(a)
        lab, n = cc_label(a != bg)
        if n == 0:
            return False
        sizes = [(lab == k).sum() for k in range(1, n + 1)]
        big = 1 + int(np.argmax(sizes))
        out = np.where(lab == big, a, bg)
        return np.array_equal(out, b)
    if all_(keep_largest):
        return "keep_largest_object"

    # --- output = color of the largest object, as a constant grid ---
    def color_of_largest(a, b):
        if len(set(b.ravel())) != 1:
            return False
        bg = bg_color(a)
        lab, n = cc_label(a != bg)
        if n == 0:
            return False
        sizes = [(lab == k).sum() for k in range(1, n + 1)]
        big = 1 + int(np.argmax(sizes))
        col = a[lab == big]
        return len(set(col)) == 1 and b.ravel()[0] == col[0]
    if all_(color_of_largest):
        return "color_of_largest"

    return None


def main():
    counts = Counter()
    by_label = defaultdict(list)
    for i in range(1, 401):
        tf = DATA / f"task{i:03d}.json"
        if not tf.exists():
            continue
        task = json.loads(tf.read_text())
        pairs = [(np.array(e["input"]), np.array(e["output"])) for e in task.get("train", [])]
        if not pairs:
            continue
        lab = detect(pairs) or "still_OTHER"
        counts[lab] += 1
        by_label[lab].append(f"task{i:03d}")

    print("extended families (content grids, all train pairs):")
    for lab, c in counts.most_common():
        if lab == "still_OTHER":
            continue
        print(f"  {lab:22s} {c:3d}   e.g. {', '.join(by_label[lab][:6])}")
    reach = sum(c for l, c in counts.items() if l != "still_OTHER")
    print(f"\nnewly reachable: {reach}   still OTHER: {counts['still_OTHER']}")


if __name__ == "__main__":
    main()
