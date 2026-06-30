"""Solver library: detect a transformation from train examples, emit a tiny ONNX graph.

Each solver is (detector -> params or None, builder). `solve_task` runs them in a
priority cascade (cheapest / most-specific first) and returns the first ONNX model
that reproduces every training example exactly.

Ported and refactored from the public `conv1x1` solver notebook into a registry so
new templates are easy to add (see STRATEGY.md Phase 1).
"""
import numpy as np
import onnxruntime as ort

import onnx_ops as ops
from onnx_ops import C, GH, GW, one_hot, to_labels


# ---------------------------------------------------------------- verification
def runs_correct(model, ti, to):
    """Real correctness check (matches neurogolf_utils.run_network / verify_subset):
    threshold output at >0 and require it to EXACTLY equal the one-hot expected,
    including all-zero padding cells. NOT argmax — argmax wrongly treats empty cells
    as color 0 and would pass networks the official scorer fails.
    """
    try:
        sess = ort.InferenceSession(model.SerializeToString(),
                                    providers=["CPUExecutionProvider"])
        for t, o in zip(ti, to):
            p = (sess.run(["output"], {"input": t})[0] > 0.0).astype(float)
            if not np.array_equal(p, o):
                return False
        return True
    except Exception:
        return False


# ---------------------------------------------------------------- detectors
def detect_color_permute(ti, to):
    """Find a single color->color map consistent across all example pixels."""
    tc, oc = np.concatenate(ti), np.concatenate(to)
    mp = {}
    for e in range(tc.shape[0]):
        for h in range(GH):
            for w in range(GW):
                ic = int(np.argmax(tc[e, :, h, w]))
                oc_ = int(np.argmax(oc[e, :, h, w]))
                if oc[e, oc_, h, w] == 0:
                    continue
                if oc_ in mp and mp[oc_] != ic:
                    return None
                mp[oc_] = ic
    idx = [mp.get(i, i) for i in range(C)]
    return idx if len(set(idx)) == len(idx) else None


def detect_conv1x1(ti, to):
    """General (possibly many-to-one) color map as a 1x1 conv weight matrix."""
    tc, oc = np.concatenate(ti), np.concatenate(to)
    W = np.zeros((C, C, 1, 1), np.float32)
    # bias in (-1, 0): correct channel -> 1 + bias > 0, others -> bias < 0, padding ->
    # bias < 0. Critical: the official check is (output > 0) STRICT, so bias must NOT
    # be -1 (that makes the correct channel exactly 0 -> fails as "no color").
    B = np.full(C, -0.5, np.float32)
    for e in range(tc.shape[0]):
        for h in range(GH):
            for w in range(GW):
                ic = int(np.argmax(tc[e, :, h, w]))
                oc_ = int(np.argmax(oc[e, :, h, w]))
                if oc[e, oc_, h, w] == 0:
                    continue
                if W[oc_, ic, 0, 0] == 0:
                    W[oc_, ic, 0, 0] = 1.0
    if np.count_nonzero(W) == 0:
        return None
    # validate with the real semantics: (output > 0) must equal the one-hot expected
    for e in range(tc.shape[0]):
        p = np.einsum('oc,nchw->nohw', W[:, :, 0, 0], tc[e:e + 1]) + B[None, :, None, None]
        if not np.array_equal((p[0] > 0.0).astype(np.float32), oc[e]):
            return None
    return W, B


def detect_row_permute(ti, to):
    perms = []
    for t, o in zip(ti, to):
        ta, oa = to_labels(t), to_labels(o)
        p = []
        for h in range(GH):
            for ih in range(GH):
                if (ta[ih] == oa[h]).all():
                    p.append(ih)
                    break
            else:
                return None
        if len(set(p)) != GH:
            return None
        perms.append(p)
    return perms[0] if all(p == perms[0] for p in perms) else None


def detect_col_permute(ti, to):
    perms = []
    for t, o in zip(ti, to):
        ta, oa = to_labels(t), to_labels(o)
        p = []
        for w in range(GW):
            for iw in range(GW):
                if (ta[:, iw] == oa[:, w]).all():
                    p.append(iw)
                    break
            else:
                return None
        if len(set(p)) != GW:
            return None
        perms.append(p)
    return perms[0] if all(p == perms[0] for p in perms) else None


def detect_translation(ti, to):
    drdc = None
    for t, o in zip(ti, to):
        ta, oa = to_labels(t), to_labels(o)
        h, w = ta.shape
        found = None
        for dr in range(h):
            for dc in range(w):
                if np.array_equal(np.roll(np.roll(ta, -dr, 0), -dc, 1), oa):
                    found = (dr, dc)
                    break
            if found:
                break
        if found is None:
            return None
        if drdc is None:
            drdc = found
        elif drdc != found:
            return None
    return drdc


def detect_factorized_gather(ti, to):
    """Independent row-index + col-index remap (covers many geometric ops)."""
    hi, wi = to_labels(ti[0]).shape
    e0ti, e0to = to_labels(ti[0]), to_labels(to[0])
    ho, wo = e0to.shape
    mp = {}
    for r in range(ho):
        for c in range(wo):
            ov = int(e0to[r, c])
            cd = np.argwhere(e0ti == ov)
            if len(cd) == 0:
                return None
            best = None
            for cr, cc in cd:
                cr, cc = int(cr), int(cc)
                if all(to_labels(ti[i])[cr, cc] == to_labels(to[i])[r, c]
                       for i in range(1, len(ti))):
                    best = (cr, cc)
                    break
            mp[(r, c)] = best or (int(cd[0][0]), int(cd[0][1]))
    for r in range(ho):
        if not all(mp[(r, c)][0] == mp[(r, 0)][0] for c in range(wo)):
            return None
    for c in range(wo):
        if not all(mp[(r, c)][1] == mp[(0, c)][1] for r in range(ho)):
            return None
    ri = [mp[(r, 0)][0] for r in range(ho)] + [hi] * (GH - ho)
    ci = [mp[(0, c)][1] for c in range(wo)] + [wi] * (GW - wo)
    return ri, ci


# ---------------------------------------------------------------- content detectors
def detect_upscale(pairs):
    """Integer block upscale by a CONSTANT factor (size-agnostic builder)."""
    facs = set()
    for a, b in pairs:
        if a.size == 0 or b.shape[0] % a.shape[0] or b.shape[1] % a.shape[1]:
            return None
        ry, rx = b.shape[0] // a.shape[0], b.shape[1] // a.shape[1]
        if (ry, rx) == (1, 1) or not np.array_equal(np.repeat(np.repeat(a, ry, 0), rx, 1), b):
            return None
        facs.add((ry, rx))
    return next(iter(facs)) if len(facs) == 1 else None


def detect_tile(pairs):
    """Tile of a CONSTANT-size block by a CONSTANT factor."""
    in_shapes = {a.shape for a, _ in pairs}
    if len(in_shapes) != 1:
        return None
    facs = set()
    for a, b in pairs:
        if b.shape[0] % a.shape[0] or b.shape[1] % a.shape[1]:
            return None
        ry, rx = b.shape[0] // a.shape[0], b.shape[1] // a.shape[1]
        if (ry, rx) == (1, 1) or not np.array_equal(np.tile(a, (ry, rx)), b):
            return None
        facs.add((ry, rx))
    if len(facs) != 1:
        return None
    (IH, IW), (ry, rx) = next(iter(in_shapes)), next(iter(facs))
    return (IH, IW, ry, rx) if IH * ry <= GH and IW * rx <= GW else None


def detect_symmetrize(pairs, axis):
    """out = (input where non-background, else its mirror). bg=0. Only the MIRROR axis
    must be constant (a vertical mirror is W-agnostic, a horizontal mirror H-agnostic)."""
    if not all(a.shape == b.shape for a, b in pairs):
        return None
    dims = {a.shape[axis] for a, _ in pairs}  # axis 0 = rows (H), axis 1 = cols (W)
    if len(dims) != 1:
        return None
    for a, b in pairs:
        m = a[::-1, :] if axis == 0 else a[:, ::-1]
        if not np.array_equal(np.where(a != 0, a, m), b):
            return None
    return next(iter(dims))  # the constant mirror-axis length


def detect_const_freq(pairs, most):
    """Output is a constant-size grid filled with the most/least frequent input color."""
    outs = [b for _, b in pairs]
    if len({o.shape for o in outs}) != 1:
        return None
    for a, b in pairs:
        if len(set(b.ravel())) != 1:
            return None
        vals, cnts = np.unique(a, return_counts=True)
        pick = vals[np.argmax(cnts)] if most else vals[np.argmin(cnts)]
        if b.ravel()[0] != pick:
            return None
    return outs[0].shape  # (H, W)


# ---------------------------------------------------------------- main cascade
def solve_task(train):
    """train: list of {"input": grid, "output": grid} (lists of ints 0-9).

    Detects the transformation on the CONTENT grids (actual HxW, not the 30x30
    canvas) and emits a content-aware ONNX graph. Geometric ops that need a fixed
    size are only attempted when that size is constant across examples; the official
    gate (incl. arc-gen) is the final arbiter. Returns (model, method) or (None, None).
    """
    pairs, ti, to = [], [], []
    for eg in train:
        a = np.array(eg["input"]); b = np.array(eg["output"])
        if max(a.shape + b.shape) > GH:
            return None, None
        pairs.append((a, b))
        ti.append(one_hot(a.astype(np.float32), *a.shape))
        to.append(one_hot(b.astype(np.float32), *b.shape))
    if not pairs:
        return None, None

    same_shape = all(a.shape == b.shape for a, b in pairs)
    in_shapes = {a.shape for a, _ in pairs}
    one_size = len(in_shapes) == 1
    H, W = next(iter(in_shapes)) if one_size else (None, None)

    def try_(model, name):
        return (model, name) if runs_correct(model, ti, to) else (None, None)

    def all_(fn):
        return all(fn(a, b) for a, b in pairs)

    # 0. Identity (size-agnostic)
    if all_(lambda a, b: np.array_equal(a, b)):
        return ops.m_identity(), "identity"

    # 1. Transpose (size-agnostic on the canvas)
    if all_(lambda a, b: a.shape[::-1] == b.shape and np.array_equal(a.T, b)):
        m, n = try_(ops.m_transpose(), "transpose")
        if m: return m, n

    # 2. Color permutation (Gather axis=1) then general 1x1 Conv color map
    idx = detect_color_permute(ti, to)
    if idx is not None:
        m, n = try_(ops.m_gather1(idx, axis=1), "color_permute")
        if m: return m, n
    if same_shape:  # 1x1 conv is per-pixel -> size-agnostic, no one_size needed
        res = detect_conv1x1(ti, to)
        if res is not None:
            m, n = try_(ops.m_conv1x1(*res), "conv1x1")
            if m: return m, n

    # 3. Upscale by constant factor (size-agnostic)
    fac = detect_upscale(pairs)
    if fac is not None:
        m, n = try_(ops.m_upscale(*fac), "upscale")
        if m: return m, n

    # 4. Flips / rot180 (need a constant content size)
    if same_shape and one_size:
        if all_(lambda a, b: np.array_equal(a[:, ::-1], b)):
            m, n = try_(ops.m_flip_h(W), "flip_h")
            if m: return m, n
        if all_(lambda a, b: np.array_equal(a[::-1, :], b)):
            m, n = try_(ops.m_flip_v(H), "flip_v")
            if m: return m, n
        if all_(lambda a, b: np.array_equal(np.rot90(a, 2), b)):
            m, n = try_(ops.m_rot180(H, W), "rot180")
            if m: return m, n

    # 5. Rot90 / Rot270 (need a constant content size)
    if one_size:
        if all_(lambda a, b: np.array_equal(np.rot90(a, 1), b)):
            m, n = try_(ops.m_rot90(H, W), "rot90")
            if m: return m, n
        if all_(lambda a, b: np.array_equal(np.rot90(a, 3), b)):
            m, n = try_(ops.m_rot270(H, W), "rot270")
            if m: return m, n

    # 6. Tile (constant block + factor)
    til = detect_tile(pairs)
    if til is not None:
        m, n = try_(ops.m_tile(*til), "tile")
        if m: return m, n

    # 7. Symmetry completion (fill background by mirror; mirror-axis must be constant)
    for axis, name in [(0, "symmetrize_v"), (1, "symmetrize_h")]:
        D = detect_symmetrize(pairs, axis)
        if D is not None:
            hh, ww = (D, GW) if axis == 0 else (GH, D)
            m, n = try_(ops.m_symmetrize(hh, ww, axis), name)
            if m: return m, n

    # 8. Constant output = most/least frequent color (histogram -> extreme channel)
    for most, name in [(True, "const_most_freq"), (False, "const_least_freq")]:
        res = detect_const_freq(pairs, most)
        if res is not None:
            m, n = try_(ops.m_const_freq(most, *res), name)
            if m: return m, n

    # 9. Constant output (>=2 identical outputs; expensive dense tensor)
    outs = [b for _, b in pairs]
    if len(outs) >= 2 and all(np.array_equal(outs[0], o) for o in outs[1:]):
        m, n = try_(ops.m_const(outs[0], *outs[0].shape), "const")
        if m: return m, n

    # 8. Factorized row/col gather (catch-all geometric, needs constant size)
    if one_size and same_shape:
        res = detect_factorized_gather(ti, to)
        if res is not None:
            m, n = try_(ops.m_gather2(*res), "factorized_gather")
            if m: return m, n

    return None, None
