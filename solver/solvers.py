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
    B = np.full(C, -1.0, np.float32)
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
    for e in range(tc.shape[0]):
        p = np.einsum('oc,nchw->nohw', W[:, :, 0, 0], tc[e:e + 1]) + B[None, :, None, None]
        if (np.argmax(p[0], axis=0) != np.argmax(oc[e], axis=0)).any():
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


# ---------------------------------------------------------------- main cascade
def solve_task(train):
    """train: list of {"input": grid, "output": grid} (python lists of ints).

    Returns (model, method_name) or (None, None) if no template matched.
    Grids larger than the 30x30 canvas are rejected (caller should fall back).
    """
    ti, to, shapes = [], [], []
    for eg in train:
        ia = np.array(eg["input"], np.float32)
        oa = np.array(eg["output"], np.float32)
        hi, wi, ho, wo = ia.shape[0], ia.shape[1], oa.shape[0], oa.shape[1]
        if max(hi, ho) > GH or max(wi, wo) > GW:
            return None, None
        ti.append(one_hot(ia, hi, wi))
        to.append(one_hot(oa, ho, wo))
        shapes.append((hi, wi, ho, wo))
    if not ti:
        return None, None

    same_inout = all(h == ho and w == wo for (h, w, ho, wo) in shapes)
    same_size = len(set(shapes)) == 1
    Hi = max(s[0] for s in shapes); Wi = max(s[1] for s in shapes)
    Ho = max(s[2] for s in shapes); Wo = max(s[3] for s in shapes)

    def try_(model, name):
        return (model, name) if runs_correct(model, ti, to) else (None, None)

    # 0. Identity
    if all((a == b).all() for a, b in zip(ti, to)):
        return ops.m_identity(), "identity"

    # 1. Transpose
    if same_inout and all(h == wo and w == ho for (h, w, ho, wo) in shapes):
        m, n = try_(ops.m_transpose(), "transpose")
        if m: return m, n

    # 2. Color permutation (Gather axis=1) — cheapest color op
    idx = detect_color_permute(ti, to)
    if idx is not None:
        m, n = try_(ops.m_gather1(idx, axis=1), "color_permute")
        if m: return m, n

    # 3. Conv1x1 color mapping (general map)
    if same_inout and same_size:
        res = detect_conv1x1(ti, to)
        if res is not None:
            m, n = try_(ops.m_conv1x1(*res), "conv1x1")
            if m: return m, n

    # 4. Constant output (expensive — only if >=2 outputs and all identical, else
    #    a single example would just be memorized and fail the generalization gate)
    outs = [to_labels(t) for t in to]
    if len(outs) >= 2 and all(np.array_equal(outs[0], o) for o in outs[1:]):
        m, n = try_(ops.m_const(outs[0], *outs[0].shape), "const")
        if m: return m, n

    # 5. Flip H / Flip V / Rot180 (single-node builders -> no intermediate tensor)
    if all(np.array_equal(to_labels(t)[:, ::-1], to_labels(o)) for t, o in zip(ti, to)):
        m, n = try_(ops.m_flip_h(), "flip_h")
        if m: return m, n
    if all(np.array_equal(to_labels(t)[::-1, :], to_labels(o)) for t, o in zip(ti, to)):
        m, n = try_(ops.m_flip_v(), "flip_v")
        if m: return m, n
    if all(np.array_equal(np.rot90(to_labels(t), 2), to_labels(o)) for t, o in zip(ti, to)):
        m, n = try_(ops.m_rot180(), "rot180")
        if m: return m, n

    # 6. Rot90 / Rot270
    if same_inout and same_size and Hi == Wo and Wi == Ho:
        for ang, maker, name in [(1, ops.m_rot90, "rot90"), (3, ops.m_rot270, "rot270")]:
            if all(np.array_equal(np.rot90(to_labels(t), ang), to_labels(o))
                   for t, o in zip(ti, to)):
                m, n = try_(maker(Hi, Wi), name)
                if m: return m, n

    # 7. Row permute
    p = detect_row_permute(ti, to)
    if p is not None:
        m, n = try_(ops.m_gather1(p, axis=2), "row_permute")
        if m: return m, n

    # 8. Column permute
    p = detect_col_permute(ti, to)
    if p is not None:
        m, n = try_(ops.m_gather1(p, axis=3), "col_permute")
        if m: return m, n

    # 9. Crop (centered)
    if same_inout is False and same_size:
        hi, wi, ho, wo = shapes[0]
        if ho < hi or wo < wi:
            dr, dc = (hi - ho) // 2, (wi - wo) // 2
            if all(np.array_equal(to_labels(o), to_labels(t)[dr:dr + ho, dc:dc + wo])
                   for t, o in zip(ti, to)):
                m, n = try_(ops.m_crop(dr, dc, ho, wo), "crop")
                if m: return m, n

    # 10. Translation (roll)
    if same_inout and same_size and Hi <= 10 and Wi <= 10:
        res = detect_translation(ti, to)
        if res and (res[0] or res[1]):
            m, n = try_(ops.m_translate(Hi, Wi, *res), "translate")
            if m: return m, n

    # 11. Tile
    ish = set(to_labels(t).shape for t in ti)
    if len(ish) == 1:
        IH, IW = next(iter(ish))
        ratios = set()
        for t, o in zip(ti, to):
            OHH, OWW = to_labels(o).shape
            if OHH % IH == 0 and OWW % IW == 0:
                rH, rW = OHH // IH, OWW // IW
                if not (rH == 1 and rW == 1):
                    ratios.add((rH, rW))
        if len(ratios) == 1:
            rH, rW = next(iter(ratios))
            if IH * rH <= GH and IW * rW <= GW and all(
                    np.array_equal(np.tile(to_labels(t)[:IH, :IW], (rH, rW)), to_labels(o))
                    for t, o in zip(ti, to)):
                m, n = try_(ops.m_tile(IH, IW, rH, rW), "tile")
                if m: return m, n

    # 12. Factorized gather (catch-all geometric)
    if same_size:
        res = detect_factorized_gather(ti, to)
        if res is not None:
            m, n = try_(ops.m_gather2(*res), "factorized_gather")
            if m: return m, n

    return None, None
