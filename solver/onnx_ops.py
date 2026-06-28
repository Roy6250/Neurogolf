"""Tiny ONNX graph builders for NeuroGolf 2026.

The competition feeds each task as a one-hot grid tensor of shape [1, C, GH, GW]
(C=10 colors, 30x30 grid). Color of a cell = argmax over the channel axis.
Every builder here returns a minimal onnx.ModelProto for one transformation.

Ported and cleaned from the public `conv1x1` solver notebook.
"""
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

C, GH, GW = 10, 30, 30
_OPSET = [helper.make_opsetid("", 11)]


def vi(name, dtype, *dims):
    return helper.make_tensor_value_info(name, dtype, list(dims))


def mk(nodes, inits=None, value_info=None):
    """Wrap nodes into a single-input/single-output [1,C,GH,GW] model."""
    graph = helper.make_graph(
        nodes, "g",
        [vi("input", TensorProto.FLOAT, 1, C, GH, GW)],
        [vi("output", TensorProto.FLOAT, 1, C, GH, GW)],
        initializer=inits or [],
        value_info=value_info or [],
    )
    return helper.make_model(graph, ir_version=8, opset_imports=_OPSET)


def _int64(arr, name):
    return numpy_helper.from_array(np.asarray(arr, np.int64), name)


# --- transformation builders (cheapest forms preferred: Gather > Conv > dense) ---

def m_identity():
    return mk([helper.make_node("Identity", ["input"], ["output"])])


def m_transpose():
    return mk([helper.make_node("Transpose", ["input"], ["output"], perm=[0, 1, 3, 2])])


def m_gather1(idx, axis=1):
    """Single Gather along one axis (color/row/col permutation)."""
    return mk([helper.make_node("Gather", ["input", "idx"], ["output"], axis=axis)],
              [_int64(idx, "idx")])


def m_gather2(row_idx, col_idx):
    """Factorized row+col Gather (flips, rotations, translations, factorized remap)."""
    return mk(
        [helper.make_node("Gather", ["input", "ri"], ["g0"], axis=2),
         helper.make_node("Gather", ["g0", "ci"], ["output"], axis=3)],
        [_int64(row_idx, "ri"), _int64(col_idx, "ci")],
    )


# --- content-aware builders: operate on the top-left HxW region, leave padding zero.
# A fixed graph can only do these when the relevant size/factor is constant across the
# task's examples (the official gate, incl. arc-gen, enforces that). Size-agnostic ops
# (identity, transpose, color, upscale-by-factor) work regardless of grid size.

def _rev_prefix(n, length):
    """Indices that reverse the first n entries and leave [n:length) in place."""
    return [n - 1 - i for i in range(n)] + list(range(n, length))


def m_flip_h(W):
    """Horizontal flip of Wxsomething content: single Gather(axis=3), no intermediate."""
    return m_gather1(_rev_prefix(W, GW), axis=3)


def m_flip_v(H):
    """Vertical flip: single Gather(axis=2), no intermediate."""
    return m_gather1(_rev_prefix(H, GH), axis=2)


def m_rot180(H, W):
    """180deg on HxW content = reverse first H rows then first W cols (2 Gathers)."""
    return m_gather2(_rev_prefix(H, GH), _rev_prefix(W, GW))


def m_upscale(ry, rx):
    """Integer block upscale by (ry,rx): out[i,j]=in[i//ry, j//rx]. Size-AGNOSTIC
    (depends only on the factor) — idx i//ry sends padding rows to padding."""
    return m_gather2([i // ry for i in range(GH)], [j // rx for j in range(GW)])


def m_tile(IH, IW, ry, rx):
    """Tile an IHxIW block ry x rx times: out[i,j]=in[i%IH,j%IW] inside the tiled
    region, 0 outside (indices past the region point at a known-zero padding line)."""
    row = [(i % IH) if i < IH * ry else IH for i in range(GH)]
    col = [(j % IW) if j < IW * rx else IW for j in range(GW)]
    return m_gather2(row, col)


def m_conv1x1(W, B):
    """1x1 Conv color mapping. Use only when not a pure permutation (else Gather)."""
    return mk(
        [helper.make_node("Conv", ["input", "W", "B"], ["output"], kernel_shape=[1, 1])],
        [numpy_helper.from_array(W.astype(np.float32), "W"),
         numpy_helper.from_array(B.astype(np.float32), "B")],
    )


def m_const(label_grid, H, W):
    """Constant output. EXPENSIVE (one-hot tensor up to C*GH*GW). Last resort."""
    c = np.zeros((1, C, GH, GW), dtype=np.float32)
    for r in range(H):
        for w in range(W):
            v = int(label_grid[r, w])
            if 0 <= v < C:
                c[0, v, r, w] = 1.0
    return mk(
        [helper.make_node("Mul", ["input", "z"], ["zd"]),
         helper.make_node("ReduceSum", ["zd"], ["s"], axes=[1, 2, 3], keepdims=1),
         helper.make_node("Add", ["s", "c"], ["output"])],
        [numpy_helper.from_array(np.array(0.0, np.float32), "z"),
         numpy_helper.from_array(c, "c")],
    )


def m_crop(dr, dc, ho, wo):
    return mk(
        [helper.make_node("Slice", ["input", "st", "en"], ["a"]),
         helper.make_node("Pad", ["a", "pa"], ["output"], mode="constant")],
        [_int64([0, 0, dr, dc], "st"),
         _int64([1, C, dr + ho, dc + wo], "en"),
         _int64([0, 0, 0, 0, 0, 0, GH - ho, GW - wo], "pa")],
    )


def m_translate(hi, wi, dr, dc):
    ri = [(r - dr) % hi for r in range(hi)] + list(range(hi, GH))
    ci = [(c - dc) % wi for c in range(wi)] + list(range(wi, GW))
    return m_gather2(ri, ci)


def m_rot90(H, W):
    """90deg CCW on HxW content = Transpose then reverse the first W rows.

    A rotation mixes row/col so it cannot be two independent Gathers (that yields
    only flips); it is Transpose (size-agnostic) + a content-aware row flip. The
    transposed content is WxH, so we reverse its first W rows."""
    return mk(
        [helper.make_node("Transpose", ["input"], ["t"], perm=[0, 1, 3, 2]),
         helper.make_node("Gather", ["t", "ri"], ["output"], axis=2)],
        [_int64(_rev_prefix(W, GH), "ri")],
    )


def m_rot270(H, W):
    """90deg CW on HxW content = Transpose then reverse the first H columns."""
    return mk(
        [helper.make_node("Transpose", ["input"], ["t"], perm=[0, 1, 3, 2]),
         helper.make_node("Gather", ["t", "ci"], ["output"], axis=3)],
        [_int64(_rev_prefix(H, GW), "ci")],
    )


# --- helpers ---

def one_hot(grid, H, W):
    """ARC label grid -> [1, C, GH, GW] one-hot, zero-padded beyond HxW."""
    r = np.zeros((1, C, GH, GW), dtype=np.float32)
    for h in range(H):
        for w in range(W):
            v = int(grid[h, w])
            if 0 <= v < C:
                r[0, v, h, w] = 1.0
    return r


def to_labels(t):
    """[1, C, GH, GW] one-hot -> [GH, GW] label grid via channel argmax."""
    return np.argmax(t[0], axis=0)
