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


def m_flip_h():
    """Horizontal flip = single Slice reversing axis 3 (steps=-1). One node to
    `output`, no intermediate, only ~4 tiny index params."""
    return mk(
        [helper.make_node("Slice", ["input", "st", "en", "ax", "sp"], ["output"])],
        [_int64([GW - 1], "st"), _int64([-GW - 1], "en"), _int64([3], "ax"), _int64([-1], "sp")],
    )


def m_flip_v():
    """Vertical flip = single Slice reversing axis 2 (steps=-1). No intermediate."""
    return mk(
        [helper.make_node("Slice", ["input", "st", "en", "ax", "sp"], ["output"])],
        [_int64([GH - 1], "st"), _int64([-GH - 1], "en"), _int64([2], "ax"), _int64([-1], "sp")],
    )


def m_rot180():
    """180deg = single Slice reversing both spatial axes (steps=-1). One node, no
    intermediate; index tensors are tiny (8 params total) vs a 2-Gather form whose
    intermediate [1,10,30,30] float32 tensor would cost ~48 KB of 'memory'."""
    return mk(
        [helper.make_node("Slice", ["input", "st", "en", "ax", "sp"], ["output"])],
        [_int64([GH - 1, GW - 1], "st"), _int64([-GH - 1, -GW - 1], "en"),
         _int64([2, 3], "ax"), _int64([-1, -1], "sp")],
    )


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


def m_tile(hi, wi, rH, rW):
    oh_, ow = hi * rH, wi * rW
    return mk(
        [helper.make_node("Slice", ["input", "st", "en"], ["cr"]),
         helper.make_node("Tile", ["cr", "rp"], ["tl"]),
         helper.make_node("Pad", ["tl", "pa"], ["output"], mode="constant")],
        [_int64([0, 0, 0, 0], "st"),
         _int64([1, C, hi, wi], "en"),
         _int64([1, 1, rH, rW], "rp"),
         _int64([0, 0, 0, 0, 0, 0, GH - oh_, GW - ow], "pa")],
    )


def m_translate(hi, wi, dr, dc):
    ri = [(r - dr) % hi for r in range(hi)] + list(range(hi, GH))
    ci = [(c - dc) % wi for c in range(wi)] + list(range(wi, GW))
    return m_gather2(ri, ci)


def m_rot90(hi=GH, wi=GW):
    """90deg CCW on the full square canvas = flip_v(transpose).

    NOTE: a rotation mixes row/col, so it CANNOT be two independent Gathers
    (that only yields flips). Composed as Transpose then row-reverse Gather.
    Valid for the square 30x30 canvas; non-square content needs real harness
    semantics (Unknown #3).
    """
    return mk(
        [helper.make_node("Transpose", ["input"], ["t"], perm=[0, 1, 3, 2]),
         helper.make_node("Gather", ["t", "ri"], ["output"], axis=2)],
        [_int64(list(range(GH - 1, -1, -1)), "ri")],
    )


def m_rot270(hi=GH, wi=GW):
    """90deg CW on the full square canvas = flip_h(transpose)."""
    return mk(
        [helper.make_node("Transpose", ["input"], ["t"], perm=[0, 1, 3, 2]),
         helper.make_node("Gather", ["t", "ci"], ["output"], axis=3)],
        [_int64(list(range(GW - 1, -1, -1)), "ci")],
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
