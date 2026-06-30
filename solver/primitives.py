"""Foundational opset-10/11 ONNX primitives for the object-reasoning tail.

Two load-bearing graphs, each verified against a numpy reference:

  compress_rows  - drop a data-dependent subset of rows, stack survivors top, static shape
                   (triangular-MatMul prefix sum + permutation-MatMul compaction; no NonZero)
  cc_label       - 4-connected connected components via unrolled neighbour-max propagation

These operate on a single 2-D plane [H,W] (a color/mask channel). Integration with the
one-hot [1,10,30,30] competition tensor (per-channel) comes on top of these.

  ./.venv/bin/python solver/primitives.py
"""
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort

OPSET = 11


def _f(arr, name):
    return numpy_helper.from_array(np.asarray(arr, np.float32), name)


def _i(arr, name):
    return numpy_helper.from_array(np.asarray(arr, np.int64), name)


def make_model(nodes, inputs, outputs, inits):
    g = helper.make_graph(nodes, "g", inputs, outputs, inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", OPSET)])
    onnx.checker.check_model(m)
    return m


def run(model, **feeds):
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run(None, feeds)[0]


# --------------------------------------------------------------- compress_rows
def compress_rows_model(H, W):
    """Drop UNIFORM rows (max==min); stack the rest at the top, zeros below. Static [H,W]."""
    L = np.tril(np.ones((H, H), np.float32), -1)       # strict lower tri: L[i,j]=1 iff j<i
    aar = np.arange(H, dtype=np.float32).reshape(H, 1)  # [H,1] = [0,1,...,H-1]
    nodes = [
        # keep[i] = 1 iff row i is non-uniform
        helper.make_node("ReduceMax", ["X"], ["rmax"], axes=[1], keepdims=1),     # [H,1]
        helper.make_node("ReduceMin", ["X"], ["rmin"], axes=[1], keepdims=1),
        helper.make_node("Sub", ["rmax", "rmin"], ["rng"]),
        helper.make_node("Greater", ["rng", "half"], ["kb"]),
        helper.make_node("Cast", ["kb"], ["keep"], to=TensorProto.FLOAT),         # [H,1]
        # dest[i] = #kept rows strictly before i  (prefix sum as triangular MatMul)
        helper.make_node("MatMul", ["L", "keep"], ["dest"]),                      # [H,1]
        helper.make_node("Reshape", ["dest", "sh_1H"], ["dest_1H"]),              # [1,H]
        # R[a,i] = (a == dest[i]) * keep[i]  (permutation/scatter matrix)
        helper.make_node("Sub", ["aar", "dest_1H"], ["diff"]),                    # [H,H]
        helper.make_node("Abs", ["diff"], ["adiff"]),
        helper.make_node("Less", ["adiff", "half"], ["eqb"]),
        helper.make_node("Cast", ["eqb"], ["eqf"], to=TensorProto.FLOAT),
        helper.make_node("Reshape", ["keep", "sh_1H"], ["keep_1H"]),              # [1,H]
        helper.make_node("Mul", ["eqf", "keep_1H"], ["R"]),                       # [H,H]
        # apply: pull kept rows to the top
        helper.make_node("MatMul", ["R", "X"], ["Y"]),                            # [H,W]
    ]
    inits = [_f(L, "L"), _f(aar, "aar"), _f([0.5], "half"), _i([1, H], "sh_1H")]
    return make_model(
        nodes,
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [H, W])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [H, W])],
        inits,
    )


def compress_rows_numpy(X):
    keep = [i for i in range(X.shape[0]) if X[i].max() != X[i].min()]
    out = np.zeros_like(X)
    out[:len(keep)] = X[keep]
    return out


# --------------------------------------------------------------- cc_label
def cc_label_model(H, W):
    """4-connected connected components of a 0/1 mask -> label grid (each component = its
    max seed id). Unrolled neighbour-max propagation for T=H+W rounds, re-masked each round."""
    seed = (np.arange(H * W, dtype=np.float32) + 1).reshape(H, W)
    up = [max(i - 1, 0) for i in range(H)]
    down = [min(i + 1, H - 1) for i in range(H)]
    left = [max(j - 1, 0) for j in range(W)]
    right = [min(j + 1, W - 1) for j in range(W)]
    inits = [_f(seed, "seed"),
             _i(up, "iu"), _i(down, "id"), _i(left, "il"), _i(right, "ir")]
    nodes = [helper.make_node("Mul", ["M", "seed"], ["L0"])]
    cur = "L0"
    for t in range(H + W):
        u, d, le, ri, nx, L1 = (f"u{t}", f"d{t}", f"le{t}", f"ri{t}", f"nx{t}", f"L{t+1}")
        nodes += [
            helper.make_node("Gather", [cur, "iu"], [u], axis=0),
            helper.make_node("Gather", [cur, "id"], [d], axis=0),
            helper.make_node("Gather", [cur, "il"], [le], axis=1),
            helper.make_node("Gather", [cur, "ir"], [ri], axis=1),
            helper.make_node("Max", [cur, u, d, le, ri], [nx]),
            helper.make_node("Mul", [nx, "M"], [L1]),
        ]
        cur = L1
    nodes.append(helper.make_node("Identity", [cur], ["Y"]))
    return make_model(
        nodes,
        [helper.make_tensor_value_info("M", TensorProto.FLOAT, [H, W])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [H, W])],
        inits,
    )


def cc_label_numpy(mask):
    H, W = mask.shape
    lab = np.zeros((H, W), int)
    cur = 0
    for i in range(H):
        for j in range(W):
            if mask[i, j] and lab[i, j] == 0:
                cur += 1
                st = [(i, j)]; lab[i, j] = cur
                while st:
                    y, x = st.pop()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and lab[ny, nx] == 0:
                            lab[ny, nx] = cur; st.append((ny, nx))
    return lab


def same_partition(a, b):
    """True iff a and b induce the same partition on nonzero cells (label values may differ)."""
    a, b = np.asarray(a), np.asarray(b)
    if (a > 0).any() != (b > 0).any():
        return False
    # map each grid's labels to canonical ids by first-occurrence, then compare
    def canon(x):
        out = np.zeros_like(x, int); nxt = {}; k = 0
        for idx, v in np.ndenumerate(x):
            v = int(round(float(v)))
            if v == 0:
                continue
            if v not in nxt:
                k += 1; nxt[v] = k
            out[idx] = nxt[v]
        return out
    return np.array_equal(canon(a), canon(b))


# --------------------------------------------------------------- verify
def main():
    rng = np.random.default_rng(0)
    ok = True

    # compress_rows
    cp = 0
    for _ in range(20):
        H, W = int(rng.integers(3, 9)), int(rng.integers(3, 9))
        X = rng.integers(0, 4, (H, W)).astype(np.float32)
        # force a few uniform rows
        for r in rng.choice(H, size=rng.integers(0, H), replace=False):
            X[r] = rng.integers(0, 4)
        got = run(compress_rows_model(H, W), X=X)
        cp += np.array_equal(got, compress_rows_numpy(X))
    print(f"compress_rows: {cp}/20 exact vs numpy")
    ok &= cp == 20

    # cc_label
    cc = 0
    for _ in range(20):
        H, W = int(rng.integers(4, 10)), int(rng.integers(4, 10))
        M = (rng.random((H, W)) < 0.5).astype(np.float32)
        got = run(cc_label_model(H, W), M=M)
        cc += same_partition(got, cc_label_numpy(M.astype(int)))
    print(f"cc_label:      {cc}/20 same partition vs numpy")
    ok &= cc == 20

    print("ALL PRIMITIVES VERIFIED" if ok else "FAILURES PRESENT")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
