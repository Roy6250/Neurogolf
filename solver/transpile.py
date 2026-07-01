"""Prototype arc-dsl -> ONNX transpiler.

Parses each matched solve_<hash>'s AST and, when it is a linear chain of WHITELISTED
grid->grid primitives, emits an ONNX graph by SPLICING our existing single-in/single-out
[1,10,30,30] builders. Size-dependent primitives (mirrors/rot/up/downscale) read the task's
constant content size; non-constant or unsupported solvers are skipped. The official gate is
the arbiter. Adding a primitive handler => more tasks, automatically.

  ./.venv/bin/python solver/transpile.py [N]
"""
import ast
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import helper

import onnx_ops as ops
from onnx_ops import C, GH, GW
import scorer

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "Dataset"
ARC_DSL = Path("/private/tmp/claude-501/-Users-ananyaroy-sayantan/"
               "0aaa9c77-4c01-4eec-b934-8bb413b54cbc/scratchpad/arc-dsl")


# ---------------------------------------------------------------- graph splice
def splice(models):
    """Chain single-in/single-out [1,10,30,30] models input -> m0 -> m1 -> ... -> output."""
    nodes, inits = [], []
    prev = "input"
    for i, m in enumerate(models):
        g = m.graph
        last = i == len(models) - 1

        def rn(name):
            if name == "input":
                return prev
            if name == "output":
                return "output" if last else f"t{i}"
            return f"s{i}_{name}"

        for init in g.initializer:
            ni = onnx.TensorProto(); ni.CopyFrom(init); ni.name = rn(init.name)
            inits.append(ni)
        for node in g.node:
            nn = onnx.NodeProto(); nn.CopyFrom(node)
            nn.input[:] = [rn(x) if x else x for x in node.input]
            nn.output[:] = [rn(x) for x in node.output]
            nn.name = nn.output[0]
            nodes.append(nn)
        prev = f"t{i}"
    graph = helper.make_graph(nodes, "g",
                              [ops.vi("input", onnx.TensorProto.FLOAT, 1, C, GH, GW)],
                              [ops.vi("output", onnx.TensorProto.FLOAT, 1, C, GH, GW)],
                              initializer=inits)
    m = helper.make_model(graph, ir_version=8,
                          opset_imports=[helper.make_opsetid("", 11)])
    onnx.checker.check_model(m)
    return m


# ---------------------------------------------------------------- primitive handlers
# Each returns a single-in/single-out builder model. ctx carries the task's (H, W) when
# the input size is constant (None otherwise). Handlers return None if not expressible.
def _downscale(f):
    row = [min(i * f, GH - 1) for i in range(GH)]
    col = [min(j * f, GW - 1) for j in range(GW)]
    return ops.m_gather2(row, col)


def _switch(a, b):
    idx = list(range(C)); idx[a], idx[b] = b, a
    return ops.m_gather1(idx, axis=1)


def _replace(a, b):
    W = np.zeros((C, C, 1, 1), np.float32)
    for c in range(C):
        W[c, c, 0, 0] = 1.0
    W[a, a, 0, 0] = 0.0          # color a no longer maps to itself
    W[b, a, 0, 0] = 1.0          # ... it maps to b instead
    return ops.m_conv1x1(W, np.full(C, -0.5, np.float32))


def handler(call, ctx):
    """call: (name, [int args]). Returns a builder model or None if unsupported."""
    name, args = call
    H, W = ctx
    size_ok = H is not None
    if name == "dmirror":
        return ops.m_transpose()
    if name == "rot180":
        return ops.m_rot180(H, W) if size_ok else None
    if name in ("vmirror",) and size_ok:        # mirror over vertical axis = flip columns
        return ops.m_flip_h(W)
    if name in ("hmirror",) and size_ok:         # mirror over horizontal axis = flip rows
        return ops.m_flip_v(H)
    if name == "rot90" and size_ok:
        return ops.m_rot90(H, W)
    if name == "rot270" and size_ok:
        return ops.m_rot270(H, W)
    if name == "upscale" and args:
        return ops.m_upscale(args[0], args[0])
    if name == "downscale" and args:
        return _downscale(args[0])
    if name == "switch" and len(args) >= 2:
        return _switch(args[0], args[1])
    if name == "replace" and len(args) >= 2:
        return _replace(args[0], args[1])
    return None


# arc-dsl integer constants we need to resolve (constants.py uses names like THREE)
_CONST = {f: i for i, f in enumerate(
    ["ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE", "TEN"])}


def parse_linear(fn_node):
    """Return an ordered list of (prim_name, [int_args]) if the solver is a simple linear
    chain `x = f(prev, lits...)`, else None."""
    steps, last = [], "I"
    for stmt in fn_node.body:
        if isinstance(stmt, ast.Return):
            ret = stmt.value.id if isinstance(stmt.value, ast.Name) else None
            return steps if ret == last else None
        if not (isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call)):
            return None
        call = stmt.value
        if not isinstance(call.func, ast.Name):
            return None
        target = stmt.targets[0].id
        # first positional arg must be the running grid (prev variable or I)
        if not call.args or not isinstance(call.args[0], ast.Name) or call.args[0].id != last:
            return None
        lits = []
        for a in call.args[1:]:
            if isinstance(a, ast.Name) and a.id in _CONST:
                lits.append(_CONST[a.id])
            else:
                return None   # non-scalar arg (vector/tuple/function) -> unsupported here
        steps.append((call.func.id, lits))
        last = target
    return None


def main(limit=400):
    matched = json.loads((REPO / "arcdsl_matched.json").read_text())
    src = (ARC_DSL / "solvers.py").read_text()
    tree = ast.parse(src)
    bodies = {n.name[len("solve_"):]: n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name.startswith("solve_")}

    OUT = REPO / "models_transpiled"; OUT.mkdir(exist_ok=True)
    banked = Counter()
    total = 0.0
    for tn, h in sorted(matched.items())[:limit]:
        steps = parse_linear(bodies[h])
        if steps is None:
            continue
        task = json.loads((DATA / f"{tn}.json").read_text())
        shapes = {np.array(e["input"]).shape for e in task["train"]}
        ctx = next(iter(shapes)) if len(shapes) == 1 else (None, None)
        models = [handler(s, ctx) for s in steps]
        if any(m is None for m in models):
            continue
        try:
            graph = splice(models)
        except Exception:
            continue
        ok, right, wrong = scorer.verify(graph, task)
        frac = right / (right + wrong) if (right + wrong) else 0.0
        if frac > 0:
            pts, mem, par = scorer.score(graph)
            total += pts * frac
            banked[tuple(s[0] for s in steps)] += 1
            onnx.save(graph, str(OUT / f"{tn}.onnx"))
            tag = "FULL" if ok else f"~{frac:.2f}"
            print(f"  {tn} [{h}] {tag:6s} est={pts*frac:6.2f}  "
                  f"{' -> '.join(s[0] for s in steps)}", flush=True)

    print(f"\ntranspiled+banked {sum(banked.values())} tasks  est. {total:.1f} pts")
    print("programs:", dict(banked))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 400)
