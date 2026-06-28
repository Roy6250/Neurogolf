"""Behavior-preserving graph-shrinking passes (the 'golf' step).

Safe subset ported from the public 'graph surgeries' notebook. Each pass returns a
new model; `apply_all` chains them and re-checks the model. The best golf is choosing
cheap ops during synthesis (see solvers.py) — these passes are the safety net.

Not included here (add later if scoring justifies): FP16 conversion and onnxsim/
onnxoptimizer, which need extra deps and re-verification against examples.
"""
import copy
from collections import defaultdict

import numpy as np
import onnx
from onnx import numpy_helper, helper, TensorProto


def prune_unused_initializers(model):
    g = model.graph
    used = {inp for node in g.node for inp in node.input if inp}
    keep = [i for i in g.initializer if i.name in used]
    removed = len(g.initializer) - len(keep)
    del g.initializer[:]
    g.initializer.extend(keep)
    return removed


def _init_key(init):
    a = numpy_helper.to_array(init)
    return (a.dtype.str, a.shape, a.tobytes())


def deduplicate_initializers(model):
    g = model.graph
    groups = defaultdict(list)
    for init in g.initializer:
        groups[_init_key(init)].append(init.name)
    replace = {}
    for names in groups.values():
        if len(names) <= 1:
            continue
        canon = sorted(names, key=lambda s: (len(s), s))[0]
        for n in names:
            if n != canon:
                replace[n] = canon
    if not replace:
        return 0
    for node in g.node:
        for i, n in enumerate(node.input):
            if n in replace:
                node.input[i] = replace[n]
    prune_unused_initializers(model)
    return len(replace)


def eliminate_identity(model):
    g = model.graph
    outs = {o.name for o in g.output}
    repl = {}
    for node in g.node:
        if node.op_type == "Identity" and len(node.input) == 1 and node.output[0] not in outs:
            repl[node.output[0]] = node.input[0]

    def resolve(n):
        seen = set()
        while n in repl and n not in seen:
            seen.add(n); n = repl[n]
        return n

    if not repl:
        return 0
    kept = []
    for node in g.node:
        for i, n in enumerate(node.input):
            if n in repl:
                node.input[i] = resolve(n)
        if node.op_type == "Identity" and node.output[0] in repl:
            continue
        kept.append(node)
    removed = len(g.node) - len(kept)
    del g.node[:]
    g.node.extend(kept)
    return removed


# int64 index tensors that are safe to downcast to int32
_INT64_REQUIRED = {("Reshape", 1), ("Slice", 1), ("Slice", 2), ("Slice", 3), ("Slice", 4),
                   ("Pad", 1), ("Tile", 1), ("Expand", 1), ("GatherND", 1), ("ScatterND", 1),
                   ("Squeeze", 1), ("Unsqueeze", 1)}
_INT32_SAFE = {("Gather", 1), ("GatherElements", 1), ("ScatterElements", 1), ("OneHot", 0),
               ("Add", 0), ("Add", 1), ("Sub", 0), ("Sub", 1), ("Mul", 0), ("Mul", 1),
               ("Equal", 0), ("Equal", 1), ("Concat", 0), ("Concat", 1), ("Where", 1), ("Where", 2)}


def narrow_int32(model):
    g = model.graph
    consumers = defaultdict(list)
    for node in g.node:
        for pos, n in enumerate(node.input):
            if n:
                consumers[n].append((node.op_type, pos))
    converted = 0
    for i, init in enumerate(g.initializer):
        a = numpy_helper.to_array(init)
        if a.dtype != np.int64 or a.size == 0:
            continue
        if a.min() < np.iinfo(np.int32).min or a.max() > np.iinfo(np.int32).max:
            continue
        cs = consumers.get(init.name, [])
        if not cs or any(c in _INT64_REQUIRED or c not in _INT32_SAFE for c in cs):
            continue
        g.initializer[i].CopyFrom(numpy_helper.from_array(a.astype(np.int32), init.name))
        converted += 1
    return converted


def conv1x1_perm_to_gather(model):
    """Replace a 1x1 Conv that is a pure channel permutation with Gather(axis=1)."""
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    new_nodes, drop = [], set()
    converted = 0
    for node in g.node:
        if node.op_type != "Conv" or len(node.input) < 2 or node.input[1] not in inits:
            new_nodes.append(node)
            continue
        W = numpy_helper.to_array(inits[node.input[1]])
        if W.shape != (10, 10, 1, 1):
            new_nodes.append(node)
            continue
        M = W[:, :, 0, 0]
        if not (np.all(np.isclose(M, 0) | np.isclose(M, 1)) and np.all(M.sum(axis=1) == 1)):
            new_nodes.append(node)
            continue
        # bias (if any) must be zero
        if len(node.input) >= 3 and node.input[2]:
            b = inits.get(node.input[2])
            if b is None or not np.allclose(numpy_helper.to_array(b), 0):
                new_nodes.append(node)
                continue
            drop.add(node.input[2])
        idx = np.argmax(M, axis=1).astype(np.int64)
        idx_name = node.output[0] + "_gidx"
        g.initializer.append(numpy_helper.from_array(idx, idx_name))
        new_nodes.append(helper.make_node("Gather", [node.input[0], idx_name],
                                          list(node.output), axis=1))
        drop.add(node.input[1])
        converted += 1
    if converted:
        del g.node[:]
        g.node.extend(new_nodes)
        keep = [i for i in g.initializer if i.name not in drop]
        del g.initializer[:]
        g.initializer.extend(keep)
        del g.value_info[:]
    return converted


def apply_all(model, check=True):
    """Run all safe passes; return (model, report). Caller should re-verify outputs."""
    m = copy.deepcopy(model)
    report = {
        "conv1x1_to_gather": conv1x1_perm_to_gather(m),
        "identity_removed": eliminate_identity(m),
        "deduped": deduplicate_initializers(m),
        "pruned": prune_unused_initializers(m),
        "int32_narrowed": narrow_int32(m),
    }
    if check:
        onnx.checker.check_model(m)
    return m, report


def param_count(model):
    """Total initializer elements (proxy for the 'params' cost term)."""
    return sum(int(np.prod(i.dims)) if i.dims else 1 for i in model.graph.initializer)
