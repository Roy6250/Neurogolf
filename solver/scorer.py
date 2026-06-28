"""Official scoring + correctness, wrapping the competition's neurogolf_utils.

This is the ground truth (replaces the local argmax/param proxies). It uses the
real `verify_subset` (threshold >0, exact one-hot match incl. all-zero padding) and
`score_network` (memory in bytes of intermediate tensors + params in element count)
exactly as the leaderboard does.
"""
import math
import os
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime as ort

# import the competition module from the repo's Dataset/ copy
_NG_DIR = os.path.join(os.path.dirname(__file__), "..", "Dataset", "neurogolf_utils")
sys.path.insert(0, os.path.abspath(_NG_DIR))
import neurogolf_utils as ng  # noqa: E402


def _session(model, profiling=False):
    """Save -> sanitize -> ORT session (mirrors verify_network); profiling optional."""
    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    tmp.close()
    onnx.save(model, tmp.name)
    if not ng.check_network(tmp.name):
        os.unlink(tmp.name)
        return None, None, None
    sanitized = ng.sanitize_model(onnx.load(tmp.name))
    if sanitized is None:
        os.unlink(tmp.name)
        return None, None, None
    opts = ort.SessionOptions()
    # profiling is only needed for memory measurement (score); verify turns it off
    # so no trace JSON is written to cwd.
    opts.enable_profiling = bool(profiling)
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    if profiling:
        opts.profile_file_prefix = os.path.basename(tmp.name).replace(".onnx", "")
    sess = ort.InferenceSession(sanitized.SerializeToString(), opts)
    return sess, sanitized, tmp.name


def verify(model, examples, subsets=("train", "test", "arc-gen")):
    """Real correctness gate. Returns (ok, right, wrong)."""
    sess, _, path = _session(model, profiling=False)
    if sess is None:
        return False, 0, 1
    right = wrong = 0
    try:
        for s in subsets:
            r, w, _ = ng.verify_subset(sess, examples.get(s, []))
            right += r
            wrong += w
    finally:
        _cleanup(path)
    return wrong == 0, right, wrong


def score(model):
    """Return (points, memory_bytes, params) or (0.0, None, None) if unscoreable."""
    sess, sanitized, path = _session(model, profiling=True)
    if sess is None:
        return 0.0, None, None
    trace = None
    try:
        trace = sess.end_profiling()
        memory, params = ng.score_network(sanitized, trace)
    finally:
        _cleanup(path, trace)
    if memory is None or params is None or memory < 0 or params < 0:
        return 0.0, memory, params
    points = max(1.0, 25.0 - math.log(max(1.0, memory + params)))
    return points, memory, params


def _cleanup(*paths):
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.unlink(p)
            except Exception:
                pass
