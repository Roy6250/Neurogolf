import sys
import os
import math
import tempfile
import onnx
import onnxruntime
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../Dataset/neurogolf_utils'))
import neurogolf_utils

def score(onnx_path):
    model = onnx.load(onnx_path)
    sanitized = neurogolf_utils.sanitize_model(model)
    if not sanitized:
        return None
    
    # Run profiling
    options = onnxruntime.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = "trace_" + next(tempfile._get_candidate_names())
    
    try:
        session = onnxruntime.InferenceSession(sanitized.SerializeToString(), options)
    except Exception as e:
        print(f"Failed to load session for profiling: {e}")
        return None
        
    dummy_input = np.zeros((1, 10, 30, 30), dtype=np.float32)
    session.run(["output"], {"input": dummy_input})
    trace_path = session.end_profiling()
    
    memory, params = neurogolf_utils.score_network(sanitized, trace_path)
    
    # Cleanup trace
    if os.path.exists(trace_path):
        os.remove(trace_path)
        
    if memory is None or params is None or memory < 0 or params < 0:
        return None
        
    cost = max(1.0, memory + params)
    points = max(1.0, 25.0 - math.log(cost))
    return {
        "memory": memory,
        "params": params,
        "cost": cost,
        "points": points
    }
