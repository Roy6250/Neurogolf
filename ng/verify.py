import onnxruntime
import onnx
import numpy as np
import os
import json
from ng.codec import encode_grid, decode_tensor

def load_task(task_num):
    path = f"Dataset/task{task_num:03d}.json"
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

def run_network(session, grid_input):
    tensor = encode_grid(grid_input)
    result = session.run(["output"], {"input": tensor})
    # Following neurogolf_utils.run_network which treats > 0 as 1
    # We pass it to decode_tensor which checks > 0.5 (or we can use > 0.0)
    # decode_tensor handles this via >= 0.5, but neurogolf_utils uses > 0.0.
    # We will pass (result[0] > 0.0).astype(float) to decode_tensor.
    binarized = (result[0] > 0.0).astype(np.float32)
    return decode_tensor(binarized)

def verify(onnx_path, task_num):
    # Gate 1: Check Exact match
    try:
        options = onnxruntime.SessionOptions()
        session = onnxruntime.InferenceSession(onnx_path, options)
    except Exception as e:
        return False, f"Failed to load ONNX: {e}"

    task = load_task(task_num)
    if not task:
        return False, "Task file not found"
        
    all_pairs = task.get("train", []) + task.get("test", []) + task.get("arc-gen", [])
    for pair in all_pairs:
        expected = pair["output"]
        try:
            actual = run_network(session, pair["input"])
            if actual != expected:
                return False, "Mismatched output on pair"
        except Exception as e:
            return False, f"Exception during execution: {e}"
            
    # Gate 2: Random Equivalence (Deferred to M1 when numpy primitives are available)
    
    # Gate 3: Self-consistency check 
    try:
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model, full_check=True)
    except Exception as e:
        return False, f"ONNX Check failed: {e}"
        
    return True, "Passed"
