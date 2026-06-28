import sys
import os
import json
import numpy as np
import onnx
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ng.onnx_build import make_model_single_io, make_constant_node, helper
from ng.codec import encode_grid
from ng.scorer import score
from ng.registry import update_best

os.makedirs("models", exist_ok=True)

def create_identity_model():
    nodes = [
        helper.make_node("Identity", inputs=["input"], outputs=["output"])
    ]
    return make_model_single_io(nodes, [])

def create_constant_model(grid):
    tensor = encode_grid(grid)
    nodes = [
        make_constant_node("constant_output", "output", tensor)
    ]
    return make_model_single_io(nodes, [])

def get_most_frequent_output_color(task):
    counts = {}
    for pair in task.get("train", []):
        for row in pair["output"]:
            for c in row:
                counts[c] = counts.get(c, 0) + 1
    if not counts:
        return 0
    return max(counts.keys(), key=lambda k: counts[k])

print("Bootstrapping 400 models...")
for i in range(1, 401):
    path = f"Dataset/task{i:03d}.json"
    if not os.path.exists(path):
        continue
    with open(path, "r") as f:
        task = json.load(f)
        
    is_identity = True
    for pair in task.get("train", []):
        if pair["input"] != pair["output"]:
            is_identity = False
            break
            
    if is_identity:
        model = create_identity_model()
        method = "fallback_identity"
        status = "correct"
    else:
        color = get_most_frequent_output_color(task)
        model = create_constant_model([[color]])
        method = "fallback_constant"
        status = "wrong"
        
    onnx_path = f"models/task{i:03d}.onnx"
    onnx.save(model, onnx_path)
    
    score_res = score(onnx_path)
    if score_res:
        update_best(i, method, score_res["params"], score_res["memory"], score_res["points"], onnx_path, status)
    else:
        print(f"Task {i:03d}: score failed")
        
print("Bootstrapping complete.")
