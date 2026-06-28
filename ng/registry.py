import json
import os

REGISTRY_FILE = "registry.json"

def load_registry():
    if not os.path.exists(REGISTRY_FILE):
        return {}
    with open(REGISTRY_FILE, "r") as f:
        return json.load(f)

def save_registry(registry):
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)

def get_best(task_num):
    reg = load_registry()
    return reg.get(str(task_num))

def update_best(task_num, method, params, memory, points, onnx_path, status="correct"):
    reg = load_registry()
    task_key = str(task_num)
    
    current = reg.get(task_key)
    if current and current["status"] == "correct" and status == "correct":
        if points <= current["points"]:
            return False # Not an improvement
    
    if current and current["status"] == "correct" and status != "correct":
        return False # Do not replace a correct solution with a wrong one
            
    reg[task_key] = {
        "status": status,
        "method": method,
        "params": params,
        "memory": memory,
        "points": points,
        "onnx_path": onnx_path
    }
    save_registry(reg)
    return True
