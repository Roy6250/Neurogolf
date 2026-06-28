import sys
import os
import json
import zipfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ng.registry import load_registry

def build_submission():
    reg = load_registry()
    if len(reg) < 400:
        print(f"Warning: Registry only has {len(reg)} tasks.")
        
    with zipfile.ZipFile("submission.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(1, 401):
            task_key = str(i)
            if task_key not in reg:
                print(f"Missing task {i} in registry")
                continue
            onnx_path = reg[task_key]["onnx_path"]
            if not os.path.exists(onnx_path):
                print(f"Missing onnx file for task {i}: {onnx_path}")
                continue
            zf.write(onnx_path, arcname=f"task{i:03d}.onnx")
            
    print("submission.zip built successfully.")

if __name__ == "__main__":
    build_submission()
