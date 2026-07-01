import json
import onnxruntime
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import sys
sys.path.append('.')
from ng.codec import encode_grid, decode_grid

# test model 1
model_path = "models/task001.onnx"
data_path = "Dataset/task001.json"

with open(data_path) as f:
    task = json.load(f)

sess = onnxruntime.InferenceSession(model_path)
print("Inputs:", [i.name for i in sess.get_inputs()])
print("Outputs:", [o.name for o in sess.get_outputs()])

for pair in task.get("test", []):
    inp = encode_grid(pair["input"])
    # encode_grid returns something. Let's check its shape.
    print(inp.shape)
    out = sess.run(["output"], {"input": inp})[0]
    print(out.shape)
    pred_grid = decode_grid(out)
    print("Pred grid:", pred_grid)
