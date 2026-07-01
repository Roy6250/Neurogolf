import os
import json
import numpy as np
import onnxruntime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.special import softmax
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ng.codec import encode_grid, CHANNELS, HEIGHT, WIDTH

def evaluate_metrics():
    print("Evaluating metrics across all models...")
    
    y_true_all = []
    y_pred_all = []
    y_prob_all = []

    models_dir = "models"
    dataset_dir = "Dataset"
    
    tasks = [f for f in os.listdir(dataset_dir) if f.startswith("task") and f.endswith(".json")]
    tasks.sort()

    for task_file in tasks:
        task_path = os.path.join(dataset_dir, task_file)
        task_name = task_file.split(".")[0]
        model_path = os.path.join(models_dir, f"{task_name}.onnx")
        
        if not os.path.exists(model_path):
            continue
            
        with open(task_path, "r") as f:
            task_data = json.load(f)
            
        try:
            sess = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        except Exception as e:
            print(f"Failed to load model {model_path}: {e}")
            continue

        for pair in task_data.get("test", []):
            inp_tensor = encode_grid(pair["input"])
            target_grid = pair["output"]
            
            # Predict
            out_tensor = sess.run(["output"], {"input": inp_tensor})[0] # Shape: (1, 10, 30, 30)
            
            # Apply softmax to get probabilities
            out_probs = softmax(out_tensor, axis=1)[0] # Shape: (10, 30, 30)
            
            # Extract target pixels exactly matching the target grid dimensions
            tgt_h = len(target_grid)
            tgt_w = len(target_grid[0]) if tgt_h > 0 else 0
            
            for r in range(tgt_h):
                for c in range(tgt_w):
                    true_color = target_grid[r][c]
                    pred_probs = out_probs[:, r, c]
                    pred_color = np.argmax(pred_probs)
                    
                    y_true_all.append(true_color)
                    y_pred_all.append(pred_color)
                    y_prob_all.append(pred_probs)
                    
    if not y_true_all:
        print("No valid test cases evaluated.")
        return

    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)
    y_prob = np.array(y_prob_all)
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    
    # Using weighted average for Precision, Recall, F1 due to class imbalance
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # ROC-AUC requires identifying all unique classes present in true labels to compute ovr properly
    # If not all 10 classes are present in the dataset, we must specify labels parameter.
    # We will force the labels to be 0-9.
    labels = np.arange(10)
    try:
        roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted', labels=labels)
    except Exception as e:
        print(f"Warning: ROC-AUC calculation failed: {e}")
        roc_auc = float('nan')

    print("\n--- Metric Results ---")
    print(f"Total Pixels Evaluated: {len(y_true)}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

if __name__ == "__main__":
    evaluate_metrics()
