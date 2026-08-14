"""
Day 1 refinement: threshold sweep on the already-trained baseline model.

Training decides WHAT the model learns; the classification threshold
decides HOW its probability output gets turned into a yes/no. These are
two separate knobs. pos_weight (in baseline.py) already pushed the model
toward flagging more cases as positive -- that's likely why 0.5 gave such
high recall already. This script sweeps the threshold on the SAME
trained model (loaded from metrics/baseline_model.pt) to find a better
operating point, with zero retraining cost.

Run (after baseline.py has been run at least once):
    python src/threshold_analysis.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import train_test_split

from data_loader import load_data
from model import SimpleMLP

METRICS_DIR = os.path.join(os.path.dirname(__file__), "..", "metrics")


def analyze():
    X, y, _ = load_data()
    # same random_state as baseline.py -> reconstructs the EXACT same
    # held-out test set, not a fresh sample
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_test_t = torch.tensor(X_test)

    model = SimpleMLP(input_dim=X_test.shape[1])
    model.load_state_dict(torch.load(os.path.join(METRICS_DIR, "baseline_model.pt")))
    model.eval()

    with torch.no_grad():
        probs = torch.sigmoid(model(X_test_t)).numpy()

    print(f"{'threshold':>10} {'precision':>10} {'recall':>10} {'f1':>10}")
    best_f1, best_thresh = -1.0, 0.5
    for t in np.arange(0.10, 0.95, 0.05):
        preds = (probs > t).astype("float32")
        p = precision_score(y_test, preds, zero_division=0)
        r = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        marker = "  <-- current (baseline.py default)" if abs(t - 0.5) < 0.026 else ""
        print(f"{t:>10.2f} {p:>10.3f} {r:>10.3f} {f1:>10.3f}{marker}")
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    print(f"\nBest F1 at threshold={best_thresh:.2f} (f1={best_f1:.3f})")
    print("Pick the row that fits your pitch's priorities, not necessarily the best-F1 one --")
    print("e.g. if you want to argue 'we catch almost everyone,' favor a row with higher recall.")

    # precision-recall curve -- save as an image, directly usable in the PPT
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(rec_curve, prec_curve, linewidth=2, color="#2563eb")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Baseline Model: Precision-Recall Trade-off")
    plt.grid(alpha=0.3)
    os.makedirs(METRICS_DIR, exist_ok=True)
    out_path = os.path.join(METRICS_DIR, "pr_curve.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved precision-recall curve to {out_path}")


if __name__ == "__main__":
    analyze()
