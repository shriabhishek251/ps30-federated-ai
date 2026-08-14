"""
Phase 3 (first half): centralized baseline.
 
Trains on the FULL, unpartitioned dataset -- this is the accuracy ceiling
that the federated + DP results get compared against on the dashboard.
Without this number, "our federated model gets 87% accuracy" means
nothing to a judge; "our federated model gets 87% vs. 91% centralized"
tells the whole story in one line.
 
Run:
    python src/baseline.py
"""
 
import json
import os
 
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
 
from data_loader import load_data
from model import SimpleMLP
 
METRICS_DIR = os.path.join(os.path.dirname(__file__), "..", "metrics")
 
 
def train_baseline(epochs: int = 20, lr: float = 1e-3, batch_size: int = 256):
    X, y, feature_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
 
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_test_t = torch.tensor(X_test)
    y_test_t = torch.tensor(y_test)
 
    model = SimpleMLP(input_dim=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
 
    # the dataset is ~86% negative / ~14% positive -- an unweighted loss
    # lets the model coast to high accuracy by mostly predicting the
    # majority class while barely detecting real cases (which is exactly
    # what produced accuracy=0.86 / f1=0.25 on the first run). pos_weight
    # scales up the loss contribution of positive examples so the rare
    # class actually gets learned.
    num_pos = y_train.sum()
    num_neg = len(y_train) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos])
    print(f"pos_weight={pos_weight.item():.2f} (neg={int(num_neg)}, pos={int(num_pos)})")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
 
    n = X_train_t.shape[0]
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]
 
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
 
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch + 1}/{epochs}  train_loss={epoch_loss / n:.4f}")
 
    # evaluation
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_test_t)).numpy()
    preds = (probs > 0.5).astype("float32")
 
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    print(f"\nBASELINE  accuracy={acc:.4f}  f1={f1:.4f}  precision={precision:.4f}  "
          f"recall={recall:.4f}  roc_auc={auc:.4f}")
 
    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(os.path.join(METRICS_DIR, "baseline.json"), "w") as f:
        json.dump({
            "accuracy": acc, "f1": f1, "precision": precision,
            "recall": recall, "roc_auc": auc, "epochs": epochs,
            "pos_weight": pos_weight.item(),
        }, f, indent=2)
 
    torch.save(model.state_dict(), os.path.join(METRICS_DIR, "baseline_model.pt"))
    return acc, f1
 
 
if __name__ == "__main__":
    train_baseline()