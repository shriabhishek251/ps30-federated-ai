"""
Day 2/5: shared definitions used by both client_app.py and server_app.py --
the model, per-client data loading, local training, and the centralized
evaluation that produces the number actually comparable to Day 1's
baseline.

Design choices worth remembering for the pitch:
- Same SimpleMLP architecture and same pos_weight as the Day 1 baseline.
- True Cumulative Privacy (Day 5 feature): Opacus calculates privacy across 
  the ENTIRE federated lifespan (total_rounds * local_epochs), not just 
  a single round.
- Fault Tolerance (Day 5 feature): Clients can drop offline without crashing 
  the server aggregate.
"""

import os
import sys
import tempfile

import joblib
import pandas as pd
import torch
from opacus import PrivacyEngine
from torch.utils.data import DataLoader, TensorDataset

# Identical resolution logic to src/data_loader.py
_MARKER_FILE = os.path.join(tempfile.gettempdir(), "ps30_federated_ai_project_root.txt")


def _resolve_project_root() -> str:
    env_val = os.environ.get("PS30_PROJECT_ROOT")
    if env_val:
        return env_val
    if os.path.exists(_MARKER_FILE):
        try:
            with open(_MARKER_FILE) as f:
                marked = f.read().strip()
            if marked:
                return marked
        except OSError:
            pass
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


_PROJECT_ROOT = _resolve_project_root()
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
METRICS_DIR = os.path.join(_PROJECT_ROOT, "metrics")

sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
from model import SimpleMLP  # noqa: E402

TARGET_COL = "Diabetes_binary"
NUM_FEATURES = 21          # must match data_loader.py's feature count (21 columns, minus target)
GLOBAL_POS_WEIGHT = 6.18   # same value baseline.py computed on the full training set


def _scaler():
    path = os.path.join(METRICS_DIR, "scaler.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python src/data_loader.py` first (it saves "
            f"the shared scaler as a side effect of load_data())."
        )
    return joblib.load(path)


def get_model() -> SimpleMLP:
    return SimpleMLP(input_dim=NUM_FEATURES)


def load_client_data(partition_id: int, batch_size: int = 128) -> DataLoader:
    csv_path = os.path.join(DATA_DIR, f"client_{partition_id}.csv")
    df = pd.read_csv(csv_path)
    y = df[TARGET_COL].values.astype("float32")
    X = df.drop(columns=[TARGET_COL]).values.astype("float32")
    X = _scaler().transform(X).astype("float32")

    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def load_global_test_data():
    from data_loader import get_train_test_indices, load_data 

    X, y, _ = load_data(save_scaler=False)
    _, test_idx = get_train_test_indices()
    return torch.tensor(X[test_idx]), torch.tensor(y[test_idx])


def train_fn(model, dataloader, epochs, lr, device) -> float:
    model.to(device)
    pos_weight = torch.tensor([GLOBAL_POS_WEIGHT]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    running_loss, n_batches = 0.0, 0
    for _ in range(epochs):
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
    return running_loss / max(n_batches, 1)


# --- DAY 5 FIX: Added 'total_rounds' parameter for true cumulative privacy ---
def train_fn_dp(model, dataloader, epochs, lr, device,
                 target_epsilon, target_delta, max_grad_norm, total_rounds=10):
    """
    Local training with Opacus DP-SGD with True Cumulative Composition.
    Instead of calculating epsilon for a single round, this calculates the 
    noise required to preserve the target privacy budget across the ENTIRE 
    lifespan of the federated training loop (all rounds * local epochs).
    """
    model.to(device)
    pos_weight = torch.tensor([GLOBAL_POS_WEIGHT]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Calculate total epochs across all communication rounds for composition
    cumulative_epochs = epochs * total_rounds

    privacy_engine = PrivacyEngine()
    dp_model, dp_optimizer, dp_dataloader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        epochs=cumulative_epochs,  # OVERRIDE: Inform Opacus of the total steps
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        max_grad_norm=max_grad_norm,
    )

    dp_model.train()
    running_loss, n_batches = 0.0, 0
    for _ in range(epochs):
        for xb, yb in dp_dataloader:
            xb, yb = xb.to(device), yb.to(device)
            dp_optimizer.zero_grad()
            loss = criterion(dp_model(xb), yb)
            loss.backward()
            dp_optimizer.step()
            running_loss += loss.item()
            n_batches += 1

    epsilon_spent = privacy_engine.get_epsilon(delta=target_delta)

    # Unwrap back to a plain module
    clean_model = dp_model._module if hasattr(dp_model, "_module") else dp_model

    avg_loss = running_loss / max(n_batches, 1)
    return clean_model, avg_loss, epsilon_spent


def eval_fn(model, dataloader, device):
    model.to(device)
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()
    correct, total_loss, n = 0, 0.0, 0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            total_loss += criterion(logits, yb).item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == yb).sum().item()
            n += len(yb)
    return total_loss / max(len(dataloader), 1), correct / max(n, 1)


def centralized_evaluate(model, device, threshold: float = 0.5) -> dict:
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                  recall_score, roc_auc_score)

    X_test, y_test = load_global_test_data()
    model.to(device)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_test.to(device))).cpu().numpy()
    preds = (probs > threshold).astype("float32")
    y_test_np = y_test.numpy()

    return {
        "accuracy": float(accuracy_score(y_test_np, preds)),
        "f1": float(f1_score(y_test_np, preds, zero_division=0)),
        "precision": float(precision_score(y_test_np, preds, zero_division=0)),
        "recall": float(recall_score(y_test_np, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test_np, probs)),
    }