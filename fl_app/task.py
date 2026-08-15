"""
Day 2: shared definitions used by both client_app.py and server_app.py --
the model, per-client data loading, local training, and the centralized
evaluation that produces the number actually comparable to Day 1's
baseline.

Design choices worth remembering for the pitch:
- Same SimpleMLP architecture and same pos_weight as the Day 1 baseline,
  so any accuracy difference reflects federation/DP, not "we also
  quietly changed the model."
- Feature scaling stats (mean/std) were fit once in Day 1 and are reused
  here via metrics/scaler.pkl -- aggregate column statistics, not patient
  records, so sharing them doesn't violate "raw data never leaves the
  client." Every published FL paper makes an equivalent assumption.
- Each client trains on 100% of its own (test-excluded) shard. The
  GLOBAL model is evaluated centrally on the server, against the exact
  same held-out test set and threshold=0.5 as the Day 1 baseline.
"""

import os
import sys
import tempfile

import joblib
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# Identical resolution logic to src/data_loader.py, and it has to be
# duplicated here (not imported) because finding src/ at all is what
# this logic is for -- can't import data_loader.py's version of it
# before src/ is even on sys.path. Same marker file path as
# data_loader.py writes to, so once `python src/data_loader.py` has been
# run directly one time, this always finds the real project, regardless
# of where flwr run copies this file to or how stale its background
# SuperLink process is.
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
    """Loads ONE client's shard (already excludes the global test rows --
    see data_loader.py's partition_dirichlet), applies the shared scaler,
    returns a DataLoader ready for local training."""
    csv_path = os.path.join(DATA_DIR, f"client_{partition_id}.csv")
    df = pd.read_csv(csv_path)
    y = df[TARGET_COL].values.astype("float32")
    X = df.drop(columns=[TARGET_COL]).values.astype("float32")
    X = _scaler().transform(X).astype("float32")

    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def load_global_test_data():
    """The SAME held-out test set baseline.py evaluates on, reconstructed
    via the shared get_train_test_indices() helper so it can never
    silently drift out of sync with Day 1's split."""
    from data_loader import get_train_test_indices, load_data  # src/ already on sys.path (see top of file)

    X, y, _ = load_data(save_scaler=False)  # scaler already saved on Day 1; don't refit here
    _, test_idx = get_train_test_indices()
    return torch.tensor(X[test_idx]), torch.tensor(y[test_idx])


def train_fn(model, dataloader, epochs, lr, device) -> float:
    """Local training loop -- identical loss setup to the Day 1 baseline
    (same pos_weight), just running on one client's shard instead of the
    full dataset."""
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


def eval_fn(model, dataloader, device):
    """Local (per-client, on its own shard) eval -- diagnostic only, used
    for the round-by-round terminal printout. NOT the number that gets
    compared to the baseline; that's centralized_evaluate() below."""
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
    """Evaluates the GLOBAL (aggregated) model on the shared held-out test
    set -- THIS is the number that gets compared directly to the Day 1
    baseline (accuracy=0.7241, f1=0.444, roc_auc=0.826, threshold=0.50)."""
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
