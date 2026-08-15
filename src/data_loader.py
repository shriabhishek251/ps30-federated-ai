"""
Data loading + non-IID partitioning for PS30.

Dataset: CDC Diabetes Health Indicators (public, ~250k rows, binary
classification). Chosen because it's tabular -> a plain MLP is enough,
which sidesteps Opacus's BatchNorm incompatibility entirely, and it
naturally supports partitioning by a real demographic feature (Age),
which gives us a genuine non-IID story instead of a random split.

Two things happen here, and they're deliberately kept separate:
  1. download/load the full dataset (one function)
  2. split it into uneven, non-identical shards, one per simulated
     "hospital" client (a second function)

Run this file directly to generate the client shards:
    python src/data_loader.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import tempfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Path resolution, robust to `flwr run` executing this file from a copy
# in an isolated directory rather than the real project folder.
# Priority: (1) explicit env var, (2) a marker file written the last
# time this script ran directly (see bottom of file) -- this is the
# important one, since it works even when a stale background Flower
# process never saw an env var set in a later shell session, because
# it's a filesystem read, not something tied to process ancestry,
# (3) naive __file__-relative guess, for the very first run.
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

RAW_CSV_PATH = os.path.join(DATA_DIR, "diabetes_raw.csv")
TARGET_COL = "Diabetes_binary"


def download_dataset() -> str:
    """
    Downloads the dataset. Two ways, tried in this order:

      1. ucimlrepo (preferred) — no Kaggle account/login needed at all,
         just needs internet. This dataset is mirrored on the UCI ML
         Repository as "CDC Diabetes Health Indicators" (id=891).
      2. kagglehub (fallback) — needs a Kaggle account; kagglehub will
         prompt for auth the first time, or reads ~/.kaggle/kaggle.json.

    Both need internet — this will NOT work in a sandboxed/offline
    environment. Run it on your laptop, in Colab, or via Claude Code
    (which has real network access on your machine).

    Manual fallback if both of the above give you trouble:
      1. Go to kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
      2. Download specifically "diabetes_binary_health_indicators_BRFSS2015.csv"
         (NOT the "5050split" or "012" variants — those are different label setups)
      3. Save it as data/diabetes_raw.csv
    """
    if os.path.exists(RAW_CSV_PATH):
        print(f"Found existing raw data at {RAW_CSV_PATH}, skipping download.")
        return RAW_CSV_PATH

    os.makedirs(os.path.dirname(RAW_CSV_PATH), exist_ok=True)

    try:
        from ucimlrepo import fetch_ucirepo

        print("Fetching via ucimlrepo (no login required)...")
        dataset = fetch_ucirepo(id=891)  # CDC Diabetes Health Indicators
        X = dataset.data.features
        y = dataset.data.targets
        df = pd.concat([X, y], axis=1)
        if TARGET_COL not in df.columns:
            # target column may come back under a slightly different name
            df = df.rename(columns={y.columns[0]: TARGET_COL})
        df.to_csv(RAW_CSV_PATH, index=False)
        print(f"Saved to {RAW_CSV_PATH} ({len(df)} rows)")
        return RAW_CSV_PATH
    except Exception as e:
        print(f"ucimlrepo path failed ({e}), falling back to kagglehub...")

    import kagglehub

    path = kagglehub.dataset_download("alexteboul/diabetes-health-indicators-dataset")

    # the kaggle dataset ships THREE csvs: the full imbalanced one, a
    # "5050split" balanced one, and a 3-class "012" one -- we want
    # specifically the full binary-imbalanced file, not just anything
    # matching "binary" (the 5050split file also has "binary" in its name)
    exact = [f for f in os.listdir(path)
             if f == "diabetes_binary_health_indicators_BRFSS2015.csv"]
    if not exact:
        raise FileNotFoundError(
            f"Expected 'diabetes_binary_health_indicators_BRFSS2015.csv' in {path}. "
            f"Files present: {os.listdir(path)}. Copy the right one to {RAW_CSV_PATH} manually."
        )

    src = os.path.join(path, exact[0])
    pd.read_csv(src).to_csv(RAW_CSV_PATH, index=False)
    print(f"Saved to {RAW_CSV_PATH}")
    return RAW_CSV_PATH


def get_train_test_indices(csv_path: str = RAW_CSV_PATH, test_size: float = 0.2,
                            random_state: int = 42):
    """
    Returns (train_idx, test_idx) row-position arrays into the raw CSV.
    THIS is the single source of truth for which rows are "test" --
    baseline.py, the client partitioner below, and Day 2's centralized
    federated evaluation all call this SAME function, so the held-out
    test set can never silently drift or leak into a client's training
    shard. random_state is fixed at 42 to match baseline.py exactly;
    don't parameterize it away.
    """
    df = pd.read_csv(csv_path)
    n = len(df)
    train_idx, test_idx = train_test_split(
        np.arange(n), test_size=test_size, random_state=random_state,
        stratify=df[TARGET_COL].values,
    )
    return train_idx, test_idx


def load_data(csv_path: str = RAW_CSV_PATH, save_scaler: bool = True):
    """
    Loads and lightly cleans the dataset. Returns (X, y) as numpy arrays,
    features standardized (mean 0, std 1) — matters for MLP training
    stability and for Opacus's gradient clipping to behave sensibly later.

    The fitted scaler is persisted to metrics/scaler.pkl (when
    save_scaler=True) so every federated client can apply the EXACT same
    transform to its own local shard in Day 2, rather than each client
    fitting a different scaler on its own non-IID slice -- which would
    make the aggregated model's weights meaningless (the same weight
    would represent a different physical quantity per client). This
    reuses column-level statistics (mean/std), not patient records --
    the same public-schema simplification virtually every FL paper makes.
    """
    df = pd.read_csv(csv_path)
    y = df[TARGET_COL].values.astype("float32")
    X = df.drop(columns=[TARGET_COL]).values.astype("float32")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if save_scaler:
        os.makedirs(METRICS_DIR, exist_ok=True)
        joblib.dump(scaler, os.path.join(METRICS_DIR, "scaler.pkl"))

    return X, y, df.drop(columns=[TARGET_COL]).columns.tolist()


def partition_dirichlet(csv_path: str = RAW_CSV_PATH, num_clients: int = 4,
                         alpha: float = 0.5, label_col: str = TARGET_COL,
                         seed: int = 42, train_only: bool = True):
    """
    Splits the dataset into `num_clients` non-IID shards using a
    Dirichlet distribution over the label -- the standard non-IID
    partitioning method in federated learning research (this is what
    "Dirichlet partition, alpha=0.5" refers to in FL papers).

    train_only=True (default, and important): partitions ONLY the rows
    that baseline.py's train split used, via get_train_test_indices().
    The held-out test rows are excluded entirely from every client shard
    -- otherwise a federated client could train on a row that's also
    used to evaluate the baseline, silently inflating federated results
    versus the honest Day 1 number.

    How it works: for each class (0 = no diabetes, 1 = diabetes), draw a
    proportion vector from Dirichlet(alpha, ..., alpha) over the clients,
    then hand out that class's rows according to those proportions. Low
    alpha -> proportions are extreme (some clients get almost none of a
    class) -> strongly non-IID. High alpha -> proportions flatten out
    toward equal shares -> closer to IID. alpha=0.5 is a common middle
    setting: real skew, but every client still gets some of both classes.

    Saves one CSV per client to data/client_<i>.csv and returns the list
    of dataframes.
    """
    rng = np.random.default_rng(seed)
    df = pd.read_csv(csv_path)

    if train_only:
        train_idx, _ = get_train_test_indices(csv_path)  # fixed random_state=42 inside
        df = df.loc[train_idx].reset_index(drop=True)
        print(f"Partitioning {len(df)} TRAIN-only rows across {num_clients} clients "
              f"(the held-out test split is excluded, matching baseline.py exactly).")

    client_indices = [[] for _ in range(num_clients)]
    for label_value in sorted(df[label_col].unique()):
        idx = df.index[df[label_col] == label_value].to_numpy().copy()
        rng.shuffle(idx)

        proportions = rng.dirichlet(alpha=[alpha] * num_clients)
        # convert proportions -> cut points over this class's indices
        cuts = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        splits = np.split(idx, cuts)

        for client_id, split_idx in enumerate(splits):
            client_indices[client_id].extend(split_idx.tolist())

    os.makedirs(DATA_DIR, exist_ok=True)

    shards = []
    for i, idx in enumerate(client_indices):
        shard = df.loc[idx].sample(frac=1, random_state=seed).reset_index(drop=True)
        shard.to_csv(os.path.join(DATA_DIR, f"client_{i}.csv"), index=False)
        pos_rate = shard[label_col].mean()
        print(f"client_{i}: {len(shard)} rows, {label_col} positive rate = {pos_rate:.2%}")
        shards.append(shard)

    return shards


def partition_by_age(csv_path: str = RAW_CSV_PATH, num_clients: int = 4, by_column: str = "Age"):
    """
    Alternate partitioner: sorts by `by_column` and cuts into unequal
    contiguous chunks (e.g. "Client 0" skews younger, "Client 3" skews
    older). Simpler to explain in one sentence than Dirichlet, but not
    the technique the FL literature uses -- kept here as a fallback /
    talking point about feature skew vs. label skew if you want both.
    """
    df = pd.read_csv(csv_path)
    df = df.sort_values(by=by_column).reset_index(drop=True)

    n = len(df)
    weights = [0.15, 0.20, 0.30, 0.35][:num_clients]
    if len(weights) < num_clients:
        weights = [1.0 / num_clients] * num_clients

    shards = []
    start = 0
    os.makedirs(DATA_DIR, exist_ok=True)
    for i, w in enumerate(weights):
        end = n if i == len(weights) - 1 else start + int(n * w)
        shard = df.iloc[start:end].reset_index(drop=True)
        shard.to_csv(os.path.join(DATA_DIR, f"client_{i}.csv"), index=False)
        print(f"client_{i}: {len(shard)} rows, {by_column} range "
              f"[{shard[by_column].min()}, {shard[by_column].max()}]")
        shards.append(shard)
        start = end

    return shards


if __name__ == "__main__":
    # remember the real project root for any OTHER process that can't
    # discover it itself -- specifically flwr run's isolated app copy,
    # whether reached via env var or a stale background SuperLink that
    # started before any env var was set. Filesystem read, not
    # environment inheritance, so it works regardless of process timing.
    try:
        with open(_MARKER_FILE, "w") as f:
            f.write(_PROJECT_ROOT)
        print(f"(recorded project root at {_MARKER_FILE} for flwr run to find)")
    except OSError as e:
        print(f"warning: couldn't write marker file ({e}) -- flwr run may need "
              f"PS30_PROJECT_ROOT set explicitly")

    path = download_dataset()
    partition_dirichlet(path, num_clients=4, alpha=0.5)
