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
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "diabetes_raw.csv")
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


def load_data(csv_path: str = RAW_CSV_PATH):
    """
    Loads and lightly cleans the dataset. Returns (X, y) as numpy arrays,
    features standardized (mean 0, std 1) — matters for MLP training
    stability and for Opacus's gradient clipping to behave sensibly later.
    """
    df = pd.read_csv(csv_path)
    y = df[TARGET_COL].values.astype("float32")
    X = df.drop(columns=[TARGET_COL]).values.astype("float32")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, df.drop(columns=[TARGET_COL]).columns.tolist()


def partition_dirichlet(csv_path: str = RAW_CSV_PATH, num_clients: int = 4,
                         alpha: float = 0.5, label_col: str = TARGET_COL,
                         seed: int = 42):
    """
    Splits the full dataset into `num_clients` non-IID shards using a
    Dirichlet distribution over the label -- the standard non-IID
    partitioning method in federated learning research (this is what
    "Dirichlet partition, alpha=0.5" refers to in FL papers).

    How it works: for each class (0 = no diabetes, 1 = diabetes), draw a
    proportion vector from Dirichlet(alpha, ..., alpha) over the clients,
    then hand out that class's rows according to those proportions. Low
    alpha -> proportions are extreme (some clients get almost none of a
    class) -> strongly non-IID. High alpha -> proportions flatten out
    toward equal shares -> closer to IID. alpha=0.5 is a common middle
    setting: real skew, but every client still gets some of both classes.

    Narrative for the pitch: different "hospitals" end up seeing different
    diabetes prevalence in their patient population -- plausible in real
    life, and a stronger non-IID story than an arbitrary age cut.

    Saves one CSV per client to data/client_<i>.csv and returns the list
    of dataframes.
    """
    rng = np.random.default_rng(seed)
    df = pd.read_csv(csv_path)

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

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    shards = []
    for i, idx in enumerate(client_indices):
        shard = df.loc[idx].sample(frac=1, random_state=seed).reset_index(drop=True)
        shard.to_csv(os.path.join(data_dir, f"client_{i}.csv"), index=False)
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
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    for i, w in enumerate(weights):
        end = n if i == len(weights) - 1 else start + int(n * w)
        shard = df.iloc[start:end].reset_index(drop=True)
        shard.to_csv(os.path.join(data_dir, f"client_{i}.csv"), index=False)
        print(f"client_{i}: {len(shard)} rows, {by_column} range "
              f"[{shard[by_column].min()}, {shard[by_column].max()}]")
        shards.append(shard)
        start = end

    return shards


if __name__ == "__main__":
    path = download_dataset()
    partition_dirichlet(path, num_clients=4, alpha=0.5)
