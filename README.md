# PS30 — Privacy-Preserving Federated AI Platform

Train one shared model across simulated clients (hospitals) that each
hold a non-identical slice of patient data, without any client's raw
data ever leaving its own shard — with differential privacy and a
simplified secure aggregation layer on top.

**Dataset:** CDC Diabetes Health Indicators (~250k rows, binary
classification). Chosen because a plain MLP handles it well, which
avoids Opacus's BatchNorm incompatibility, and it partitions
meaningfully by Age to simulate demographically distinct clients.

## Status

- [x] Day 1 — project structure, baseline model
      - accuracy=0.7241, precision=0.309, recall=0.791, f1=0.444, **roc_auc=0.826**
      - trained with `pos_weight=6.18` (loss-reweighted for class imbalance)
      - operating threshold locked at **0.50** (79% recall / 31% precision) —
        chosen deliberately for a screening use case; see `src/threshold_analysis.py`
        for the full precision/recall sweep and reasoning
      - this threshold must be reused for every later phase's evaluation,
        so federated/DP results stay comparable to this baseline
- [ ] Day 2 — Flower FedAvg loop
      - `fl_app/` package (client_app.py, server_app.py, task.py) + `pyproject.toml`
      - 4 simulated clients, weighted FedAvg, centralized eval each round
        on the SAME held-out test set + threshold as the Day 1 baseline
      - fixed a data-leakage bug: client shards now exclude test rows
        (see `get_train_test_indices()` in `data_loader.py`)
      - run with `flwr run .` — see below
- [x] Day 3 — Opacus differential privacy
      - `train_fn_dp()` in `fl_app/task.py`, toggled via `use-dp`/`target-epsilon`
      - full off/1/4/8 sweep, in `metrics/dp_sweep_summary.json`:

        | run | accuracy | f1 | precision | recall | roc_auc |
        |---|---|---|---|---|---|
        | dp-off | 0.833 | 0.455 | – | – | 0.825 |
        | eps8 | 0.865 | 0.226 | 0.560 | 0.141 | 0.820 |
        | eps4 | 0.865 | 0.211 | 0.568 | 0.129 | 0.820 |
        | eps1 | 0.865 | 0.194 | 0.576 | 0.117 | 0.816 |

      - key finding: DP-SGD's per-sample gradient clipping partially
        neutralizes the Day 1 pos_weight class-imbalance fix -- AUC stays
        fairly flat across ε, but F1/recall drop sharply the instant DP
        turns on at all, since clipping caps loud (weighted) positive-class
        gradients the same as everything else. A real, explainable
        interaction, not a bug -- see `train_fn_dp()`'s docstring.
      - simplification: per-round budget, not composed across rounds
- [ ] Day 4 — secure aggregation + Streamlit dashboard
- [ ] Day 5 — deploy + demo polish

## Setup

### Option A — locally (your laptop is fine, no GPU needed for this model)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Option B — Google Colab (handy for sharing progress with teammates)

Paste into the first cell of a new Colab notebook:

```python
!git clone <YOUR_GITHUB_REPO_URL>
%cd ps30-federated-ai
!pip install -r requirements.txt
```

## Getting the dataset

The loader tries `kagglehub` automatically (needs a Kaggle account —
it'll prompt for an API token the first time). If that's a hassle:

1. Search Kaggle for "Diabetes Health Indicators Dataset" (by alexteboul)
2. Download the **binary classification** CSV variant
3. Save it as `data/diabetes_raw.csv`

Then run:

```bash
python src/data_loader.py
```

This downloads (or finds) the raw CSV and writes `data/client_0.csv`
through `data/client_3.csv` — four non-IID shards, partitioned using a
**Dirichlet distribution (α=0.5) over the diabetes label**, the standard
non-IID partitioning method used in FL research. Each client ends up
with a different diabetes prevalence rate — a plausible stand-in for
different hospitals seeing different patient populations. (An alternate
age-sort partitioner, `partition_by_age`, is also in `data_loader.py` if
you want a second, simpler-to-explain non-IID angle for the pitch.)

## Day 1: centralized baseline

```bash
python src/baseline.py
```

Trains the MLP on the full dataset and prints/saves accuracy + F1 to
`metrics/baseline.json`. This number is the ceiling everything else
(federated, then federated+DP) gets compared against on the dashboard.

## Day 2: federated training (Flower)

Re-run the partitioner first — it now excludes test rows (a leakage fix
that matters: earlier client shards could include rows also used for
baseline evaluation):

```bash
python src/data_loader.py
```

Then run the federated simulation:

```bash
pip install -r requirements.txt
flwr run . --stream
```

`--stream` shows live round-by-round logs: each client's local train
loss, each client's local eval (diagnostic only, not the number that
matters), and — the number that matters — a centralized evaluation line
after every round:

```
[round 5] centralized eval (same test set as baseline): accuracy=0.71 f1=0.42 roc_auc=0.80
```

That's evaluated on the exact same held-out test set as the Day 1
baseline (`accuracy=0.7241 f1=0.444 roc_auc=0.826`), so the two numbers
are directly comparable. Override any default from `pyproject.toml`
without touching code, e.g.:

```bash
flwr run . --run-config "num-server-rounds=15 local-epochs=3" --stream
```

Final model saves to `metrics/federated_model.pt`, full round history to
`metrics/federated_history.json` (dashboard fuel for Day 4).

## Day 3: differential privacy (Opacus)

Each client's local training can now optionally run through Opacus
DP-SGD (clip + calibrated noise), targeting a chosen privacy budget ε.
Run the full off/1/4/8 sweep your dashboard needs — each command below
saves to its own tagged files, so nothing gets overwritten:

```bash
flwr run . --run-config "use-dp=false" --stream                          # dp-off (reproduces Day 2)
flwr run . --run-config "use-dp=true target-epsilon=8" --stream          # loose privacy
flwr run . --stream                                                       # target-epsilon=4 (the default)
flwr run . --run-config "use-dp=true target-epsilon=1" --stream          # strict privacy
```

Every run appends its final result to `metrics/dp_sweep_summary.json` —
after all four, that one file has everything Day 4's dashboard needs to
plot accuracy/F1/AUC vs. ε. Per-run round-by-round detail also saves to
`metrics/federated_history_<tag>.json` (tags: `dp-off`, `eps1`, `eps4`, `eps8`).

**Worth understanding, not just running:** each ε here is a per-round,
per-client budget (a fresh Opacus engine every round) — it does not
account for privacy composing across all 10 rounds, so the true
cumulative privacy loss over a full run is higher than the nominal
number. That's a standard, named simplification in DP-FL demos, not an
oversight — mention it if asked.

## Project structure

```
ps30-federated-ai/
├── pyproject.toml          # Flower app config (Day 2+)
├── requirements.txt
├── data/                  # gitignored — never commit patient data or shards
├── metrics/                # gitignored (except *.json/*.png) — regenerated by runs
├── src/
│   ├── data_loader.py     # download + non-IID partition + train/test split (Day 1)
│   ├── model.py           # shared MLP definition (Day 1)
│   ├── baseline.py        # centralized baseline trainer (Day 1)
│   └── threshold_analysis.py  # precision/recall threshold sweep (Day 1)
├── fl_app/                 # Flower app (Day 2)
│   ├── task.py             # model + data loading + train/eval, shared by both apps
│   ├── client_app.py       # ClientApp: local training per simulated client
│   └── server_app.py       # ServerApp: FedAvg + centralized evaluation
└── dashboard/
    └── app.py              # Streamlit privacy-utility dashboard (Day 4)
```

## Why these choices (for your pitch / judges' Q&A)

- **Flower**, not a hand-rolled loop: it's a real open-source FL
  framework with a simulation runtime built for exactly this, so
  round-based FedAvg across virtual clients doesn't have to be
  reinvented under a 5-day deadline.
- **Opacus DP-SGD**: clips + adds calibrated noise to each client's
  local update before it ever leaves the client, governed by a privacy
  budget (ε). Lower ε = more noise = more privacy = lower accuracy —
  a genuine trade-off we show on the dashboard, not hide.
- **Secure aggregation is a simplified stand-in**: real secure
  aggregation is research-grade cryptography, out of scope for a
  hackathon. Pairwise random masks that cancel out only when every
  client's masked update is summed is stated plainly as a
  proof-of-concept of the underlying principle, not production crypto.