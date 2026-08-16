"""
Day 3: Flower ServerApp. Runs FedAvg for N rounds across the 4 simulated
clients (weighted by each client's local data size), then, after every
round, evaluates the aggregated GLOBAL model centrally -- on the EXACT
same held-out test set and threshold as the Day 1 baseline -- so the
numbers are directly, honestly comparable, not apples to oranges.

Each run (dp-off, eps=1, eps=4, eps=8, ...) saves to its OWN tagged
filenames, and appends its final result into metrics/dp_sweep_summary.json
-- so running the full off/1/4/8 sweep across several `flwr run`
invocations accumulates all four results in one place for the Day 4
dashboard, instead of each run silently overwriting the last one's files.
"""

import json
import os

import torch
from flwr.common import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.server import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from .task import METRICS_DIR, centralized_evaluate, get_model

app = ServerApp()


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes")


def make_global_evaluate(device, history: list):
    """Returns the evaluate_fn the strategy calls after every round.
    History is accumulated here directly (a closure list), rather than
    inspected off the Result object afterward, since Result's exact
    attribute names for this weren't something I could verify without
    flwr installed -- this way is self-contained and doesn't depend on
    guessing that shape."""

    def global_evaluate(round_num: int, arrays: ArrayRecord) -> MetricRecord:
        model = get_model()
        model.load_state_dict(arrays.to_torch_state_dict())
        results = centralized_evaluate(model, device, threshold=0.5)
        print(f"[round {round_num}] centralized eval (same test set as baseline): "
              f"accuracy={results['accuracy']:.4f} f1={results['f1']:.4f} "
              f"roc_auc={results['roc_auc']:.4f}")
        history.append({"round": round_num, **results})
        return MetricRecord(results)

    return global_evaluate


@app.main()
def main(grid: Grid, context: Context) -> None:
    num_rounds = int(context.run_config["num-server-rounds"])
    lr = float(context.run_config["learning-rate"])
    fraction_evaluate = float(context.run_config["fraction-evaluate"])
    use_dp = _as_bool(context.run_config["use-dp"])
    target_epsilon = float(context.run_config["target-epsilon"]) if use_dp else None

    run_tag = f"eps{target_epsilon:g}" if use_dp else "dp-off"
    print(f"\n=== Run: {run_tag} (use_dp={use_dp}"
          f"{f', target_epsilon={target_epsilon}' if use_dp else ''}) ===\n")

    model = get_model()
    initial_arrays = ArrayRecord(model.state_dict())

    strategy = FedAvg(fraction_evaluate=fraction_evaluate)

    device = torch.device("cpu")
    history: list = []

    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=make_global_evaluate(device, history),
    )

    # save this run's final global model + full round history, tagged so
    # sweeping off/1/4/8 doesn't overwrite the previous run's files
    final_model = get_model()
    final_model.load_state_dict(result.arrays.to_torch_state_dict())
    os.makedirs(METRICS_DIR, exist_ok=True)
    torch.save(final_model.state_dict(), os.path.join(METRICS_DIR, f"federated_model_{run_tag}.pt"))

    with open(os.path.join(METRICS_DIR, f"federated_history_{run_tag}.json"), "w") as f:
        json.dump(history, f, indent=2)

    # accumulate into one sweep-summary file so Day 4 has a single place
    # to read all off/1/4/8 results from
    summary_path = os.path.join(METRICS_DIR, "dp_sweep_summary.json")
    summary = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except (json.JSONDecodeError, OSError):
            summary = {}

    final = history[-1] if history else {}
    print(f"\nFinal result for run '{run_tag}': {final}")
    if "dp-off" in summary and run_tag != "dp-off":
        off = summary["dp-off"]
        print(f"vs. this project's own DP-off run: accuracy={off.get('accuracy'):.4f} "
              f"f1={off.get('f1'):.4f} roc_auc={off.get('roc_auc'):.4f}")
    print(f"(Day 1 centralized baseline, for reference: accuracy=0.7241 f1=0.444 roc_auc=0.826)")

    summary[run_tag] = {
        "use_dp": use_dp,
        "target_epsilon": target_epsilon,
        **{k: v for k, v in final.items() if k != "round"},
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved model to metrics/federated_model_{run_tag}.pt")
    print(f"Saved round history to metrics/federated_history_{run_tag}.json")
    print(f"Updated sweep summary at metrics/dp_sweep_summary.json")
