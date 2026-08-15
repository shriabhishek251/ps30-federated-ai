"""
Day 2: Flower ServerApp. Runs FedAvg for N rounds across the 4 simulated
clients (weighted by each client's local data size), then, after every
round, evaluates the aggregated GLOBAL model centrally -- on the EXACT
same held-out test set and threshold as the Day 1 baseline -- so the two
numbers are directly, honestly comparable, not apples to oranges.
"""

import json
import os

import torch
from flwr.common import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.server import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from .task import METRICS_DIR, centralized_evaluate, get_model

app = ServerApp()


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
    num_rounds = context.run_config["num-server-rounds"]
    lr = context.run_config["learning-rate"]
    fraction_evaluate = context.run_config["fraction-evaluate"]

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

    # save the final global model
    final_model = get_model()
    final_model.load_state_dict(result.arrays.to_torch_state_dict())
    os.makedirs(METRICS_DIR, exist_ok=True)
    torch.save(final_model.state_dict(), os.path.join(METRICS_DIR, "federated_model.pt"))

    # save round-by-round centralized-eval history for the Day 4 dashboard
    with open(os.path.join(METRICS_DIR, "federated_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    final = history[-1] if history else {}
    print(f"\nFinal federated (centralized-eval) result: {final}")
    print(f"Compare directly to Day 1 baseline: accuracy=0.7241 f1=0.444 roc_auc=0.826")
    print(f"Saved final model to metrics/federated_model.pt")
    print(f"Saved round history to metrics/federated_history.json")
