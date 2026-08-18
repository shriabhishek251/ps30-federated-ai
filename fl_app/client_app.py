"""
Day 3: Flower ClientApp. Each simulated client (hospital) trains the
shared model on ONLY its own local shard, optionally through Opacus
DP-SGD (use-dp / target-epsilon in pyproject.toml, overridable via
--run-config), then sends back the updated weights and a training-loss
number. Raw patient rows never leave load_client_data() -- only model
weights and scalar metrics cross this boundary.
"""

import torch
from flwr.client import ClientApp
from flwr.common import ArrayRecord, Context, Message, MetricRecord, RecordDict

from .task import eval_fn, get_model, load_client_data, train_fn, train_fn_dp

app = ClientApp()


def _as_bool(value) -> bool:
    """Defensive: --run-config CLI overrides pass values as strings
    (e.g. "false"), which Python treats as truthy if checked naively."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes")


@app.train()
def train(msg: Message, context: Context) -> Message:
    partition_id = context.node_config["partition-id"]
    
    # --- DAY 5 BONUS: SIMULATED CLIENT DROPOUT ---
    # Flower automatically inserts the round number in the message config.
    server_round = int(msg.content["config"].get("server-round", msg.content["config"].get("server_round", 1)))
    
    if server_round == 5 and partition_id == 1:
        print("\n[!] 💥 CRITICAL: SIMULATING NETWORK FAILURE! Client 1 dropping offline mid-round!\n")
        raise RuntimeError("Simulated hospital power outage / network failure.")
    # ---------------------------------------------
    
    batch_size = int(context.run_config["batch-size"])
    local_epochs = int(context.run_config["local-epochs"])
    use_dp = _as_bool(context.run_config["use-dp"])
    lr = float(msg.content["config"]["lr"])
    
    # Extracting the total rounds from the config so we can calculate true cumulative privacy
    total_rounds = int(context.run_config.get("num-rounds", 10))

    model = get_model()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cpu")  # CPU-only by design, see Day 1 hardware notes

    dataloader = load_client_data(partition_id, batch_size)
    metrics = {"num-examples": len(dataloader.dataset)}

    if use_dp:
        target_epsilon = float(context.run_config["target-epsilon"])
        target_delta = float(context.run_config["target-delta"])
        max_grad_norm = float(context.run_config["max-grad-norm"])
        
        # DAY 5 FIX: Pass total_rounds into train_fn_dp for cumulative Opacus tracking
        model, avg_loss, epsilon_spent = train_fn_dp(
            model, dataloader, local_epochs, lr, device,
            target_epsilon, target_delta, max_grad_norm,
            total_rounds  # <--- New parameter passed to task.py
        )
        metrics["epsilon_spent"] = float(epsilon_spent)
        metrics["dp_enabled"] = 1.0
    else:
        avg_loss = train_fn(model, dataloader, local_epochs, lr, device)
        metrics["epsilon_spent"] = -1.0  # sentinel: DP was off this run
        metrics["dp_enabled"] = 0.0

    metrics["train_loss"] = avg_loss

    reply_content = RecordDict({
        "arrays": ArrayRecord(model.state_dict()),
        "metrics": MetricRecord(metrics),
    })
    return Message(content=reply_content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Diagnostic, per-client local evaluation (own shard, not a proper
    held-out split) -- gives round-by-round signal in the terminal that
    training is actually converging. The number that matters for the
    pitch is the CENTRALIZED evaluation in server_app.py, run on the same
    test set as the Day 1 baseline."""
    partition_id = context.node_config["partition-id"]
    batch_size = context.run_config["batch-size"]

    model = get_model()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cpu")

    dataloader = load_client_data(partition_id, batch_size)
    loss, acc = eval_fn(model, dataloader, device)

    reply_content = RecordDict({
        "metrics": MetricRecord({
            "eval_loss": loss,
            "eval_acc": acc,
            "num-examples": len(dataloader.dataset),
        }),
    })
    return Message(content=reply_content, reply_to=msg)