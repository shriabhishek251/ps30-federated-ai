"""
Day 2: Flower ClientApp. Each simulated client (hospital) trains the
shared model on ONLY its own local shard, then sends back the updated
weights and a training-loss number. Raw patient rows never leave
load_client_data() -- only model weights and scalar metrics cross this
boundary, which is the entire point of federated learning.
"""

import torch
from flwr.client import ClientApp
from flwr.common import ArrayRecord, Context, Message, MetricRecord, RecordDict

from .task import eval_fn, get_model, load_client_data, train_fn

app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    partition_id = context.node_config["partition-id"]
    batch_size = context.run_config["batch-size"]
    local_epochs = context.run_config["local-epochs"]
    lr = msg.content["config"]["lr"]

    model = get_model()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cpu")  # CPU-only by design, see Day 1 hardware notes

    dataloader = load_client_data(partition_id, batch_size)
    avg_loss = train_fn(model, dataloader, local_epochs, lr, device)

    reply_content = RecordDict({
        "arrays": ArrayRecord(model.state_dict()),
        "metrics": MetricRecord({
            "train_loss": avg_loss,
            "num-examples": len(dataloader.dataset),
        }),
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
