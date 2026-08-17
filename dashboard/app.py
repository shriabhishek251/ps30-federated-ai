"""
Day 4: the live dashboard -- ties Day 1's baseline, Day 2's federated
result, Day 3's DP sweep, and Day 4's secure-aggregation demo into one
view. This is the artifact you actually present from.

Run:
    streamlit run dashboard/app.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from secure_agg import generate_pairwise_seeds, mask_update, secure_sum  # noqa: E402

METRICS_DIR = os.path.join(os.path.dirname(__file__), "..", "metrics")

SWEEP_LABELS = {
    "dp-off": "Federated (no DP)",
    "eps8": "Federated + DP (\u03b5=8)",
    "eps4": "Federated + DP (\u03b5=4)",
    "eps1": "Federated + DP (\u03b5=1)",
}


# ---------------------------------------------------------------------
# Pure data-loading/transform functions, kept separate from st.* calls
# so they're testable without Streamlit installed at all.
# ---------------------------------------------------------------------

def load_json(name: str):
    path = os.path.join(METRICS_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def build_comparison_table(baseline, sweep: dict) -> pd.DataFrame:
    """One row per setting we have real results for -- baseline, then
    each DP sweep point that's actually been run. Skips anything
    missing rather than showing fabricated zeros."""
    rows = []
    if baseline:
        rows.append({
            "Setting": "Centralized baseline",
            "Accuracy": baseline.get("accuracy"),
            "F1": baseline.get("f1"),
            "ROC-AUC": baseline.get("roc_auc"),
        })
    for tag, label in SWEEP_LABELS.items():
        if tag in sweep:
            r = sweep[tag]
            rows.append({
                "Setting": label,
                "Accuracy": r.get("accuracy"),
                "F1": r.get("f1"),
                "ROC-AUC": r.get("roc_auc"),
            })
    return pd.DataFrame(rows)


def build_history_df(history: list) -> pd.DataFrame:
    df = pd.DataFrame(history)
    if "round" in df.columns:
        df = df.set_index("round")
    return df


def run_secure_agg_demo(seed: int, param_count: int = 8, client_ids=(0, 1, 2, 3)):
    """Small param_count (8, not the real 3,521) purely so the numbers
    are readable in a table -- the underlying math is identical, see
    src/secure_agg.py's own self-test for the full-size version."""
    rng = np.random.default_rng(seed)
    true_updates = {cid: rng.normal(0, 1, param_count) for cid in client_ids}
    pairwise_seeds = generate_pairwise_seeds(list(client_ids), master_seed=seed)
    masked_updates = {
        cid: mask_update(cid, true_updates[cid], list(client_ids), pairwise_seeds)
        for cid in client_ids
    }
    true_sum = np.sum(list(true_updates.values()), axis=0)
    recovered_sum = secure_sum(masked_updates)
    max_error = float(np.max(np.abs(true_sum - recovered_sum)))
    return true_updates, masked_updates, true_sum, recovered_sum, max_error


# ---------------------------------------------------------------------
# Streamlit page
# ---------------------------------------------------------------------

def main():
    st.set_page_config(page_title="PS30 \u2014 Federated AI Dashboard", layout="wide")
    st.title("Privacy-Preserving Federated AI \u2014 Live Dashboard")
    st.caption(
        "One shared model trained across simulated hospitals, none of which "
        "ever shares raw patient data \u2014 with differential privacy and "
        "secure aggregation on top."
    )

    baseline = load_json("baseline.json")
    sweep = load_json("dp_sweep_summary.json") or {}

    st.header("1. Accuracy vs. privacy: the full trade-off")
    comparison = build_comparison_table(baseline, sweep)
    if not comparison.empty:
        st.bar_chart(comparison.set_index("Setting")[["F1", "ROC-AUC"]])
        st.dataframe(comparison, hide_index=True, use_container_width=True)
        st.caption(
            "Watch F1 drop sharply the instant DP turns on, while ROC-AUC "
            "barely moves \u2014 gradient clipping caps every example's update "
            "equally, partially undoing the class-imbalance fix from the "
            "baseline. The model still ranks correctly; the 0.5 decision "
            "threshold is just no longer optimal for its shifted output."
        )
    else:
        st.warning("No metrics yet \u2014 run `python src/baseline.py` and the "
                   "`flwr run` sweep first.")

    st.header("2. Explore a specific privacy budget")
    choice = st.select_slider("Privacy budget (\u03b5)", options=["off", "8", "4", "1"], value="4")
    tag = "dp-off" if choice == "off" else f"eps{choice}"

    history = load_json(f"federated_history_{tag}.json")
    if history:
        hdf = build_history_df(history)
        cols_present = [c for c in ["accuracy", "f1", "roc_auc"] if c in hdf.columns]
        st.line_chart(hdf[cols_present])

        latest = sweep.get(tag, {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{latest.get('accuracy', 0):.3f}")
        c2.metric("F1", f"{latest.get('f1', 0):.3f}")
        c3.metric("ROC-AUC", f"{latest.get('roc_auc', 0):.3f}")
    else:
        run_cmd = ('flwr run . --run-config "use-dp=false"' if tag == "dp-off"
                   else f'flwr run . --run-config "use-dp=true target-epsilon={choice}"')
        st.info(f"No run recorded for this setting yet. Run:\n\n`{run_cmd}`")

    st.header("3. Secure aggregation: what the server actually sees")
    st.caption(
        "Simplified pairwise masking \u2014 a proof-of-concept of the underlying "
        "principle, explicitly not production cryptography (see README). "
        "Shown here at 8 dimensions for readability; src/secure_agg.py's own "
        "self-test runs the identical math at our model's real 3,521-parameter size."
    )

    if "sa_seed" not in st.session_state:
        st.session_state["sa_seed"] = 0
    if st.button("Regenerate client updates"):
        st.session_state["sa_seed"] = np.random.randint(0, 1_000_000)

    true_updates, masked_updates, true_sum, recovered_sum, max_error = run_secure_agg_demo(
        st.session_state["sa_seed"]
    )

    client_ids = list(true_updates.keys())
    col1, col2 = st.columns(2)
    with col1:
        st.write("**True local updates (never transmitted)**")
        true_df = pd.DataFrame(true_updates).T
        true_df.index = [f"Client {c}" for c in client_ids]
        st.dataframe(true_df.style.format("{:.2f}").background_gradient(cmap="RdBu", axis=None))
    with col2:
        st.write("**Masked updates (what the server actually receives)**")
        masked_df = pd.DataFrame(masked_updates).T
        masked_df.index = [f"Client {c}" for c in client_ids]
        st.dataframe(masked_df.style.format("{:.2f}").background_gradient(cmap="RdBu", axis=None))

    st.write("**Sum check \u2014 the only computation the server ever performs:**")
    sum_df = pd.DataFrame({"True sum": true_sum, "Recovered from masked sum": recovered_sum})
    st.dataframe(sum_df.T.style.format("{:.4f}"))

    if max_error < 1e-8:
        st.success(
            f"Max difference between true and recovered sum: {max_error:.2e}. "
            f"The masks cancelled exactly \u2014 no individual client's update "
            f"was ever visible to the server, only this aggregate."
        )
    else:
        st.error(f"Max difference: {max_error:.2e} \u2014 masks did not cancel correctly.")


if __name__ == "__main__":
    main()
