
"""
Day 4: simplified secure aggregation via pairwise random masking.

Explicitly a proof-of-concept of the underlying PRINCIPLE, not
production cryptography. Real secure aggregation (Bonawitz et al. 2017,
the protocol behind Google's production federated learning) uses
Diffie-Hellman key exchange, Shamir secret sharing for dropout recovery,
and authenticated encryption -- genuinely research-grade engineering,
and correctly out of scope for a 5-day build. This module demonstrates
the core mechanism that makes secure aggregation work at all: pairwise
masks that cancel exactly under summation.

The principle: every pair of clients shares one random mask. Client A
ADDS it, client B SUBTRACTS it (or vice versa -- any consistent,
antisymmetric convention works). Any ONE client's masked update is
indistinguishable from random noise. But summed across every client,
every pairwise mask appears exactly once as +mask and once as -mask, so
they all cancel -- the SUM is exact, even though no individual update
was ever visible. The server only ever needs that sum.

In a real deployment, the pairwise shared value would come from a
Diffie-Hellman exchange between each pair of clients, so the SERVER
never learns it either -- only the two clients in that pair do. Here we
simulate that end state directly (a deterministic seed per pair) since
implementing DH key exchange itself is exactly the "production crypto"
piece the roadmap scoped out.
"""

import numpy as np


def generate_pairwise_seeds(client_ids, master_seed: int = 42) -> dict:
    """
    One seed per unordered pair of clients. In a real protocol this
    value would be the output of a Diffie-Hellman exchange between that
    specific pair -- known only to them, not to the server or any other
    client. Deterministic here (master_seed) purely so this demo is
    reproducible; NOT a real key-derivation function.
    """
    seeds = {}
    ids = sorted(client_ids)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            seeds[(a, b)] = master_seed + (hash((a, b)) % 100_000)
    return seeds


def _pairwise_mask(seed: int, shape) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=1.0, size=shape)


def mask_update(client_id, true_update: np.ndarray, client_ids, pairwise_seeds: dict) -> np.ndarray:
    """
    Client-side. Adds a mask for every pairing this client is part of --
    +mask if this client is the first ("a") in that pair, -mask if it's
    the second ("b"). That antisymmetry is what makes everything cancel
    once every client's masked update gets summed.
    """
    masked = true_update.copy()
    for (a, b), seed in pairwise_seeds.items():
        if client_id not in (a, b):
            continue
        mask = _pairwise_mask(seed, true_update.shape)
        masked = masked + mask if client_id == a else masked - mask
    return masked


def secure_sum(masked_updates: dict) -> np.ndarray:
    """
    Server-side. This is the ONLY operation the server ever performs on
    client updates -- summing the masked values it received. It never
    sees, and cannot recover, any individual client's real update; only
    this sum, in which every pairwise mask has exactly cancelled.
    """
    return np.sum(list(masked_updates.values()), axis=0)


if __name__ == "__main__":
    # Self-check + demo, sized to our actual model's parameter count
    # (SimpleMLP: (21*64+64) + (64*32+32) + (32*1+1) = 3,521 params) so
    # this isn't an abstract toy dimension -- it's the real update size.
    PARAM_COUNT = 21 * 64 + 64 + 64 * 32 + 32 + 32 * 1 + 1
    print(f"Simulating masking over {PARAM_COUNT}-dim updates (our model's real size)\n")

    rng = np.random.default_rng(0)
    client_ids = [0, 1, 2, 3]
    true_updates = {cid: rng.normal(0, 1, PARAM_COUNT) for cid in client_ids}
    pairwise_seeds = generate_pairwise_seeds(client_ids)

    masked_updates = {
        cid: mask_update(cid, true_updates[cid], client_ids, pairwise_seeds)
        for cid in client_ids
    }

    # 1. an individual masked update should look nothing like the real one
    for cid in client_ids:
        diff = np.linalg.norm(masked_updates[cid] - true_updates[cid])
        print(f"client {cid}: ||masked - true|| = {diff:8.2f}  (large = genuinely hidden)")

    # 2. but the SUM of masked updates must exactly equal the true sum
    true_sum = np.sum(list(true_updates.values()), axis=0)
    recovered_sum = secure_sum(masked_updates)
    max_error = float(np.max(np.abs(true_sum - recovered_sum)))
    print(f"\nmax|true_sum - recovered_sum| = {max_error:.2e}  (should be ~0)")

    assert max_error < 1e-8, "masks did not cancel -- masking scheme is broken"
    print("PASSED: every individual update was hidden; the aggregate was exact.")
