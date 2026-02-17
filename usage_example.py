"""
Usage example for the RA-PLD privacy accounting library.

This library computes tight (epsilon, delta)-DP guarantees for the random
allocation subsampling scheme using Privacy Loss Distributions (PLDs).
"""

import numpy as np

from PLD_accounting.types import (
    PrivacyParams,
    AllocationSchemeConfig,
    Direction,
    BoundType,
)
from PLD_accounting.random_allocation_accounting import (
    numerical_allocation_epsilon,
    allocation_PLD,
)
from PLD_accounting.subsample_PLD import subsample_PLD


# ---------------------------------------------------------------------------
# Example 1: Compute epsilon directly (k=10, t=1000)
#
# Matches the epsilon-from-sigma experiment with num_selected=10.
# Using sigma=3.0 and coarser resolution for reasonable runtime.
# ---------------------------------------------------------------------------

delta = 1e-6

params = PrivacyParams(
    sigma=3.0,          # Gaussian noise multiplier
    num_steps=1000,     # Total gradient steps across training
    num_selected=10,    # Clients selected per step (k > 1 allocation scheme)
    num_epochs=1,
    delta=delta,
)

config = AllocationSchemeConfig(
    loss_discretization=0.05,
    tail_truncation=delta * 0.1,
    max_grid_mult=50_000,
)

epsilon = numerical_allocation_epsilon(
    params=params,
    config=config,
    direction=Direction.BOTH,
    bound_type=BoundType.DOMINATES,
)

print(f"[Example 1] Epsilon for sigma={params.sigma}, t={params.num_steps}, "
      f"k={params.num_selected}, delta={delta:.0e}: {epsilon:.4f}")


# ---------------------------------------------------------------------------
# Example 2: Create a PLD object and query epsilon for multiple delta values
#
# Useful when you want to explore the privacy-delta trade-off without
# rerunning the convolution for each delta.
# ---------------------------------------------------------------------------

pld = allocation_PLD(
    params=params,
    config=config,
    direction=Direction.BOTH,
    bound_type=BoundType.DOMINATES,
)

print("\n[Example 2] Epsilon vs delta from a single PLD computation:")
for target_delta in [1e-4, 1e-5, 1e-6]:
    eps = pld.get_epsilon_for_delta(target_delta)
    print(f"  delta={target_delta:.0e}  ->  epsilon={eps:.4f}")


# ---------------------------------------------------------------------------
# Example 3: PREAMBLE-style analysis — subsample + compose
#
# Models a federated learning setup where the allocation scheme is applied
# per round, followed by subsampling clients and composing across rounds.
#
# Pattern:
#   base_PLD  = allocation PLD for one training round
#   subsampled = amplify via subsampling (Poisson with probability q)
#   composed   = privacy after R repeated rounds
# ---------------------------------------------------------------------------

# Training setup
sigma        = 3.0    # Large sigma → compact PLD, faster computation
num_steps    = 100    # Steps inside one round
num_selected = 10     # Clients selected per step
q            = 0.05   # Subsampling probability per round
num_rounds   = 50     # Total number of rounds composed

# Scale config to account for composition and subsampling.
# loss_discretization scales as 1/sqrt(num_rounds * q) (central-limit-theorem-like).
# tail_truncation must cover the full composed tail, so it scales as 1/(num_rounds * q).
scale = num_rounds * q
config_preamble = AllocationSchemeConfig(
    loss_discretization=config.loss_discretization / np.sqrt(scale),
    tail_truncation=config.tail_truncation / scale,
    max_grid_mult=config.max_grid_mult,
)

preamble_params = PrivacyParams(
    sigma=sigma,
    num_steps=num_steps,
    num_selected=num_selected,
    num_epochs=1,
    delta=delta,
)

base_pld = allocation_PLD(
    params=preamble_params,
    config=config_preamble,
    direction=Direction.BOTH,
)

subsampled = subsample_PLD(
    pld=base_pld,
    sampling_probability=q,
    bound_type=BoundType.DOMINATES,
)

composed = subsampled.self_compose(num_rounds)

eps_preamble = composed.get_epsilon_for_delta(delta)
print(f"\n[Example 3] PREAMBLE-style: sigma={sigma}, t={num_steps}, k={num_selected}, "
      f"q={q}, R={num_rounds}, delta={delta:.0e}  ->  epsilon={eps_preamble:.4f}")
