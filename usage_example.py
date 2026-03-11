"""
Usage example for the PLD_accounting privacy accounting library.

This library computes tight (epsilon, delta)-DP guarantees for the random
allocation subsampling scheme using Privacy Loss Distributions (PLDs).

Installation:
    pip install PLD_accounting

Or from source:
    git clone https://github.com/moshenfeld/PLD_accounting.git
    cd PLD_accounting
    pip install .

Run this example:
    python usage_example.py
"""

import numpy as np

from PLD_accounting.types import (
    PrivacyParams,
    AllocationSchemeConfig,
    Direction,
    BoundType,
)
from PLD_accounting.random_allocation_api import (
    allocation_PLD,
    numerical_allocation_epsilon,
    numerical_allocation_epsilon_range,
)
from PLD_accounting.subsample_PLD import subsample_PLD


# ===========================================================================
# Example 1: Simple Adaptive API (No Structs Required)
#
# The easiest way to get started — just provide the basic parameters:
#   - sigma: Gaussian noise multiplier
#   - num_steps: Total training steps
#   - num_selected: Clients selected per step (k)
#   - delta: Target delta for (ε, δ)-DP
#
# The adaptive API automatically tunes the resolution to achieve accurate
# bounds, returning both upper and lower epsilon estimates.
# ===========================================================================

print("=" * 70)
print("Example 1: Simple Adaptive API (Recommended Starting Point)")
print("=" * 70)

epsilon_upper, epsilon_lower = numerical_allocation_epsilon_range(
    sigma=3.0,
    num_steps=100,
    delta=1e-6,
    num_selected=10,
    epsilon_accuracy=-1,  # -1 means: use 10% of Poisson estimate as target accuracy
)

print(f"\nPrivacy guarantee with automatic resolution tuning:")
print(f"  Parameters: σ=3.0, num_steps=100, num_selected=10, δ=1e-6")
print(f"  Epsilon upper bound: {epsilon_upper:.4f}")
print(f"  Epsilon lower bound: {epsilon_lower:.4f}")
print(f"  Gap: {epsilon_upper - epsilon_lower:.4f} ({100*(epsilon_upper - epsilon_lower)/epsilon_upper:.2f}%)")


# ===========================================================================
# Understanding the Configuration Structs
#
# For more control, you can use the struct-based API. Here's what each
# parameter means:
#
# PrivacyParams:
#   - sigma: Gaussian noise multiplier (higher = more privacy)
#   - num_steps: Total gradient steps across training
#   - num_selected: Number of clients selected per step (k)
#   - num_epochs: Training epochs (usually 1 for federated learning)
#   - delta: Target delta for (ε, δ)-DP
#
# AllocationSchemeConfig:
#   - loss_discretization: Grid spacing for PLD (smaller = tighter, slower)
#   - tail_truncation: Probability mass to truncate from tails
#   - max_grid_mult: Maximum grid size for convolution operations
#
# Direction:
#   - REMOVE: Privacy loss when removing a record
#   - ADD: Privacy loss when adding a record
#   - BOTH: Analyze both directions (most common)
#
# BoundType:
#   - DOMINATES: Upper bound (pessimistic, safe for privacy proofs)
#   - IS_DOMINATED: Lower bound (optimistic, for research)
# ===========================================================================


# ===========================================================================
# Example 2: Direct Epsilon Computation with Structs
#
# This gives you full control over all parameters. Useful when you know
# exactly what resolution and configuration you need.
# ===========================================================================

print("\n" + "=" * 70)
print("Example 2: Direct Epsilon Computation (Full Control)")
print("=" * 70)

delta = 1e-6

params = PrivacyParams(
    sigma=3.0,          # Gaussian noise multiplier
    num_steps=100,      # Total gradient steps across training
    num_selected=10,    # Clients selected per step (k > 1 allocation scheme)
    num_epochs=1,
    delta=delta,
)

config = AllocationSchemeConfig(
    loss_discretization=0.05,     # Coarser = faster but looser bounds
    tail_truncation=delta * 0.1,  # Truncate small probability mass
    max_grid_mult=50_000,         # Maximum grid size for convolution
)

epsilon = numerical_allocation_epsilon(
    params=params,
    config=config,
    direction=Direction.BOTH,      # Analyze both ADD and REMOVE
    bound_type=BoundType.DOMINATES, # Upper bound (safe for proofs)
)

print(f"\nDirect epsilon computation:")
print(f"  Parameters: σ={params.sigma}, num_steps={params.num_steps}, k={params.num_selected}, δ={delta:.0e}")
print(f"  Epsilon: {epsilon:.4f}")


# ===========================================================================
# Example 3: Reusable PLD for Exploring ε-δ Tradeoffs
#
# Computing a PLD (Privacy Loss Distribution) object once lets you
# efficiently query multiple (ε, δ) pairs without recomputing the
# entire convolution.
#
# This is useful when you've completed training and want to understand
# what privacy guarantees you can claim for different delta values.
# ===========================================================================

print("\n" + "=" * 70)
print("Example 3: Reusable PLD for Multiple Delta Values")
print("=" * 70)

pld = allocation_PLD(
    params=params,
    config=config,
    direction=Direction.BOTH,
    bound_type=BoundType.DOMINATES,
)

print(f"\nQuerying epsilon for multiple delta values (single PLD computation):")
for target_delta in [1e-4, 1e-5, 1e-6]:
    eps = pld.get_epsilon_for_delta(target_delta)
    print(f"  δ={target_delta:.0e}  →  ε={eps:.4f}")

print(f"\nYou can also query delta for a given epsilon:")
print(f"  ε=2.0  →  δ={pld.get_delta_for_epsilon(2.0):.2e}")


# ===========================================================================
# Example 4: Advanced — PREAMBLE-style Subsampling + Composition
#
# This models a realistic federated learning setup with three layers:
#   1. Random allocation within each round (k clients per step)
#   2. Subsampling amplification across rounds (probability q)
#   3. Composition across multiple rounds
#
# Privacy Accounting Pipeline:
#   base_pld   = allocation_PLD(...)           # One training round
#        ↓
#   subsampled = subsample_PLD(base_pld, q)    # Amplify by subsampling
#        ↓
#   composed   = subsampled.self_compose(R)    # Compose across R rounds
#        ↓
#   epsilon    = composed.get_epsilon_for_delta(δ)
#
# Configuration Scaling:
#   When composing and subsampling, we need to adjust the config:
#   - loss_discretization scales as 1/√(R·q)  [CLT-like behavior]
#   - tail_truncation scales as 1/(R·q)       [tail accumulation]
# ===========================================================================

print("\n" + "=" * 70)
print("Example 4: PREAMBLE-style — Subsampling + Composition")
print("=" * 70)

# Training setup
sigma        = 3.0    # Large sigma → compact PLD, faster computation
num_steps    = 100    # Steps inside one round
num_selected = 10     # Clients selected per step
q            = 0.05   # Subsampling probability per round
num_rounds   = 50     # Total number of rounds composed

print(f"\nScenario:")
print(f"  - Per round: {num_steps} steps, {num_selected} clients/step, σ={sigma}")
print(f"  - Across rounds: subsampling probability q={q}, {num_rounds} rounds")
print(f"  - Target: (ε, {delta:.0e})-DP\n")

# Scale config to account for composition and subsampling
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

print("Computing privacy guarantee:")
print(f"  [1/4] Computing base PLD for one round...")
base_pld = allocation_PLD(
    params=preamble_params,
    config=config_preamble,
    direction=Direction.BOTH,
)

print(f"  [2/4] Applying subsampling amplification (q={q})...")
subsampled = subsample_PLD(
    pld=base_pld,
    sampling_probability=q,
    bound_type=BoundType.DOMINATES,
)

print(f"  [3/4] Composing across {num_rounds} rounds...")
composed = subsampled.self_compose(num_rounds)

print(f"  [4/4] Querying final epsilon...")
eps_preamble = composed.get_epsilon_for_delta(delta)

print(f"\n✓ Final privacy guarantee: (ε={eps_preamble:.4f}, δ={delta:.0e})")
print(f"  Full parameters: σ={sigma}, t={num_steps}, k={num_selected}, q={q}, R={num_rounds}")

print("\n" + "=" * 70)
print("All examples completed!")
print("=" * 70)
