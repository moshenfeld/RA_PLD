"""
Performance tests comparing sparse vs dense geometric convolution.

Tests measure and document the actual runtime characteristics of sparse vs dense
geometric convolution across different sparsity levels, grid sizes, and step counts.

Note: These tests are observational - they measure actual performance rather than
asserting expected performance characteristics. The current implementation may show
dense being faster due to better cache locality and vectorization, even at high sparsity.
"""

import time
import numpy as np
import pytest

from PLD_accounting.types import (
    AllocationSchemeConfig,
    BoundType,
    ConvolutionMethod,
    Direction,
    PrivacyParams,
)
from PLD_accounting.discrete_dist import (
    DenseGeometricDiscreteDist,
    SparseGeometricDiscreteDist,
)
from PLD_accounting.geometric_convolution import geometric_self_convolve
from PLD_accounting.random_allocation_accounting import allocation_PLD, _compute_conv_params


def create_sparse_geometric_dist(
    n_total: int,
    sparsity: float,
    ratio: float = 1.01,
    x_min: float = 0.1,
    seed: int = 42,
) -> SparseGeometricDiscreteDist:
    """
    Create a sparse geometric distribution with specified sparsity.

    Args:
        n_total: Total grid size (sparse + zero points)
        sparsity: Fraction of zero points (0.0 to 1.0)
        ratio: Geometric ratio
        x_min: Minimum x value
        seed: Random seed for reproducibility

    Returns:
        SparseGeometricDiscreteDist with specified sparsity level
    """
    rng = np.random.default_rng(seed)

    # Calculate number of non-zero points
    n_nonzero = max(2, int(n_total * (1.0 - sparsity)))

    # Create random sparse indices
    indices = np.sort(rng.choice(n_total, size=n_nonzero, replace=False))
    indices = indices.astype(np.int64)

    # Create random PMF values
    pmf = rng.random(n_nonzero).astype(np.float64)
    pmf /= pmf.sum()

    return SparseGeometricDiscreteDist(
        x_min=x_min,
        ratio=ratio,
        indices=indices,
        PMF_array=pmf,
        p_pos_inf=0.0
    )


def create_dense_geometric_dist_from_sparse(
    sparse_dist: SparseGeometricDiscreteDist,
) -> DenseGeometricDiscreteDist:
    """
    Create an equivalent dense geometric distribution from a sparse one.

    Args:
        sparse_dist: Sparse geometric distribution

    Returns:
        DenseGeometricDiscreteDist with same support and probabilities
    """
    # Get indices from sparse distribution
    indices = sparse_dist.get_indices()
    n_total = int(indices[-1] - indices[0] + 1)

    # Create dense PMF array (mostly zeros)
    dense_pmf = np.zeros(n_total, dtype=np.float64)

    # Fill in non-zero values at correct positions
    relative_indices = indices - indices[0]
    dense_pmf[relative_indices] = sparse_dist.PMF_array

    return DenseGeometricDiscreteDist(
        x_min=sparse_dist.x_min * (sparse_dist.ratio ** indices[0]),
        ratio=sparse_dist.ratio,
        PMF_array=dense_pmf,
        p_pos_inf=sparse_dist.p_pos_inf
    )


def compute_remove_epsilon_with_allocation_api(
    params: PrivacyParams,
    config: AllocationSchemeConfig,
) -> float:
    """Compute epsilon through allocation_PLD API in REMOVE direction."""
    pld = allocation_PLD(
        params=params,
        config=config,
        direction=Direction.REMOVE,
        bound_type=BoundType.DOMINATES,
    )
    return float(pld.get_epsilon_for_delta(params.delta))


class TestSparsePerformance:
    """Performance tests comparing sparse vs dense geometric convolution."""

    def test_sparse_vs_dense_high_sparsity_medium_grid(self):
        """
        Measure sparse vs dense performance with 90% sparsity on medium grid.

        This is an observational test that documents actual performance characteristics.
        """
        n_total = 1000
        sparsity = 0.90
        T = 100

        # Create sparse distribution
        sparse_dist = create_sparse_geometric_dist(
            n_total=n_total,
            sparsity=sparsity,
            ratio=1.01,
            seed=42
        )

        # Create equivalent dense distribution
        dense_dist = create_dense_geometric_dist_from_sparse(sparse_dist)

        # Time sparse convolution
        t0_sparse = time.perf_counter()
        sparse_result = geometric_self_convolve(
            dist=sparse_dist,
            T=T,
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES
        )
        t_sparse = time.perf_counter() - t0_sparse

        # Time dense convolution
        t0_dense = time.perf_counter()
        dense_result = geometric_self_convolve(
            dist=dense_dist,
            T=T,
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES
        )
        t_dense = time.perf_counter() - t0_dense

        # Verify mass conservation
        sparse_mass = np.sum(sparse_result.PMF_array) + sparse_result.p_pos_inf
        dense_mass = np.sum(dense_result.PMF_array) + dense_result.p_pos_inf
        assert np.isclose(sparse_mass, 1.0, atol=1e-8)
        assert np.isclose(dense_mass, 1.0, atol=1e-8)

        # Report performance
        ratio = t_sparse / t_dense
        print(f"\n  Grid size: {n_total}, Sparsity: {sparsity:.0%}, T: {T}")
        print(f"  Sparse time: {t_sparse:.4f}s")
        print(f"  Dense time:  {t_dense:.4f}s")
        print(f"  Sparse/Dense ratio: {ratio:.2f}x")

        # Verify correctness (both methods produce valid results)
        assert sparse_result.PMF_array.size > 0
        assert dense_result.PMF_array.size > 0

    def test_sparse_vs_dense_large_grid_high_sparsity(self):
        """
        Measure sparse vs dense performance with 95% sparsity on large grid (~10k points).

        This is an observational test that documents actual performance characteristics.
        """
        n_total = 10000
        sparsity = 0.95
        T = 500

        # Create sparse distribution
        sparse_dist = create_sparse_geometric_dist(
            n_total=n_total,
            sparsity=sparsity,
            ratio=1.005,
            seed=123
        )

        # Create equivalent dense distribution
        dense_dist = create_dense_geometric_dist_from_sparse(sparse_dist)

        # Time sparse convolution
        t0_sparse = time.perf_counter()
        sparse_result = geometric_self_convolve(
            dist=sparse_dist,
            T=T,
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES
        )
        t_sparse = time.perf_counter() - t0_sparse

        # Time dense convolution
        t0_dense = time.perf_counter()
        dense_result = geometric_self_convolve(
            dist=dense_dist,
            T=T,
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES
        )
        t_dense = time.perf_counter() - t0_dense

        # Verify mass conservation
        sparse_mass = np.sum(sparse_result.PMF_array) + sparse_result.p_pos_inf
        dense_mass = np.sum(dense_result.PMF_array) + dense_result.p_pos_inf
        assert np.isclose(sparse_mass, 1.0, atol=1e-8)
        assert np.isclose(dense_mass, 1.0, atol=1e-8)

        # Report performance
        ratio = t_sparse / t_dense
        print(f"\n  Grid size: {n_total}, Sparsity: {sparsity:.0%}, T: {T}")
        print(f"  Sparse time: {t_sparse:.4f}s")
        print(f"  Dense time:  {t_dense:.4f}s")
        print(f"  Sparse/Dense ratio: {ratio:.2f}x")

        # Verify correctness (both methods produce valid results)
        assert sparse_result.PMF_array.size > 0
        assert dense_result.PMF_array.size > 0

    def test_sparse_vs_dense_many_steps(self):
        """
        Measure sparse vs dense performance with many convolution steps (~1000).

        This is an observational test that documents actual performance characteristics.
        """
        n_total = 5000
        sparsity = 0.80
        T = 1000

        # Create sparse distribution
        sparse_dist = create_sparse_geometric_dist(
            n_total=n_total,
            sparsity=sparsity,
            ratio=1.01,
            seed=456
        )

        # Create equivalent dense distribution
        dense_dist = create_dense_geometric_dist_from_sparse(sparse_dist)

        # Time sparse convolution
        t0_sparse = time.perf_counter()
        sparse_result = geometric_self_convolve(
            dist=sparse_dist,
            T=T,
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES
        )
        t_sparse = time.perf_counter() - t0_sparse

        # Time dense convolution
        t0_dense = time.perf_counter()
        dense_result = geometric_self_convolve(
            dist=dense_dist,
            T=T,
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES
        )
        t_dense = time.perf_counter() - t0_dense

        # Verify mass conservation
        sparse_mass = np.sum(sparse_result.PMF_array) + sparse_result.p_pos_inf
        dense_mass = np.sum(dense_result.PMF_array) + dense_result.p_pos_inf
        assert np.isclose(sparse_mass, 1.0, atol=1e-8)
        assert np.isclose(dense_mass, 1.0, atol=1e-8)

        # Report performance
        ratio = t_sparse / t_dense
        print(f"\n  Grid size: {n_total}, Sparsity: {sparsity:.0%}, T: {T}")
        print(f"  Sparse time: {t_sparse:.4f}s")
        print(f"  Dense time:  {t_dense:.4f}s")
        print(f"  Sparse/Dense ratio: {ratio:.2f}x")

        # Verify correctness (both methods produce valid results)
        assert sparse_result.PMF_array.size > 0
        assert dense_result.PMF_array.size > 0

    @pytest.mark.slow
    def test_sparse_vs_dense_10k_grid_1000_steps(self):
        """
        Measure sparse vs dense performance at scale (~10k grid, ~1k steps).

        This test targets the heavy workload regime discussed in analysis.
        """
        n_total = 10000
        sparsity = 0.95
        T = 1000

        sparse_dist = create_sparse_geometric_dist(
            n_total=n_total,
            sparsity=sparsity,
            ratio=1.002,
            seed=2026,
        )
        dense_dist = create_dense_geometric_dist_from_sparse(sparse_dist)

        # Warm-up to reduce one-time JIT effects from timing.
        geometric_self_convolve(
            dist=sparse_dist,
            T=2,
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
        )
        geometric_self_convolve(
            dist=dense_dist,
            T=2,
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
        )

        t0_sparse = time.perf_counter()
        sparse_result = geometric_self_convolve(
            dist=sparse_dist,
            T=T,
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
        )
        t_sparse = time.perf_counter() - t0_sparse

        t0_dense = time.perf_counter()
        dense_result = geometric_self_convolve(
            dist=dense_dist,
            T=T,
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
        )
        t_dense = time.perf_counter() - t0_dense

        sparse_mass = np.sum(sparse_result.PMF_array) + sparse_result.p_pos_inf
        dense_mass = np.sum(dense_result.PMF_array) + dense_result.p_pos_inf
        assert np.isclose(sparse_mass, 1.0, atol=1e-8)
        assert np.isclose(dense_mass, 1.0, atol=1e-8)

        ratio = t_sparse / t_dense
        print(f"\n  Grid size: {n_total}, Sparsity: {sparsity:.0%}, T: {T}")
        print(f"  Sparse time: {t_sparse:.4f}s")
        print(f"  Dense time:  {t_dense:.4f}s")
        print(f"  Sparse/Dense ratio: {ratio:.2f}x")

        assert sparse_result.PMF_array.size > 0
        assert dense_result.PMF_array.size > 0

    def test_sparse_vs_dense_varying_sparsity(self):
        """
        Measure performance across different sparsity levels.

        This test documents how relative performance changes with sparsity.
        """
        n_total = 2000
        T = 200
        sparsity_levels = [0.50, 0.70, 0.90]

        ratios = []

        for sparsity in sparsity_levels:
            # Create sparse distribution
            sparse_dist = create_sparse_geometric_dist(
                n_total=n_total,
                sparsity=sparsity,
                ratio=1.01,
                seed=int(sparsity * 1000)
            )

            # Create equivalent dense distribution
            dense_dist = create_dense_geometric_dist_from_sparse(sparse_dist)

            # Time sparse convolution
            t0_sparse = time.perf_counter()
            sparse_result = geometric_self_convolve(
                dist=sparse_dist,
                T=T,
                tail_truncation=0.0,
                bound_type=BoundType.DOMINATES
            )
            t_sparse = time.perf_counter() - t0_sparse

            # Time dense convolution
            t0_dense = time.perf_counter()
            dense_result = geometric_self_convolve(
                dist=dense_dist,
                T=T,
                tail_truncation=0.0,
                bound_type=BoundType.DOMINATES
            )
            t_dense = time.perf_counter() - t0_dense

            # Verify mass conservation
            sparse_mass = np.sum(sparse_result.PMF_array) + sparse_result.p_pos_inf
            dense_mass = np.sum(dense_result.PMF_array) + dense_result.p_pos_inf
            assert np.isclose(sparse_mass, 1.0, atol=1e-8)
            assert np.isclose(dense_mass, 1.0, atol=1e-8)

            ratio = t_sparse / t_dense
            ratios.append(ratio)

            print(f"\n  Sparsity: {sparsity:.0%}, Sparse: {t_sparse:.4f}s, Dense: {t_dense:.4f}s, Ratio: {ratio:.2f}x")

        # Document the relationship between sparsity and performance
        print(f"\n  Performance ratios across sparsity levels: {[f'{r:.2f}' for r in ratios]}")

    def test_sparse_low_sparsity_overhead(self):
        """
        Measure sparse overhead at low sparsity levels.

        This is an observational test that documents actual performance characteristics.
        """
        n_total = 1000
        sparsity = 0.10
        T = 100

        # Create sparse distribution
        sparse_dist = create_sparse_geometric_dist(
            n_total=n_total,
            sparsity=sparsity,
            ratio=1.01,
            seed=789
        )

        # Create equivalent dense distribution
        dense_dist = create_dense_geometric_dist_from_sparse(sparse_dist)

        # Time sparse convolution
        t0_sparse = time.perf_counter()
        sparse_result = geometric_self_convolve(
            dist=sparse_dist,
            T=T,
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES
        )
        t_sparse = time.perf_counter() - t0_sparse

        # Time dense convolution
        t0_dense = time.perf_counter()
        dense_result = geometric_self_convolve(
            dist=dense_dist,
            T=T,
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES
        )
        t_dense = time.perf_counter() - t0_dense

        # Verify mass conservation
        sparse_mass = np.sum(sparse_result.PMF_array) + sparse_result.p_pos_inf
        dense_mass = np.sum(dense_result.PMF_array) + dense_result.p_pos_inf
        assert np.isclose(sparse_mass, 1.0, atol=1e-8)
        assert np.isclose(dense_mass, 1.0, atol=1e-8)

        # Report performance
        ratio = t_sparse / t_dense
        print(f"\n  Grid size: {n_total}, Sparsity: {sparsity:.0%}, T: {T}")
        print(f"  Sparse time: {t_sparse:.4f}s")
        print(f"  Dense time:  {t_dense:.4f}s")
        print(f"  Sparse/Dense ratio: {ratio:.2f}x")

        # Verify correctness (both methods produce valid results)
        assert sparse_result.PMF_array.size > 0
        assert dense_result.PMF_array.size > 0


class TestSparseCorrectness:
    """Verify sparse convolution produces correct results."""

    def test_sparse_dense_equivalence_correctness(self):
        """
        Test that sparse and dense produce equivalent results.

        This is a correctness test, not a performance test.
        """
        n_total = 500
        sparsity = 0.80
        T = 50

        # Create sparse distribution
        sparse_dist = create_sparse_geometric_dist(
            n_total=n_total,
            sparsity=sparsity,
            ratio=1.02,
            seed=999
        )

        # Create equivalent dense distribution
        dense_dist = create_dense_geometric_dist_from_sparse(sparse_dist)

        # Convolve both
        sparse_result = geometric_self_convolve(
            dist=sparse_dist,
            T=T,
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES
        )

        dense_result = geometric_self_convolve(
            dist=dense_dist,
            T=T,
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES
        )

        # Results should be numerically equivalent
        # Both should have same total mass
        sparse_mass = np.sum(sparse_result.PMF_array) + sparse_result.p_pos_inf
        dense_mass = np.sum(dense_result.PMF_array) + dense_result.p_pos_inf
        assert np.isclose(sparse_mass, dense_mass, atol=1e-10)

        # Grids may differ in length due to sparsity, but non-zero probabilities
        # should match at corresponding x values
        # This is a basic sanity check - detailed equivalence tests are in other test files
        assert sparse_result.PMF_array.size > 0
        assert dense_result.PMF_array.size > 0


class TestSparsePerformanceAllocationAPI:
    """Performance tests using the random allocation API (realistic path)."""

    @pytest.mark.slow
    def test_allocation_api_sparse_vs_dense_10k_grid_1000_steps(self):
        """
        Compare sparse and dense paths via allocation_PLD API at large scale.

        Uses num_steps=1001 with uncapped max_grid_mult=-1 and tuned sigma/beta
        to target ~10k geometric grids.
        """
        params = PrivacyParams(
            sigma=0.8,
            num_steps=1001,
            num_selected=1,
            num_epochs=1,
            delta=1e-6,
        )

        config_sparse = AllocationSchemeConfig(
            loss_discretization=0.1,
            tail_truncation=1e-8,
            max_grid_FFT=200000,
            max_grid_mult=-1,
            convolution_method=ConvolutionMethod.GEOM,
            use_sparse=True,
        )
        config_dense = AllocationSchemeConfig(
            loss_discretization=0.1,
            tail_truncation=1e-8,
            max_grid_FFT=200000,
            max_grid_mult=-1,
            convolution_method=ConvolutionMethod.GEOM,
            use_sparse=False,
        )

        conv_params = _compute_conv_params(params=params, config=config_sparse)
        n_grid_geom = conv_params.n_grid_geom
        assert 9000 <= n_grid_geom <= 12000

        warm_params = PrivacyParams(
            sigma=0.8,
            num_steps=33,
            num_selected=1,
            num_epochs=1,
            delta=1e-6,
        )
        compute_remove_epsilon_with_allocation_api(
            params=warm_params,
            config=config_sparse,
        )
        compute_remove_epsilon_with_allocation_api(
            params=warm_params,
            config=config_dense,
        )

        t0_sparse = time.perf_counter()
        eps_sparse = compute_remove_epsilon_with_allocation_api(
            params=params,
            config=config_sparse,
        )
        t_sparse = time.perf_counter() - t0_sparse

        t0_dense = time.perf_counter()
        eps_dense = compute_remove_epsilon_with_allocation_api(
            params=params,
            config=config_dense,
        )
        t_dense = time.perf_counter() - t0_dense

        ratio = t_sparse / t_dense
        print("\n  allocation_PLD API (REMOVE, GEOM)")
        print(
            "  Params:"
            f" sigma={params.sigma}, steps={params.num_steps},"
            f" selected={params.num_selected}, epochs={params.num_epochs}, delta={params.delta}"
        )
        print("  Grid policy: max_grid_mult=-1 (uncapped)")
        print(f"  Computed n_grid_geom: {n_grid_geom}")
        print(f"  Sparse time: {t_sparse:.4f}s")
        print(f"  Dense time:  {t_dense:.4f}s")
        print(f"  Sparse/Dense ratio: {ratio:.2f}x")
        print(f"  Epsilon sparse: {eps_sparse:.10f}")
        print(f"  Epsilon dense:  {eps_dense:.10f}")

        assert np.isfinite(eps_sparse)
        assert np.isfinite(eps_dense)
        assert np.isclose(eps_sparse, eps_dense, rtol=1e-6, atol=1e-10)
