"""
Integration tests for subsampling behavior.
"""
from functools import partial
import math
import numpy as np
import pytest

from PLD_accounting.discrete_dist import GeneralDiscreteDist, LinearDiscreteDist, PLDRealization
from PLD_accounting.subsample_PLD import (
    subsample_PMF,
)
from PLD_accounting.types import (
    AllocationSchemeConfig,
    BoundType,
    ConvolutionMethod,
    Direction,
    PrivacyParams,
)
from PLD_accounting.random_allocation_accounting import _allocation_PMF_core as allocation_PMF_core
from PLD_accounting.random_allocation_gaussian import gaussian_allocation_PMF_core


def _upper_to_lower(dist: GeneralDiscreteDist) -> GeneralDiscreteDist:
    if dist.p_neg_inf > 0:
        raise ValueError("Expected p_neg_inf=0 for upper PLD")
    losses = dist.x_array
    probs = dist.PMF_array
    lower_probs = np.zeros_like(probs)
    mask = probs > 0
    lower_probs[mask] = np.exp(np.log(probs[mask]) - losses[mask])
    sum_prob = float(np.sum(lower_probs))
    return GeneralDiscreteDist(
        x_array=losses,
        PMF_array=lower_probs,
        p_neg_inf=max(0.0, 1.0 - sum_prob),
        p_pos_inf=0.0,
    )


def _negate_distribution(dist: GeneralDiscreteDist) -> GeneralDiscreteDist:
    return GeneralDiscreteDist(
        x_array=-np.flip(dist.x_array),
        PMF_array=np.flip(dist.PMF_array),
        p_neg_inf=dist.p_pos_inf,
        p_pos_inf=dist.p_neg_inf,
    )


def _as_realization(dist: LinearDiscreteDist) -> PLDRealization:
    return PLDRealization(
        x_min=dist.x_min,
        x_gap=dist.x_gap,
        PMF_array=dist.PMF_array,
        p_loss_inf=dist.p_pos_inf,
        p_loss_neg_inf=dist.p_neg_inf,
    )


def _total_mass(pmf) -> float:
    dense = pmf.to_dense_pmf()
    finite = math.fsum(dense._probs)
    return finite + dense._infinity_mass


def _gaussian_single_component_pmf(*,
    num_steps: int,
    sigma: float,
    config: AllocationSchemeConfig,
    direction: Direction,
    bound_type: BoundType,
) -> LinearDiscreteDist:
    return allocation_PMF_core(
        num_steps=num_steps,
        num_epochs=1,
        compute_base_pmf=partial(
            gaussian_allocation_PMF_core,
            direction=direction,
            sigma=sigma,
            config=config,
            bound_type=bound_type,
        ),
        loss_discretization=config.loss_discretization,
        tail_truncation=config.tail_truncation,
        bound_type=bound_type,
    )



class TestPLDDualRealisticScenarios:
    """Tests for PLD-dual on realistic random allocation scenarios."""

    def test_realistic_allocation_remove(self):
        params = PrivacyParams(
            sigma=1.0,
            num_steps=10,
            num_selected=5,
            delta=1e-6
        )

        config = AllocationSchemeConfig(
            loss_discretization=1e-3,
            tail_truncation=1e-8,
            max_grid_mult=10_000,
            max_grid_FFT=1_000_000,
            convolution_method=ConvolutionMethod.GEOM
        )

        new_num_steps = params.num_steps // params.num_selected
        remove_upper = _gaussian_single_component_pmf(
            num_steps=new_num_steps,
            sigma=params.sigma,
            config=config,
            direction=Direction.REMOVE,
            bound_type=BoundType.DOMINATES,
        )

        subsampled = subsample_PMF(
            base_pld=_as_realization(remove_upper),
            sampling_prob=0.01,
            direction=Direction.REMOVE,
        )

        total_mass = np.sum(subsampled.PMF_array) + subsampled.p_neg_inf + subsampled.p_pos_inf
        assert abs(total_mass - 1.0) < 1e-5
        assert np.sum(subsampled.PMF_array > 1e-10) > 5

    def test_realistic_allocation_add(self):
        params = PrivacyParams(
            sigma=1.0,
            num_steps=10,
            num_selected=5,
            delta=1e-6
        )

        config = AllocationSchemeConfig(
            loss_discretization=1e-3,
            tail_truncation=1e-8,
            max_grid_mult=10_000,
            max_grid_FFT=1_000_000,
            convolution_method=ConvolutionMethod.GEOM
        )

        new_num_steps = params.num_steps // params.num_selected
        add_upper = _gaussian_single_component_pmf(
            num_steps=new_num_steps,
            sigma=params.sigma,
            config=config,
            direction=Direction.ADD,
            bound_type=BoundType.DOMINATES,
        )

        subsampled = subsample_PMF(
            base_pld=_as_realization(add_upper),
            sampling_prob=0.01,
            direction=Direction.ADD,
        )

        total_mass = np.sum(subsampled.PMF_array) + subsampled.p_neg_inf + subsampled.p_pos_inf
        assert abs(total_mass - 1.0) < 1e-5
        assert np.sum(subsampled.PMF_array > 1e-10) > 5

    @pytest.mark.parametrize("sampling_prob", [0.001, 0.01, 0.1])
    def test_realistic_allocation_various_sampling_probs(self, sampling_prob):
        params = PrivacyParams(
            sigma=1.0,
            num_steps=10,
            num_selected=3,
            delta=1e-6
        )

        config = AllocationSchemeConfig(
            loss_discretization=1e-3,
            tail_truncation=1e-8,
            max_grid_mult=10_000,
            max_grid_FFT=1_000_000,
            convolution_method=ConvolutionMethod.GEOM
        )

        new_num_steps = params.num_steps // params.num_selected
        for direction in [Direction.REMOVE, Direction.ADD]:
            upper = _gaussian_single_component_pmf(
                num_steps=new_num_steps,
                sigma=params.sigma,
                config=config,
                direction=direction,
                bound_type=BoundType.DOMINATES,
            )

            subsampled = subsample_PMF(
                base_pld=_as_realization(upper),
                sampling_prob=sampling_prob,
                direction=direction,
            )

            total_mass = np.sum(subsampled.PMF_array) + subsampled.p_neg_inf + subsampled.p_pos_inf
            assert abs(total_mass - 1.0) < 1e-5
