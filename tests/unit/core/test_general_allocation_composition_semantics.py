"""Unit tests for random-allocation composition wiring."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import PLD_accounting.random_allocation_api as random_allocation_api_module
import PLD_accounting.random_allocation_accounting as random_allocation_accounting_module
from PLD_accounting.discrete_dist import LinearDiscreteDist, PLDRealization
from PLD_accounting.random_allocation_accounting import _allocation_PMF_core as allocation_PMF_core
from PLD_accounting.random_allocation_api import gaussian_allocation_PLD, general_allocation_PLD
from PLD_accounting.types import AllocationSchemeConfig, BoundType, ConvolutionMethod, Direction, PrivacyParams


def _simple_realization() -> PLDRealization:
    return PLDRealization(
        x_min=0.0,
        x_gap=0.5,
        PMF_array=np.array([0.6, 0.3, 0.1]),
    )


def _stub_linear_dist() -> LinearDiscreteDist:
    return LinearDiscreteDist(
        x_min=0.0,
        x_gap=0.5,
        PMF_array=np.array([0.5, 0.3, 0.2]),
    )


class TestGeneralAllocationPLDWiring:
    def test_general_allocation_uses_shared_allocation_pld(self, monkeypatch: pytest.MonkeyPatch):
        captured: dict[str, Any] = {}
        sentinel_pld = object()

        def fake_allocation_PLD(
            *,
            num_steps: int,
            num_selected: int,
            num_epochs: int,
            compute_base_pmf_remove,
            compute_base_pmf_add,
            loss_discretization: float,
            tail_truncation: float,
            bound_type: BoundType,
        ):
            captured.update(
                {
                    "num_steps": num_steps,
                    "num_selected": num_selected,
                    "num_epochs": num_epochs,
                    "compute_base_pmf_remove": compute_base_pmf_remove,
                    "compute_base_pmf_add": compute_base_pmf_add,
                    "loss_discretization": loss_discretization,
                    "tail_truncation": tail_truncation,
                    "bound_type": bound_type,
                }
            )
            return sentinel_pld

        monkeypatch.setattr(random_allocation_api_module, "allocation_PLD", fake_allocation_PLD)

        config = AllocationSchemeConfig(convolution_method=ConvolutionMethod.GEOM)
        remove_realization = _simple_realization()
        add_realization = _simple_realization()
        result = general_allocation_PLD(
            num_steps=23,
            num_selected=5,
            num_epochs=4,
            remove_realization=remove_realization,
            add_realization=add_realization,
            config=config,
            bound_type=BoundType.IS_DOMINATED,
        )

        assert result is sentinel_pld
        assert captured["num_steps"] == 23
        assert captured["num_selected"] == 5
        assert captured["num_epochs"] == 4
        assert captured["loss_discretization"] == config.loss_discretization
        assert captured["tail_truncation"] == config.tail_truncation
        assert captured["bound_type"] == BoundType.IS_DOMINATED

        remove_builder = captured["compute_base_pmf_remove"]
        add_builder = captured["compute_base_pmf_add"]
        assert callable(remove_builder)
        assert callable(add_builder)
        assert remove_builder.func is random_allocation_api_module.geometric_allocation_PMF_base_remove
        assert add_builder.func is random_allocation_api_module.geometric_allocation_PMF_base_add
        remove_base_creation = remove_builder.keywords["base_distributions_creation"]
        add_base_creation = add_builder.keywords["base_distributions_creation"]
        assert remove_base_creation.func is random_allocation_api_module.realization_remove_base_distributions
        assert add_base_creation.func is random_allocation_api_module.realization_add_base_distribution
        assert remove_base_creation.keywords == {"realization": remove_realization}
        assert add_base_creation.keywords == {"realization": add_realization}

    def test_general_allocation_rejects_num_steps_less_than_num_selected(self):
        with pytest.raises(ValueError, match="num_steps must be >= num_selected"):
            general_allocation_PLD(
                num_steps=3,
                num_selected=4,
                num_epochs=1,
                remove_realization=_simple_realization(),
                add_realization=_simple_realization(),
                config=AllocationSchemeConfig(convolution_method=ConvolutionMethod.GEOM),
            )


class TestGaussianAllocationPLDWiring:
    def test_gaussian_allocation_uses_shared_allocation_pld(self, monkeypatch: pytest.MonkeyPatch):
        captured: dict[str, Any] = {}
        sentinel_pld = object()

        def fake_allocation_PLD(
            *,
            num_steps: int,
            num_selected: int,
            num_epochs: int,
            compute_base_pmf_remove,
            compute_base_pmf_add,
            loss_discretization: float,
            tail_truncation: float,
            bound_type: BoundType,
        ):
            captured.update(
                {
                    "num_steps": num_steps,
                    "num_selected": num_selected,
                    "num_epochs": num_epochs,
                    "compute_base_pmf_remove": compute_base_pmf_remove,
                    "compute_base_pmf_add": compute_base_pmf_add,
                    "loss_discretization": loss_discretization,
                    "tail_truncation": tail_truncation,
                    "bound_type": bound_type,
                }
            )
            return sentinel_pld

        monkeypatch.setattr(random_allocation_api_module, "allocation_PLD", fake_allocation_PLD)

        config = AllocationSchemeConfig(convolution_method=ConvolutionMethod.BEST_OF_TWO)
        params = PrivacyParams(
            sigma=1.75,
            num_steps=19,
            num_selected=4,
            num_epochs=3,
        )
        result = gaussian_allocation_PLD(
            params=params,
            config=config,
            bound_type=BoundType.DOMINATES,
        )

        assert result is sentinel_pld
        assert captured["num_steps"] == 19
        assert captured["num_selected"] == 4
        assert captured["num_epochs"] == 3
        assert captured["loss_discretization"] == config.loss_discretization
        assert captured["tail_truncation"] == config.tail_truncation
        assert captured["bound_type"] == BoundType.DOMINATES

        remove_builder = captured["compute_base_pmf_remove"]
        add_builder = captured["compute_base_pmf_add"]
        assert callable(remove_builder)
        assert callable(add_builder)
        assert remove_builder.func is random_allocation_api_module.gaussian_allocation_PMF_core
        assert add_builder.func is random_allocation_api_module.gaussian_allocation_PMF_core
        assert remove_builder.keywords == {
            "direction": Direction.REMOVE,
            "sigma": params.sigma,
            "config": config,
        }
        assert add_builder.keywords == {
            "direction": Direction.ADD,
            "sigma": params.sigma,
            "config": config,
        }


class TestAllocationFinalization:
    def test_allocation_pmf_core_regrids_before_and_after_compose(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        captured: dict[str, float] = {}

        def fake_fft_self_convolve(
            *,
            dist: LinearDiscreteDist,
            T: int,
            tail_truncation: float,
            bound_type: BoundType,
            use_direct: bool,
        ) -> LinearDiscreteDist:
            del tail_truncation, bound_type, use_direct
            captured["base_gap_at_compose"] = dist.x_gap
            captured["num_epochs"] = float(T)
            return dist

        def fake_compute_base_pmf(
            *,
            num_steps: int,
            loss_discretization: float,
            tail_truncation: float,
            bound_type: BoundType,
        ) -> LinearDiscreteDist:
            del tail_truncation, bound_type
            captured["num_steps"] = float(num_steps)
            captured["core_loss_discretization"] = loss_discretization
            return _stub_linear_dist()

        monkeypatch.setattr(
            random_allocation_accounting_module,
            "FFT_self_convolve",
            fake_fft_self_convolve,
        )

        config = AllocationSchemeConfig(
            loss_discretization=0.3,
            tail_truncation=1e-8,
            convolution_method=ConvolutionMethod.FFT,
        )
        result = allocation_PMF_core(
            num_steps=7,
            num_epochs=5,
            compute_base_pmf=fake_compute_base_pmf,
            loss_discretization=config.loss_discretization,
            tail_truncation=config.tail_truncation,
            bound_type=BoundType.DOMINATES,
        )

        output_loss_discretization = config.loss_discretization / 3
        expected_core_loss = output_loss_discretization / np.sqrt(5)

        assert captured["num_steps"] == 7.0
        assert captured["num_epochs"] == 5.0
        assert np.isclose(
            captured["base_gap_at_compose"],
            expected_core_loss,
            atol=1e-12,
        )
        assert np.isclose(result.x_gap, output_loss_discretization, atol=1e-12)


class TestGeometricBaseTailScaling:
    def test_remove_base_factor_tail_scales_with_num_steps(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        captured_tails: list[float] = []
        sentinel = object()

        def fake_base_distributions_creation(
            *,
            loss_discretization: float,
            tail_truncation: float,
            bound_type: BoundType,
        ) -> tuple[LinearDiscreteDist, LinearDiscreteDist]:
            del loss_discretization, bound_type
            captured_tails.append(tail_truncation)
            return _stub_linear_dist(), _stub_linear_dist()

        monkeypatch.setattr(random_allocation_accounting_module, "exp_linear_to_geometric", lambda _dist: sentinel)
        monkeypatch.setattr(random_allocation_accounting_module, "geometric_self_convolve", lambda **_kwargs: sentinel)
        monkeypatch.setattr(random_allocation_accounting_module, "geometric_convolve", lambda **_kwargs: sentinel)
        monkeypatch.setattr(random_allocation_accounting_module, "log_geometric_to_linear", lambda _dist: _stub_linear_dist())

        for num_steps in (5, 10):
            random_allocation_accounting_module.geometric_allocation_PMF_base_remove(
                base_distributions_creation=fake_base_distributions_creation,
                num_steps=num_steps,
                loss_discretization=0.1,
                tail_truncation=1e-6,
                bound_type=BoundType.DOMINATES,
            )

        assert len(captured_tails) == 2
        assert np.isclose(captured_tails[0] * 5, captured_tails[1] * 10, atol=1e-24)

    def test_add_base_factor_tail_scales_with_num_steps(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        captured_tails: list[float] = []
        sentinel = object()

        def fake_base_distributions_creation(
            *,
            loss_discretization: float,
            tail_truncation: float,
            bound_type: BoundType,
        ) -> LinearDiscreteDist:
            del loss_discretization, bound_type
            captured_tails.append(tail_truncation)
            return _stub_linear_dist()

        monkeypatch.setattr(random_allocation_accounting_module, "exp_linear_to_geometric", lambda _dist: sentinel)
        monkeypatch.setattr(random_allocation_accounting_module, "geometric_self_convolve", lambda **_kwargs: sentinel)
        monkeypatch.setattr(random_allocation_accounting_module, "log_geometric_to_linear", lambda _dist: _stub_linear_dist())
        monkeypatch.setattr(random_allocation_accounting_module, "negate_reverse_linear_distribution", lambda _dist: _stub_linear_dist())

        for num_steps in (5, 10):
            random_allocation_accounting_module.geometric_allocation_PMF_base_add(
                base_distributions_creation=fake_base_distributions_creation,
                num_steps=num_steps,
                loss_discretization=0.1,
                tail_truncation=1e-6,
                bound_type=BoundType.DOMINATES,
            )

        assert len(captured_tails) == 2
        assert np.isclose(captured_tails[0] * 5, captured_tails[1] * 10, atol=1e-24)
