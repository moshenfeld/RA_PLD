"""Integration tests for adaptive random-allocation queries."""

import warnings

import numpy as np
import pytest

import PLD_accounting.random_allocation_api as random_allocation_api_module
from PLD_accounting import (
    AllocationSchemeConfig,
    PrivacyParams,
    numerical_allocation_delta,
    numerical_allocation_delta_range,
    numerical_allocation_epsilon,
    numerical_allocation_epsilon_range,
)
from PLD_accounting.adaptive_random_allocation import (
    AdaptiveResult,
    adaptive_convergence,
    estimate_poisson_query,
)
from PLD_accounting.random_allocation_api import (
    allocation_PLD as allocation_pld_api,
    numerical_allocation_delta as numerical_allocation_delta_api,
    numerical_allocation_delta_range as numerical_allocation_delta_range_api,
    numerical_allocation_epsilon as numerical_allocation_epsilon_api,
    numerical_allocation_epsilon_range as numerical_allocation_epsilon_range_api,
)
from PLD_accounting.random_allocation_accounting import allocation_PMF, compute_conv_params
from PLD_accounting.types import BoundType, ConvolutionMethod, Direction


def _stub_result(target_accuracy: float, value: float = 0.5, converged: bool = True) -> AdaptiveResult:
    lower_bound = value - max(target_accuracy / 2.0, 1e-12)
    return AdaptiveResult(
        upper_bound=value,
        lower_bound=lower_bound,
        absolute_gap=value - lower_bound,
        converged=converged,
        iterations=2,
        initial_discretization=0.1,
        discretization=0.05,
        initial_tail_truncation=1e-6,
        tail_truncation=5e-7,
        target_accuracy=target_accuracy,
    )


def _run_real_epsilon_adaptive(
    *,
    sigma: float,
    num_steps: int,
    delta: float,
    epsilon_accuracy: float,
    num_selected: int = 1,
    num_epochs: int = 1,
) -> AdaptiveResult:
    params = PrivacyParams(
        sigma=sigma,
        num_steps=num_steps,
        num_selected=num_selected,
        num_epochs=num_epochs,
        delta=delta,
    )

    def epsilon_discretization_gap(pld, upper_value: float, lower_value: float, discretization: float) -> float:
        return max(0.0, upper_value - lower_value)

    def epsilon_tail_gap(pld, upper_value: float, lower_value: float, tail_truncation: float) -> float:
        return max(
            0.0,
            float(pld.get_epsilon_for_delta(delta)) - float(pld.get_epsilon_for_delta(delta + tail_truncation)),
        )

    return adaptive_convergence(
        params=params,
        target_accuracy=epsilon_accuracy,
        initial_discretization=epsilon_accuracy * 4,
        initial_tail_truncation=delta / 10,
        query_func=lambda pld: float(pld.get_epsilon_for_delta(delta)),
        discretization_gap_func=epsilon_discretization_gap,
        tail_gap_func=epsilon_tail_gap,
        pld_builder=allocation_pld_api,
    )


def _run_real_delta_adaptive(
    *,
    sigma: float,
    num_steps: int,
    epsilon: float,
    delta_accuracy: float,
    num_selected: int = 1,
    num_epochs: int = 1,
) -> AdaptiveResult:
    params = PrivacyParams(
        sigma=sigma,
        num_steps=num_steps,
        num_selected=num_selected,
        num_epochs=num_epochs,
        epsilon=epsilon,
    )

    return adaptive_convergence(
        params=params,
        target_accuracy=delta_accuracy,
        initial_discretization=epsilon / 2,
        initial_tail_truncation=delta_accuracy,
        query_func=lambda pld: float(pld.get_delta_for_epsilon(epsilon)),
        discretization_gap_func=lambda pld, upper_value, lower_value, discretization: max(
            0.0,
            float(pld.get_delta_for_epsilon(epsilon)) - float(pld.get_delta_for_epsilon(epsilon + discretization)),
        ),
        tail_gap_func=lambda pld, upper_value, lower_value, tail_truncation: max(0.0, upper_value - lower_value),
        pld_builder=allocation_pld_api,
    )


class TestAdaptivePublicApi:
    def test_epsilon_range_returns_tuple(self):
        bounds = numerical_allocation_epsilon_range(
            sigma=2.0,
            num_steps=20,
            delta=1e-6,
            epsilon_accuracy=1e-2,
        )

        assert isinstance(bounds, tuple)
        assert len(bounds) == 2
        assert np.isfinite(bounds[0])
        assert np.isfinite(bounds[1])
        assert bounds[0] >= bounds[1]

    def test_delta_range_returns_tuple(self):
        bounds = numerical_allocation_delta_range(
            sigma=2.0,
            num_steps=20,
            epsilon=1.0,
            delta_accuracy=1e-6,
        )

        assert isinstance(bounds, tuple)
        assert len(bounds) == 2
        assert np.isfinite(bounds[0])
        assert np.isfinite(bounds[1])
        assert bounds[0] >= bounds[1]

    def test_epsilon_range_consistent_with_fixed_resolution(self):
        sigma = 1.5
        num_steps = 5
        delta = 1e-5
        epsilon_accuracy = 5e-2

        upper_bound, lower_bound = numerical_allocation_epsilon_range(
            sigma=sigma,
            num_steps=num_steps,
            delta=delta,
            epsilon_accuracy=epsilon_accuracy,
        )

        params = PrivacyParams(sigma=sigma, num_steps=num_steps, delta=delta)
        config_fine = AllocationSchemeConfig(
            loss_discretization=1e-2,
            tail_truncation=1e-8,
        )
        epsilon_fine = numerical_allocation_epsilon(params=params, config=config_fine)

        assert upper_bound >= epsilon_fine >= lower_bound
        assert upper_bound - lower_bound < epsilon_accuracy

    def test_delta_range_consistent_with_fixed_resolution(self):
        sigma = 1.5
        num_steps = 5
        epsilon = 1.0
        delta_accuracy = 1e-4

        upper_bound, lower_bound = numerical_allocation_delta_range(
            sigma=sigma,
            num_steps=num_steps,
            epsilon=epsilon,
            delta_accuracy=delta_accuracy,
        )

        params = PrivacyParams(sigma=sigma, num_steps=num_steps, epsilon=epsilon)
        config_fine = AllocationSchemeConfig(
            loss_discretization=1e-2,
            tail_truncation=1e-8,
        )
        delta_fine = numerical_allocation_delta(params=params, config=config_fine)

        assert upper_bound >= delta_fine >= lower_bound
        assert upper_bound - lower_bound < delta_accuracy

    @pytest.mark.parametrize(
        ("sigma", "num_steps", "loss_discretization", "tail_truncation"),
        [
            (2.0, 20, 0.5, 1e-6),
            (1.5, 5, 0.25, 1e-4),
        ],
    )
    def test_remove_geom_uses_shared_ratio_for_delta_regressions(
        self,
        sigma: float,
        num_steps: int,
        loss_discretization: float,
        tail_truncation: float,
    ):
        params = PrivacyParams(
            sigma=sigma,
            num_steps=num_steps,
            epsilon=1.0,
        )
        config = AllocationSchemeConfig(
            loss_discretization=loss_discretization,
            tail_truncation=tail_truncation,
            convolution_method=ConvolutionMethod.GEOM,
        )

        dist = allocation_PMF(
            conv_params=compute_conv_params(params=params, config=config),
            direction=Direction.REMOVE,
            bound_type=BoundType.DOMINATES,
            convolution_method=ConvolutionMethod.GEOM,
        )

        assert np.all(np.isfinite(dist.x_array))
        assert dist.PMF_array.size > 1

    def test_default_target_accuracy_uses_delta(self, monkeypatch):
        captured: dict[str, float] = {}

        monkeypatch.setattr(random_allocation_api_module, "estimate_poisson_query", lambda **_: 0.7)

        def fake_adaptive_convergence(
            *,
            params,
            target_accuracy,
            initial_discretization,
            initial_tail_truncation,
            query_func,
            discretization_gap_func,
            tail_gap_func,
            pld_builder,
        ):
            captured["target_accuracy"] = target_accuracy
            captured["initial_discretization"] = initial_discretization
            captured["initial_tail_truncation"] = initial_tail_truncation
            return _stub_result(target_accuracy=target_accuracy, value=0.7)

        monkeypatch.setattr(random_allocation_api_module, "adaptive_convergence", fake_adaptive_convergence)

        value = numerical_allocation_epsilon_range(
            sigma=5.0,
            num_steps=10,
            delta=1e-6,
        )

        assert np.isclose(captured["target_accuracy"], 7e-2)
        assert np.isclose(captured["initial_discretization"], 0.28)
        assert np.isclose(captured["initial_tail_truncation"], 1e-7)
        assert np.allclose(value, (0.7, 0.665))

    def test_default_target_accuracy_uses_epsilon(self, monkeypatch):
        captured: dict[str, float] = {}

        monkeypatch.setattr(random_allocation_api_module, "estimate_poisson_query", lambda **_: 0.4)

        def fake_adaptive_convergence(
            *,
            params,
            target_accuracy,
            initial_discretization,
            initial_tail_truncation,
            query_func,
            discretization_gap_func,
            tail_gap_func,
            pld_builder,
        ):
            captured["target_accuracy"] = target_accuracy
            captured["initial_discretization"] = initial_discretization
            captured["initial_tail_truncation"] = initial_tail_truncation
            return _stub_result(target_accuracy=target_accuracy, value=0.4)

        monkeypatch.setattr(random_allocation_api_module, "adaptive_convergence", fake_adaptive_convergence)

        value = numerical_allocation_delta_range(
            sigma=5.0,
            num_steps=10,
            epsilon=1.0,
        )

        assert np.isclose(captured["target_accuracy"], 4e-2)
        assert np.isclose(captured["initial_discretization"], 0.5)
        assert np.isclose(captured["initial_tail_truncation"], 0.04)
        assert value == (0.4, 0.38)

    def test_delta_range_default_accuracy_handles_notebook_multi_allocation_case(self):
        upper_bound, lower_bound = numerical_allocation_delta_range(
            sigma=3.0,
            num_steps=100,
            num_selected=10,
            epsilon=2.0,
        )

        assert upper_bound >= lower_bound >= 0.0
        assert upper_bound - lower_bound < 1e-8
        assert upper_bound < 1e-6

    def test_num_selected_and_epochs_are_forwarded(self, monkeypatch):
        captured: dict[str, object] = {}

        monkeypatch.setattr(random_allocation_api_module, "estimate_poisson_query", lambda **_: 0.5)

        def fake_adaptive_convergence(
            *,
            params,
            target_accuracy,
            initial_discretization,
            initial_tail_truncation,
            query_func,
            discretization_gap_func,
            tail_gap_func,
            pld_builder,
        ):
            captured["params"] = params
            return _stub_result(target_accuracy=target_accuracy)

        monkeypatch.setattr(random_allocation_api_module, "adaptive_convergence", fake_adaptive_convergence)

        numerical_allocation_epsilon_range(
            sigma=2.0,
            num_steps=20,
            delta=1e-6,
            num_selected=5,
            num_epochs=2,
            epsilon_accuracy=1e-2,
        )

        assert captured["params"].num_selected == 5
        assert captured["params"].num_epochs == 2

    def test_api_module_exports_main_functions(self):
        assert numerical_allocation_epsilon_range_api is numerical_allocation_epsilon_range
        assert numerical_allocation_delta_range_api is numerical_allocation_delta_range
        assert numerical_allocation_epsilon_api is numerical_allocation_epsilon
        assert numerical_allocation_delta_api is numerical_allocation_delta
        assert callable(allocation_pld_api)


class TestAdaptiveIntegrationConvergence:
    @pytest.mark.parametrize(
        ("sigma", "num_steps", "num_selected", "num_epochs", "delta", "epsilon_accuracy"),
        [
            (1.2, 12, 1, 1, 1e-5, 2e-2),
            (2.0, 20, 1, 1, 1e-6, 1e-2),
        ],
    )
    def test_epsilon_range_converges_within_requested_gap_and_brackets_fine_value(
        self,
        sigma: float,
        num_steps: int,
        num_selected: int,
        num_epochs: int,
        delta: float,
        epsilon_accuracy: float,
    ):
        result = _run_real_epsilon_adaptive(
            sigma=sigma,
            num_steps=num_steps,
            num_selected=num_selected,
            num_epochs=num_epochs,
            delta=delta,
            epsilon_accuracy=epsilon_accuracy,
        )

        params = PrivacyParams(
            sigma=sigma,
            num_steps=num_steps,
            num_selected=num_selected,
            num_epochs=num_epochs,
            delta=delta,
        )
        epsilon_fine = numerical_allocation_epsilon(
            params=params,
            config=AllocationSchemeConfig(
                loss_discretization=1e-2,
                tail_truncation=1e-8,
            ),
        )

        assert result.converged
        assert result.absolute_gap < result.target_accuracy
        assert result.upper_bound >= epsilon_fine >= result.lower_bound
        assert result.iterations >= 1

    @pytest.mark.parametrize(
        ("sigma", "num_steps", "num_selected", "num_epochs", "epsilon", "delta_accuracy"),
        [
            (1.5, 5, 1, 1, 1.0, 1e-4),
            (2.0, 20, 1, 1, 1.0, 1e-6),
        ],
    )
    def test_delta_range_converges_within_requested_gap_and_brackets_fine_value(
        self,
        sigma: float,
        num_steps: int,
        num_selected: int,
        num_epochs: int,
        epsilon: float,
        delta_accuracy: float,
    ):
        result = _run_real_delta_adaptive(
            sigma=sigma,
            num_steps=num_steps,
            num_selected=num_selected,
            num_epochs=num_epochs,
            epsilon=epsilon,
            delta_accuracy=delta_accuracy,
        )

        params = PrivacyParams(
            sigma=sigma,
            num_steps=num_steps,
            num_selected=num_selected,
            num_epochs=num_epochs,
            epsilon=epsilon,
        )
        delta_fine = numerical_allocation_delta(
            params=params,
            config=AllocationSchemeConfig(
                loss_discretization=1e-2,
                tail_truncation=1e-8,
            ),
        )

        assert result.converged
        assert result.absolute_gap < result.target_accuracy
        assert result.upper_bound >= delta_fine >= result.lower_bound
        assert result.iterations >= 1

    def test_public_epsilon_range_matches_direct_adaptive_result(self):
        direct = _run_real_epsilon_adaptive(
            sigma=1.5,
            num_steps=10,
            num_selected=1,
            num_epochs=1,
            delta=1e-5,
            epsilon_accuracy=2e-2,
        )
        public = numerical_allocation_epsilon_range(
            sigma=1.5,
            num_steps=10,
            num_selected=1,
            num_epochs=1,
            delta=1e-5,
            epsilon_accuracy=2e-2,
        )

        assert direct.converged
        assert np.allclose(public, (direct.upper_bound, direct.lower_bound))

    def test_public_delta_range_matches_direct_adaptive_result(self):
        direct = _run_real_delta_adaptive(
            sigma=1.7,
            num_steps=10,
            num_selected=1,
            num_epochs=1,
            epsilon=0.9,
            delta_accuracy=5e-5,
        )
        public = numerical_allocation_delta_range(
            sigma=1.7,
            num_steps=10,
            num_selected=1,
            num_epochs=1,
            epsilon=0.9,
            delta_accuracy=5e-5,
        )

        assert direct.converged
        assert np.allclose(public, (direct.upper_bound, direct.lower_bound))


class TestAdaptiveConvergence:
    def test_returns_bounds_and_metadata(self):
        params = PrivacyParams(sigma=2.0, num_steps=20, delta=1e-6)

        class FakePLD:
            def __init__(self, value: float):
                self.value = value

        def fake_builder(*, params, config, direction, bound_type):
            if bound_type == BoundType.DOMINATES:
                return FakePLD(0.55)
            return FakePLD(0.545)

        result = adaptive_convergence(
            params=params,
            target_accuracy=1e-2,
            initial_discretization=0.1,
            initial_tail_truncation=1e-6,
            query_func=lambda pld: pld.value,
            discretization_gap_func=lambda pld, upper, lower, disc: upper - lower,
            tail_gap_func=lambda pld, upper, lower, tail: 0.0,
            pld_builder=fake_builder,
        )

        assert isinstance(result, AdaptiveResult)
        assert np.isclose(result.upper_bound, 0.55)
        assert np.isclose(result.lower_bound, 0.545)
        assert result.converged
        assert result.absolute_gap < result.target_accuracy

    def test_propagates_builder_errors(self):
        params = PrivacyParams(sigma=2.0, num_steps=20, delta=1e-6)

        def fake_builder(*, params, config, direction, bound_type):
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            adaptive_convergence(
                params=params,
                target_accuracy=1e-2,
                initial_discretization=0.1,
                initial_tail_truncation=1e-6,
                query_func=lambda pld: pld,
                discretization_gap_func=lambda pld, upper, lower, disc: 0.0,
                tail_gap_func=lambda pld, upper, lower, tail: 0.0,
                pld_builder=fake_builder,
            )

    def test_tracks_best_bounds_across_non_monotonic_refinement(self):
        params = PrivacyParams(sigma=2.0, num_steps=20, delta=1e-6)
        call_count = {"dominates": 0, "dominated": 0}

        class FakePLD:
            def __init__(self, value: float):
                self.value = value

        def fake_builder(*, params, config, direction, bound_type):
            if bound_type == BoundType.DOMINATES:
                call_count["dominates"] += 1
                if call_count["dominates"] == 1:
                    return FakePLD(0.55)
                return FakePLD(0.56)

            call_count["dominated"] += 1
            if call_count["dominated"] == 1:
                return FakePLD(0.54)
            return FakePLD(0.541)

        result = adaptive_convergence(
            params=params,
            target_accuracy=1e-2,
            initial_discretization=0.1,
            initial_tail_truncation=1e-6,
            query_func=lambda pld: pld.value,
            discretization_gap_func=lambda pld, upper, lower, disc: upper - lower,
            tail_gap_func=lambda pld, upper, lower, tail: 0.0,
            pld_builder=fake_builder,
        )

        assert np.isclose(result.upper_bound, 0.55)
        assert np.isclose(result.lower_bound, 0.541)
        assert result.converged

    def test_raises_on_invalid_bounds(self):
        params = PrivacyParams(sigma=2.0, num_steps=20, delta=1e-6)

        class FakePLD:
            def __init__(self, value: float):
                self.value = value

        def fake_builder(*, params, config, direction, bound_type):
            if bound_type == BoundType.DOMINATES:
                return FakePLD(0.53)
            return FakePLD(0.54)

        with pytest.raises(RuntimeError, match="invalid bounds"):
            adaptive_convergence(
                params=params,
                target_accuracy=1e-6,
                initial_discretization=0.1,
                initial_tail_truncation=1e-6,
                query_func=lambda pld: pld.value,
                discretization_gap_func=lambda pld, upper, lower, disc: upper - lower,
                tail_gap_func=lambda pld, upper, lower, tail: 0.0,
                pld_builder=fake_builder,
            )

    def test_warns_when_not_converged(self):
        params = PrivacyParams(sigma=2.0, num_steps=20, delta=1e-6)

        class FakePLD:
            def __init__(self, value: float):
                self.value = value

        def fake_builder(*, params, config, direction, bound_type):
            if bound_type == BoundType.DOMINATES:
                return FakePLD(1.0)
            return FakePLD(0.0)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = adaptive_convergence(
                params=params,
                target_accuracy=1e-12,
                initial_discretization=0.1,
                initial_tail_truncation=1e-6,
                query_func=lambda pld: pld.value,
                discretization_gap_func=lambda pld, upper, lower, disc: 0.0,
                tail_gap_func=lambda pld, upper, lower, tail: 0.0,
                pld_builder=fake_builder,
            )

        assert not result.converged
        assert any("did not converge" in str(w.message) for w in caught)

    def test_refines_discretization_when_discretization_gap_exceeds_tail_gap(self):
        params = PrivacyParams(sigma=2.0, num_steps=20, delta=1e-6)
        configs: list[AllocationSchemeConfig] = []

        class FakePLD:
            def __init__(self, value: float):
                self.value = value

        def fake_builder(*, params, config, direction, bound_type):
            configs.append(config)
            if len(configs) <= 2:
                return FakePLD(1.0 if bound_type == BoundType.DOMINATES else 0.0)
            return FakePLD(0.6 if bound_type == BoundType.DOMINATES else 0.595)

        result = adaptive_convergence(
            params=params,
            target_accuracy=1e-2,
            initial_discretization=0.2,
            initial_tail_truncation=1e-3,
            query_func=lambda pld: pld.value,
            discretization_gap_func=lambda pld, upper, lower, disc: 5e-2,
            tail_gap_func=lambda pld, upper, lower, tail: 1e-2,
            pld_builder=fake_builder,
        )

        assert result.iterations == 2
        assert np.isclose(configs[0].loss_discretization, 0.1)
        assert np.isclose(configs[0].tail_truncation, 1e-4)
        assert np.isclose(configs[2].loss_discretization, 0.05)
        assert np.isclose(configs[2].tail_truncation, 1e-4)

    def test_refines_tail_when_tail_gap_dominates_discretization_gap(self):
        params = PrivacyParams(sigma=2.0, num_steps=20, delta=1e-6)
        configs: list[AllocationSchemeConfig] = []

        class FakePLD:
            def __init__(self, value: float):
                self.value = value

        def fake_builder(*, params, config, direction, bound_type):
            configs.append(config)
            if len(configs) <= 2:
                return FakePLD(0.6 if bound_type == BoundType.DOMINATES else 0.54)
            return FakePLD(0.56 if bound_type == BoundType.DOMINATES else 0.555)

        result = adaptive_convergence(
            params=params,
            target_accuracy=1e-2,
            initial_discretization=0.2,
            initial_tail_truncation=1e-3,
            query_func=lambda pld: pld.value,
            discretization_gap_func=lambda pld, upper, lower, disc: 1e-2,
            tail_gap_func=lambda pld, upper, lower, tail: 1e-1,
            pld_builder=fake_builder,
        )

        assert result.iterations == 2
        assert np.isclose(configs[0].loss_discretization, 0.1)
        assert np.isclose(configs[2].tail_truncation, 1e-5)
        assert np.isclose(configs[0].tail_truncation, 1e-4)
        assert np.isclose(configs[2].loss_discretization, 0.1)

    def test_reports_effective_initial_values_after_clamping(self):
        params = PrivacyParams(sigma=2.0, num_steps=20, delta=1e-6)

        class FakePLD:
            def __init__(self, value: float):
                self.value = value

        def fake_builder(*, params, config, direction, bound_type):
            if bound_type == BoundType.DOMINATES:
                return FakePLD(0.55)
            return FakePLD(0.545)

        result = adaptive_convergence(
            params=params,
            target_accuracy=1e-2,
            initial_discretization=0.2,
            initial_tail_truncation=1e-3,
            query_func=lambda pld: pld.value,
            discretization_gap_func=lambda pld, upper, lower, disc: upper - lower,
            tail_gap_func=lambda pld, upper, lower, tail: 0.0,
            pld_builder=fake_builder,
        )

        assert np.isclose(result.initial_discretization, 0.1)
        assert np.isclose(result.initial_tail_truncation, 1e-4)

    def test_forces_change_after_three_stagnant_iterations(self):
        params = PrivacyParams(sigma=2.0, num_steps=20, delta=1e-6)
        configs: list[AllocationSchemeConfig] = []

        class FakePLD:
            def __init__(self, value: float):
                self.value = value

        def fake_builder(*, params, config, direction, bound_type):
            configs.append(config)
            return FakePLD(1.0 if bound_type == BoundType.DOMINATES else 0.0)

        result = adaptive_convergence(
            params=params,
            target_accuracy=1e-12,
            initial_discretization=0.1,
            initial_tail_truncation=1e-16,
            query_func=lambda pld: pld.value,
            discretization_gap_func=lambda pld, upper, lower, disc: 0.2,
            tail_gap_func=lambda pld, upper, lower, tail: 1.0,
            pld_builder=fake_builder,
        )

        assert not result.converged
        assert np.isclose(configs[0].loss_discretization, 0.1)
        assert np.isclose(configs[2].loss_discretization, 0.1)
        assert np.isclose(configs[4].loss_discretization, 0.1)
        assert np.isclose(configs[6].loss_discretization, 0.05)

    def test_raises_on_invalid_discretization_gap_estimate(self):
        params = PrivacyParams(sigma=2.0, num_steps=20, delta=1e-6)

        class FakePLD:
            def __init__(self, value: float):
                self.value = value

        def fake_builder(*, params, config, direction, bound_type):
            return FakePLD(0.5 if bound_type == BoundType.DOMINATES else 0.4)

        with pytest.raises(RuntimeError, match="invalid discretization-gap estimate"):
            adaptive_convergence(
                params=params,
                target_accuracy=1e-6,
                initial_discretization=0.1,
                initial_tail_truncation=1e-6,
                query_func=lambda pld: pld.value,
                discretization_gap_func=lambda pld, upper, lower, disc: -1.0,
                tail_gap_func=lambda pld, upper, lower, tail: 0.0,
                pld_builder=fake_builder,
            )

    def test_raises_on_invalid_tail_gap_estimate(self):
        params = PrivacyParams(sigma=2.0, num_steps=20, delta=1e-6)

        class FakePLD:
            def __init__(self, value: float):
                self.value = value

        def fake_builder(*, params, config, direction, bound_type):
            return FakePLD(0.5 if bound_type == BoundType.DOMINATES else 0.4)

        with pytest.raises(RuntimeError, match="invalid tail-gap estimate"):
            adaptive_convergence(
                params=params,
                target_accuracy=1e-6,
                initial_discretization=0.1,
                initial_tail_truncation=1e-6,
                query_func=lambda pld: pld.value,
                discretization_gap_func=lambda pld, upper, lower, disc: 0.1,
                tail_gap_func=lambda pld, upper, lower, tail: -1.0,
                pld_builder=fake_builder,
            )


class TestPoissonAdaptiveEstimate:
    def test_poisson_estimate_uses_requested_sampling_probability_and_composition(self, monkeypatch):
        captured: dict[str, float | int] = {}

        class FakePLD:
            def self_compose(self, steps: int):
                captured["compose_steps"] = steps
                return self

        def fake_from_gaussian_mechanism(**kwargs):
            captured["sampling_prob"] = kwargs["sampling_prob"]
            captured["standard_deviation"] = kwargs["standard_deviation"]
            return FakePLD()

        monkeypatch.setattr(
            "PLD_accounting.adaptive_random_allocation.privacy_loss_distribution.from_gaussian_mechanism",
            fake_from_gaussian_mechanism,
        )

        params = PrivacyParams(
            sigma=3.0,
            num_steps=200,
            num_selected=10,
            num_epochs=4,
            delta=1e-6,
        )

        estimate = estimate_poisson_query(
            params=params,
            query_func=lambda pld: 0.25,
        )

        assert np.isclose(estimate, 0.25)
        assert np.isclose(captured["standard_deviation"], 3.0)
        assert np.isclose(captured["sampling_prob"], 10 / 200)
        assert captured["compose_steps"] == 40
