import numpy as np

from PLD_accounting.discrete_dist import (
    DenseGeometricDiscreteDist,
    LinearDiscreteDistBase,
    SparseGeometricDiscreteDist,
)
from PLD_accounting.random_allocation_accounting import (
    _allocation_PMF,
    _allocation_PMF_add,
    _allocation_PMF_add_fft,
    _allocation_PMF_add_geom,
    _allocation_PMF_remove,
    _allocation_PMF_remove_fft,
    _allocation_PMF_remove_geom,
    _compute_conv_params,
    _maybe_sparsify_geometric_dist,
)
from PLD_accounting.types import (
    AllocationSchemeConfig,
    BoundType,
    ConvolutionMethod,
    Direction,
    PrivacyParams,
)


def _make_conv_params():
    params = PrivacyParams(
        sigma=1.1,
        num_steps=6,
        num_selected=2,
        num_epochs=1,
        delta=1e-6,
    )
    config = AllocationSchemeConfig(
        loss_discretization=1e-3,
        tail_truncation=1e-7,
        max_grid_mult=512,
        max_grid_FFT=4096,
    )
    return _compute_conv_params(params=params, config=config)


def _assert_same_distribution(left, right, atol: float = 1e-12):
    assert type(left) is type(right)
    assert left.x_array.shape == right.x_array.shape
    assert np.allclose(left.x_array, right.x_array, atol=atol, rtol=0.0)
    assert np.allclose(left.PMF_array, right.PMF_array, atol=atol, rtol=0.0)
    assert np.isclose(left.p_neg_inf, right.p_neg_inf, atol=atol, rtol=0.0)
    assert np.isclose(left.p_pos_inf, right.p_pos_inf, atol=atol, rtol=0.0)


def test_maybe_sparsify_geometric_dist_converts_exact_zero_bins():
    dense = DenseGeometricDiscreteDist(
        x_min=1.0,
        ratio=2.0,
        PMF_array=np.array([0.2, 0.0, 0.3, 0.5], dtype=np.float64),
        p_neg_inf=0.0,
        p_pos_inf=0.0,
    )

    sparse = _maybe_sparsify_geometric_dist(dense, use_sparse=True)

    assert isinstance(sparse, SparseGeometricDiscreteDist)
    assert np.array_equal(sparse.indices, np.array([0, 2, 3], dtype=np.int64))
    assert np.allclose(sparse.PMF_array, np.array([0.2, 0.3, 0.5], dtype=np.float64))


def test_remove_backend_helpers_match_dispatcher():
    conv_params = _make_conv_params()

    fft_direct = _allocation_PMF_remove_fft(conv_params=conv_params, bound_type=BoundType.DOMINATES)
    fft_dispatch = _allocation_PMF_remove(
        conv_params=conv_params,
        bound_type=BoundType.DOMINATES,
        convolution_method=ConvolutionMethod.FFT,
    )
    _assert_same_distribution(fft_direct, fft_dispatch)

    geom_direct = _allocation_PMF_remove_geom(
        conv_params=conv_params,
        bound_type=BoundType.DOMINATES,
        use_sparse=True,
    )
    geom_dispatch = _allocation_PMF_remove(
        conv_params=conv_params,
        bound_type=BoundType.DOMINATES,
        convolution_method=ConvolutionMethod.GEOM,
        use_sparse=True,
    )
    _assert_same_distribution(geom_direct, geom_dispatch)


def test_add_backend_helpers_match_dispatcher():
    conv_params = _make_conv_params()

    fft_direct = _allocation_PMF_add_fft(conv_params=conv_params, bound_type=BoundType.DOMINATES)
    fft_dispatch = _allocation_PMF_add(
        conv_params=conv_params,
        bound_type=BoundType.DOMINATES,
        convolution_method=ConvolutionMethod.FFT,
    )
    _assert_same_distribution(fft_direct, fft_dispatch)

    geom_direct = _allocation_PMF_add_geom(
        conv_params=conv_params,
        bound_type=BoundType.DOMINATES,
        use_sparse=True,
    )
    geom_dispatch = _allocation_PMF_add(
        conv_params=conv_params,
        bound_type=BoundType.DOMINATES,
        convolution_method=ConvolutionMethod.GEOM,
        use_sparse=True,
    )
    _assert_same_distribution(geom_direct, geom_dispatch)


def test_geometric_sparse_toggle_preserves_final_allocation_distribution():
    conv_params = _make_conv_params()

    dense_remove = _allocation_PMF(
        conv_params=conv_params,
        direction=Direction.REMOVE,
        bound_type=BoundType.DOMINATES,
        convolution_method=ConvolutionMethod.GEOM,
        use_sparse=False,
    )
    sparse_remove = _allocation_PMF(
        conv_params=conv_params,
        direction=Direction.REMOVE,
        bound_type=BoundType.DOMINATES,
        convolution_method=ConvolutionMethod.GEOM,
        use_sparse=True,
    )
    _assert_same_distribution(dense_remove, sparse_remove, atol=1e-10)

    dense_add = _allocation_PMF(
        conv_params=conv_params,
        direction=Direction.ADD,
        bound_type=BoundType.DOMINATES,
        convolution_method=ConvolutionMethod.GEOM,
        use_sparse=False,
    )
    sparse_add = _allocation_PMF(
        conv_params=conv_params,
        direction=Direction.ADD,
        bound_type=BoundType.DOMINATES,
        convolution_method=ConvolutionMethod.GEOM,
        use_sparse=True,
    )
    _assert_same_distribution(dense_add, sparse_add, atol=1e-10)


def test_allocation_pmf_always_returns_linear_structured_distribution():
    conv_params = _make_conv_params()

    for direction in (Direction.REMOVE, Direction.ADD):
        for method in (
            ConvolutionMethod.FFT,
            ConvolutionMethod.GEOM,
            ConvolutionMethod.BEST_OF_TWO,
            ConvolutionMethod.COMBINED,
        ):
            dist = _allocation_PMF(
                conv_params=conv_params,
                direction=direction,
                bound_type=BoundType.DOMINATES,
                convolution_method=method,
                use_sparse=True,
            )
            assert isinstance(dist, LinearDiscreteDistBase)
