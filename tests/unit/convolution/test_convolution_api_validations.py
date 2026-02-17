import numpy as np
import pytest

from PLD_accounting.convolution_API import (
    convolve_discrete_distributions,
    self_convolve_discrete_distributions,
)
from PLD_accounting.discrete_dist import DiscreteDist
from PLD_accounting.types import BoundType, ConvolutionMethod


def _linear_dist() -> DiscreteDist:
    return DiscreteDist(
        x_array=np.array([0.0, 0.5, 1.0], dtype=np.float64),
        PMF_array=np.array([0.2, 0.5, 0.3], dtype=np.float64),
    )


def test_self_convolve_rejects_unknown_method():
    with pytest.raises(ValueError, match="Invalid convolution_method"):
        self_convolve_discrete_distributions(
            dist=_linear_dist(),
            T=2,
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
            convolution_method="fft",  # type: ignore[arg-type]
        )


def test_convolve_rejects_unknown_method():
    with pytest.raises(Exception, match="Invalid convolution_method"):
        convolve_discrete_distributions(
            dist_1=_linear_dist(),
            dist_2=_linear_dist(),
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
            convolution_method="fft",  # type: ignore[arg-type]
        )


def test_fft_requires_linear_spacing():
    geometric = DiscreteDist(
        x_array=np.array([1.0, 2.0, 4.0], dtype=np.float64),
        PMF_array=np.array([0.3, 0.4, 0.3], dtype=np.float64),
    )
    with pytest.raises(ValueError, match="non-uniform bin widths"):
        self_convolve_discrete_distributions(
            dist=geometric,
            T=2,
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
            convolution_method=ConvolutionMethod.FFT,
        )
