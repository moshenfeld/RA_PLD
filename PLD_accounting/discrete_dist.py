from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from PLD_accounting.core_utils import PMF_MASS_TOL, enforce_mass_conservation
from PLD_accounting.types import BoundType


class DiscreteDist:
    def __init__(
        self,
        x_array: NDArray[np.float64],
        PMF_array: NDArray[np.float64],
        p_neg_inf: float = 0.0,
        p_pos_inf: float = 0.0
    ) -> None:
        self.x_array = np.asarray(x_array, dtype=np.float64)
        self.PMF_array = np.asarray(PMF_array, dtype=np.float64)
        self.p_neg_inf = float(p_neg_inf)
        self.p_pos_inf = float(p_pos_inf)
        self._validate_basic()

    def _validate_basic(self) -> None:
        if self.x_array.ndim != 1 or self.PMF_array.ndim != 1 or self.x_array.shape != self.PMF_array.shape:
            raise ValueError("x and PMF must be 1-D arrays of equal length")
        if self.x_array.size < 2:
            raise ValueError("x and PMF must contain at least 2 points")
        if not np.all(np.diff(self.x_array) > 0):
            raise ValueError("x must be strictly increasing")
        if np.any(self.PMF_array < -PMF_MASS_TOL):
            raise ValueError("PMF must be nonnegative")
        if self.p_neg_inf < 0:
            raise ValueError("p_neg_inf must be nonnegative")
        if self.p_pos_inf < 0:
            raise ValueError("p_pos_inf must be nonnegative")

    def validate_mass_conservation(self, bound_type: BoundType) -> DiscreteDist:
        self._validate_basic()

        pmf_sum = math.fsum(map(float, self.PMF_array))
        total_mass = pmf_sum + self.p_neg_inf + self.p_pos_inf
        mass_error = abs(total_mass - 1.0)

        if mass_error > PMF_MASS_TOL:
            error_msg = "MASS CONSERVATION ERROR"
            error_msg += f": Error={mass_error:.2e} (tolerance={PMF_MASS_TOL:.2e})"
            error_msg += f", PMF sum={pmf_sum:.15f}"
            error_msg += f", p_neg_inf={self.p_neg_inf:.2e}"
            error_msg += f", p_pos_inf={self.p_pos_inf:.2e}"
            error_msg += f", Total mass={total_mass:.15f}"
            raise ValueError(error_msg)

        if pmf_sum <= PMF_MASS_TOL and (self.p_neg_inf + self.p_pos_inf) >= 1.0 - PMF_MASS_TOL:
            raise ValueError("Distributions with all mass at infinity are not supported")

        if bound_type == BoundType.DOMINATES and self.p_neg_inf > 0:
            raise ValueError("DOMINATES bound_type requires p_neg_inf=0")
        if bound_type == BoundType.IS_DOMINATED and self.p_pos_inf > 0:
            raise ValueError("IS_DOMINATED bound_type requires p_pos_inf=0")
        return self

    def truncate_edges(self, tail_truncation: float, bound_type: BoundType) -> DiscreteDist:
        if tail_truncation == 0.0:
            nonzero_indices = np.nonzero(self.PMF_array)[0]
            if nonzero_indices.size == 0:
                raise ValueError("Cannot truncate distribution with zero finite mass")
            min_ind = nonzero_indices[0]
            max_ind = nonzero_indices[-1]
        else:
            cumsum_left = np.cumsum(self.PMF_array, dtype=np.float64)
            cumsum_right = np.cumsum(self.PMF_array[::-1], dtype=np.float64)
            min_ind = int(np.searchsorted(cumsum_left, tail_truncation, side="right"))
            right_cnt = int(np.searchsorted(cumsum_right, tail_truncation, side="right"))
            max_ind = self.PMF_array.size - 1 - right_cnt

        if min_ind == 0 and max_ind == len(self.PMF_array) - 1:
            return self
        if min_ind > max_ind:
            finite_mass = math.fsum(map(float, self.PMF_array))
            raise ValueError(
                f"Cannot truncate {tail_truncation:.2e} from each tail: "
                f"finite mass ({finite_mass:.2e}) < 2*tail_truncation ({2*tail_truncation:.2e})"
            )

        if max_ind - min_ind < 1:
            if min_ind > 0:
                min_ind -= 1
            elif max_ind < self.PMF_array.size - 1:
                max_ind += 1
            else:
                raise ValueError("Cannot truncate to fewer than 2 bins")

        left_mass = math.fsum(map(float, self.PMF_array[:min_ind]))
        right_mass = math.fsum(map(float, self.PMF_array[max_ind + 1:]))

        self.x_array = self.x_array[min_ind: max_ind + 1]
        self.PMF_array = self.PMF_array[min_ind: max_ind + 1]

        if bound_type == BoundType.DOMINATES:
            self.PMF_array[0] += left_mass
            self.p_pos_inf += right_mass
        elif bound_type == BoundType.IS_DOMINATED:
            self.p_neg_inf += left_mass
            self.PMF_array[-1] += right_mass
        else:
            raise ValueError(f"Unknown BoundType: {bound_type}")

        self.PMF_array, self.p_neg_inf, self.p_pos_inf = enforce_mass_conservation(
            PMF_array=self.PMF_array,
            expected_neg_inf=self.p_neg_inf,
            expected_pos_inf=self.p_pos_inf,
            bound_type=bound_type,
        )
        return self

    def copy(self) -> "DiscreteDist":
        return DiscreteDist(
            x_array=np.array(self.x_array, copy=True),
            PMF_array=np.array(self.PMF_array, copy=True),
            p_neg_inf=self.p_neg_inf,
            p_pos_inf=self.p_pos_inf
        )
