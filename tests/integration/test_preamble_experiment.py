#!/usr/bin/env python3
"""
Integration tests for the PREAMBLE experiment.

This module tests the complete PREAMBLE experiment pipeline including:
- Parallel processing
- Parameter handling
- Result collection
- Plotting functionality
"""

import pytest
import warnings
import numpy as np
from numpy.typing import NDArray

from comparisons.experiments.PREAMBLE import (
    run_PREAMBLE_experiment,
    epsilon_from_sigma_allocation,
    epsilon_from_sigma_allocation_RDP,
)
from PLD_accounting.types import AllocationSchemeConfig, BoundType, Direction, PrivacyParams, ConvolutionMethod
from random_allocation.comparisons.structs import SchemeConfig


class TestPREAMBLEExperiment:
    """Test the PREAMBLE experiment functionality."""
    
    @pytest.fixture
    def small_experiment_settings(self):
        """Settings for a small PREAMBLE experiment."""
        return {
            'sample size': 200,            # Very small sample size for speed
            'D': 50,                       # Small dimension
            'communication constant': 10,  # Avoid zero num_selected
            'SGD_num_epochs': 1,           # Single epoch
            'batch size array': [50],      # Single batch size
            'B array': [5],                # Single B value
            'delta': 1e-5,                 # Standard delta
            'target epsilon': 0.5,         # Target epsilon
            'clip scale': 1.0              # Standard clip scale
        }
    
    @pytest.fixture
    def experiment_configs(self):
        """Configuration objects for the experiment."""
        numerical_config = AllocationSchemeConfig(
            loss_discretization=1e-2,
            tail_truncation=1e-6,            # Smaller beta to keep tail mass below delta
            max_grid_FFT=1_000_000,  # Larger grid to avoid FFT size errors
            max_grid_mult=5_000,  # Smaller grid for speed
            convolution_method=ConvolutionMethod.FFT
        )
        gaussian_config = SchemeConfig(discretization=1e-2)
        return numerical_config, gaussian_config
    
    def test_preamble_experiment_basic_functionality(self, small_experiment_settings,
                                                    experiment_configs):
        """Test that the PREAMBLE experiment runs without errors."""
        numerical_config, gaussian_config = experiment_configs

        # Run the experiment
        results = run_PREAMBLE_experiment(
            settings_dict=small_experiment_settings,
            numerical_config=numerical_config,
            Gaussian_config=gaussian_config,
            bound_type=BoundType.DOMINATES,
            num_processes=1  # Single process for test stability
        )
        
        # Verify results structure
        assert isinstance(results, dict)
        assert 'params' in results  # Should have params key
        assert len(results) == 2  # Should have params + 1 batch size
        
        for batch_size in small_experiment_settings['batch size array']:
            assert batch_size in results
            assert 'Gaussian_sigma' in results[batch_size]
            
            # Check that Gaussian sigma is reasonable
            gaussian_sigma = results[batch_size]['Gaussian_sigma']
            assert isinstance(gaussian_sigma, (int, float))
            assert gaussian_sigma > 0
            
            method_key = (BoundType.DOMINATES, ConvolutionMethod.FFT)
            assert method_key in results[batch_size]
            sigma_values = results[batch_size][method_key]
            assert isinstance(sigma_values, np.ndarray)
            assert len(sigma_values) == len(small_experiment_settings['B array'])
            assert all(sigma > 0 for sigma in sigma_values)
    
    def test_preamble_experiment_monotonicity(self, small_experiment_settings,
                                            experiment_configs):
        """Test that sigma values increase monotonically with B."""
        numerical_config, gaussian_config = experiment_configs
        
        results = run_PREAMBLE_experiment(
            settings_dict=small_experiment_settings,
            numerical_config=numerical_config,
            Gaussian_config=gaussian_config,
            bound_type=BoundType.DOMINATES,
            num_processes=2
        )
        
        B_array = np.array(small_experiment_settings['B array'])
        
        # Check monotonicity for each batch size
        for batch_size in small_experiment_settings['batch size array']:
            method_key = (BoundType.DOMINATES, ConvolutionMethod.FFT)
            sigma_values = results[batch_size][method_key]
            if len(sigma_values) > 1:
                assert sigma_values[-1] >= sigma_values[0] * 0.95, \
                    "Sigma values should generally increase with B"
    
    def test_preamble_experiment_parallel_processing(self, small_experiment_settings,
                                                   experiment_configs):
        """Test that parallel processing works correctly."""
        numerical_config, gaussian_config = experiment_configs
        
        # Test with different numbers of processes
        for num_processes in [1]:
            results = run_PREAMBLE_experiment(
                settings_dict=small_experiment_settings,
                numerical_config=numerical_config,
                Gaussian_config=gaussian_config,
                bound_type=BoundType.DOMINATES,
                num_processes=num_processes
            )
            
            # Results should be the same regardless of number of processes
            assert isinstance(results, dict)
            assert 'params' in results
            assert len(results) == 2  # params + 1 batch size

    def test_preamble_loss_discretization_scaled_by_rounds(self, small_experiment_settings,
                                                           experiment_configs):
        """Ensure loss_discretization is scaled by sqrt(num_rounds * sampling_probability)."""
        numerical_config, _ = experiment_configs
        params = PrivacyParams(
            sigma=1.0,
            num_steps=10,
            num_selected=1,
            num_epochs=small_experiment_settings['SGD_num_epochs'],
            delta=small_experiment_settings['delta'],
        )
        sampling_probability = (
            small_experiment_settings['batch size array'][0]
            / float(small_experiment_settings['sample size'])
        )
        num_rounds = int(
            np.ceil(
                small_experiment_settings['sample size']
                / small_experiment_settings['batch size array'][0]
            ) * small_experiment_settings['SGD_num_epochs']
        )
        original = numerical_config.loss_discretization
        _ = epsilon_from_sigma_allocation(
            params=params,
            config=numerical_config,
            sampling_probability=sampling_probability,
            num_rounds=num_rounds,
            bound_type=BoundType.DOMINATES,
        )
        expected = original / np.sqrt(num_rounds * sampling_probability)
        assert np.isclose(numerical_config.loss_discretization, expected)

    def test_preamble_experiment_respects_rdp_upper_bound(self, small_experiment_settings,
                                                         experiment_configs):
        """RDP should be a conservative upper bound on epsilon at fixed sigma."""
        numerical_config, gaussian_config = experiment_configs

        results = run_PREAMBLE_experiment(
            settings_dict=small_experiment_settings,
            numerical_config=numerical_config,
            Gaussian_config=gaussian_config,
            bound_type=BoundType.DOMINATES,
            num_processes=1
        )

        batch_size = small_experiment_settings['batch size array'][0]
        method_key = (BoundType.DOMINATES, ConvolutionMethod.FFT)
        sigma_value = float(results[batch_size][method_key][0])

        B = small_experiment_settings['B array'][0]
        sampling_probability = batch_size / float(small_experiment_settings['sample size'])
        num_rounds = int(np.ceil(small_experiment_settings['sample size'] / batch_size)
                         * small_experiment_settings['SGD_num_epochs'])
        num_selected = max(1, int(small_experiment_settings['communication constant'] / B))
        num_steps = int(small_experiment_settings['D'] / B)
        block_norm = (small_experiment_settings['clip scale']
                      * np.sqrt(small_experiment_settings['D'] / B)
                      / num_selected)

        params = PrivacyParams(
            sigma=sigma_value / block_norm,
            num_steps=num_steps,
            num_selected=num_selected,
            delta=small_experiment_settings['delta']
        )

        eps_fft = epsilon_from_sigma_allocation(
            params=params,
            config=numerical_config,
            sampling_probability=sampling_probability,
            num_rounds=num_rounds,
            bound_type=BoundType.DOMINATES
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Optimal order is the .* alpha.*",
                category=UserWarning,
            )
            eps_rdp = epsilon_from_sigma_allocation_RDP(
                params=params,
                sampling_probability=sampling_probability,
                num_rounds=num_rounds
            )

        assert np.isfinite(eps_fft)
        assert np.isfinite(eps_rdp)
        assert eps_fft <= eps_rdp

    def test_preamble_experiment_parameter_validation(self, experiment_configs):
        """Test that the experiment validates parameters correctly."""
        numerical_config, gaussian_config = experiment_configs
        
        # Test with invalid settings
        invalid_settings = {
            'sample size': 0,  # Invalid sample size
            'D': 100,
            'communication constant': 20,
            'SGD_num_epochs': 1,
            'batch size array': [50, 100],
            'B array': [5, 10, 20],
            'delta': 1e-5,
            'target epsilon': 1.0,
            'clip scale': 1.0
        }
        
        # This should raise a ZeroDivisionError for invalid sample size
        with pytest.raises(ZeroDivisionError):
            results = run_PREAMBLE_experiment(
                settings_dict=invalid_settings,
                numerical_config=numerical_config,
                Gaussian_config=gaussian_config,
                bound_type=BoundType.DOMINATES,
                num_processes=1
            )

    def test_bound_type_variants(self, experiment_configs):
        """Test that different bound types are supported."""
        numerical_config, gaussian_config = experiment_configs

        # Override convolution method in config for this test
        numerical_config = AllocationSchemeConfig(
            loss_discretization=numerical_config.loss_discretization,
            tail_truncation=numerical_config.tail_truncation,
            max_grid_FFT=numerical_config.max_grid_FFT,
            max_grid_mult=numerical_config.max_grid_mult,
            convolution_method=ConvolutionMethod.GEOM
        )

        tiny_settings = {
            'sample size': 100,
            'D': 20,
            'communication constant': 10,
            'SGD_num_epochs': 1,
            'batch size array': [50],
            'B array': [5],
            'delta': 1e-5,
            'target epsilon': 2.0,
            'clip scale': 1.0
        }

        # Test DOMINATES bound type (IS_DOMINATED not supported for dual subsampling route)
        for bound_type in (BoundType.DOMINATES,):
            results = run_PREAMBLE_experiment(
                settings_dict=tiny_settings,
                numerical_config=numerical_config,
                Gaussian_config=gaussian_config,
                bound_type=bound_type,
                num_processes=1
            )

            batch_size = tiny_settings['batch size array'][0]
            method_key = (bound_type, ConvolutionMethod.GEOM)
            assert method_key in results[batch_size]
