#!/usr/bin/env python3
"""
Cross-parametrization sensitivity analysis for profit-sharing counterfactuals.

This script analyzes results across different cost function parametrizations,
compressing the alpha (profit-sharing) range into comparable coefficients
using linear and quadratic regression models.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from scipy.stats import gaussian_kde
import warnings
import time
from pathlib import Path
warnings.filterwarnings('ignore', category=UserWarning)

# Font configuration to match counterfactuals.py - Computer Modern without LaTeX
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif", 
    "font.serif": ["CMU Serif", "Computer Modern", "DejaVu Serif", "Times"],
    "font.size": 20
})

class ParametrizationComparator:
    def __init__(self, results_base_dir='results'):
        self.results_base_dir = Path(results_base_dir)
        self.parametrizations = {}
        # Okabe-Ito color palette for cost functions (colorblind-accessible)
        self.cost_function_colors = {
            'linear': '#0072B2',      # blue
            'quadratic': '#D55E00',   # vermillion (red-orange)
            'exponential': '#009E73', # bluish green
            'power_law': '#CC79A7'    # reddish purple
        }
        
    def discover_parametrizations(self):
        """Discover all available parametrization directories"""
        parametrizations = {}
        
        if not self.results_base_dir.exists():
            print(f"Results directory {self.results_base_dir} does not exist")
            return parametrizations
            
        for param_dir in self.results_base_dir.iterdir():
            if param_dir.is_dir():
                # Only load robustness parametrizations (skip baseline exploration runs)
                if not param_dir.name.startswith('robustness_'):
                    continue

                # Look for merged results file
                merged_file = param_dir / 'counterfactual_results_final.pkl'
                if merged_file.exists():
                    # Extract cost function type from directory name
                    cost_type = self._extract_cost_type(param_dir.name)
                    parametrizations[param_dir.name] = {
                        'path': param_dir,
                        'merged_file': merged_file,
                        'cost_type': cost_type
                    }
                    print(f"Found parametrization: {param_dir.name} (cost: {cost_type})")
                    
        self.parametrizations = parametrizations
        return parametrizations
    
    def _extract_cost_type(self, dir_name):
        """Extract cost function type from directory name"""
        # First check for robustness_ prefix (e.g., robustness_exponential_baseline)
        if 'robustness_linear' in dir_name or dir_name.startswith('robustness_linear'):
            return 'linear'
        elif 'robustness_quadratic' in dir_name or dir_name.startswith('robustness_quadratic'):
            return 'quadratic'
        elif 'robustness_exponential' in dir_name or dir_name.startswith('robustness_exponential'):
            return 'exponential'
        elif 'robustness_power_law' in dir_name or dir_name.startswith('robustness_power_law'):
            return 'power_law'
        # Then check for baseline parametrization format (e.g., markets_100_..._cost_type_linear_...)
        elif 'cost_type_linear' in dir_name:
            return 'linear'
        elif 'cost_type_quadratic' in dir_name:
            return 'quadratic'
        elif 'cost_type_exponential' in dir_name:
            return 'exponential'
        elif 'cost_type_power_law' in dir_name:
            return 'power_law'
        else:
            # Unable to determine - return unknown instead of assuming
            return 'unknown'
    
    def load_parametrization_data(self, param_name):
        """Load data for a specific parametrization"""
        if param_name not in self.parametrizations:
            raise ValueError(f"Parametrization {param_name} not found")
            
        param_info = self.parametrizations[param_name]
        
        with open(param_info['merged_file'], 'rb') as f:
            data = pickle.load(f)
            
        # Extract shares and organize data
        shares = sorted(data.keys())
        
        # Create structured dataset
        param_data = {
            'shares': np.array(shares),
            'cost_type': param_info['cost_type'],
            'hyperparameters': data[shares[0]]['hyperparameters'] if shares else {},
            'metrics': {}
        }
        
        # Extract time series metrics for each share
        for metric_name in ['mean_members_avg', 'num_cong_avg', 'mergers_per_period_avg', 'exits_per_period_avg',
                           'market_share_quantiles_avg', 'gini_quantiles_avg', 'poly_estimates_avg',
                           'temporal_poly_estimates_avg', 'temporal_poly_estimates_std']:
            param_data['metrics'][metric_name] = {}
            for share in shares:
                if metric_name in data[share]:
                    param_data['metrics'][metric_name][share] = data[share][metric_name]

        # Add rank mobility metrics if available
        for metric_name in ['rank_ranges', 'rank_std']:
            if metric_name in data[shares[0]]:
                param_data['metrics'][metric_name] = {}
                for share in shares:
                    param_data['metrics'][metric_name][share] = data[share][metric_name]

        # Add panel polynomial estimates (stored as array of all realizations, not averaged)
        if 'panel_poly_estimates_all' in data[shares[0]]:
            param_data['metrics']['panel_poly_estimates_all'] = {}
            for share in shares:
                if 'panel_poly_estimates_all' in data[share]:
                    param_data['metrics']['panel_poly_estimates_all'][share] = data[share]['panel_poly_estimates_all']

        return param_data
    
    def load_all_parametrizations(self):
        """Load data for all discovered parametrizations"""
        if not self.parametrizations:
            self.discover_parametrizations()
            
        all_data = {}
        for param_name in self.parametrizations:
            try:
                all_data[param_name] = self.load_parametrization_data(param_name)
                print(f"Loaded data for {param_name}")
            except Exception as e:
                print(f"Error loading {param_name}: {e}")
                
        return all_data
    
    def fit_alpha_sensitivity_models(self, param_data, metric_name='gini_quantiles_avg'):
        """
        Fit temporal alpha sensitivity models: for each timestep, regress metric ~ alpha
        
        Returns:
        - per_timestep_results: dict with timestep-specific coefficients showing impact of alpha over time
        - summary_stats: dict with time-averaged statistics
        """
        shares = param_data['shares']  # Array of alpha values [0.00, 0.02, ..., 0.50]
        metric_data = param_data['metrics'][metric_name]
        
        if not metric_data:
            return None, None
            
        # Organize data: shares x timesteps x quantiles matrix
        share_data = {}
        valid_shares = []
        
        for share in shares:
            if share in metric_data and metric_data[share] is not None:
                share_data[share] = metric_data[share]
                valid_shares.append(share)
        
        if not share_data:
            return None, None
            
        valid_shares = np.array(valid_shares)
        
        # Get dimensions from first valid share
        first_share_data = share_data[valid_shares[0]]
        
        # Handle different metric structures and fix transposed data
        if metric_name in ['gini_quantiles_avg', 'market_share_quantiles_avg']:
            # Check if data needs transposing (market_share_quantiles_avg is often transposed)
            if metric_name == 'market_share_quantiles_avg' and first_share_data.shape[0] < first_share_data.shape[1]:
                # Data is transposed: (n_quantiles, n_timesteps) -> (n_timesteps, n_quantiles)
                for share in valid_shares:
                    share_data[share] = share_data[share].T
                first_share_data = share_data[valid_shares[0]]
            
            # Shape: (n_timesteps, n_quantiles)
            n_timesteps, n_quantiles = first_share_data.shape
            metric_is_quantiled = True
        else:
            # Shape: (n_timesteps,) for scalar metrics like mean_members_avg
            n_timesteps = len(first_share_data)
            n_quantiles = 1
            metric_is_quantiled = False
        
        # For quantile metrics, analyze each quantile separately
        if metric_is_quantiled:
            quantile_names = ['10th', '25th', '50th', '75th', '90th', '99th', '100th']  # Standard quantiles
            results_per_quantile = {}
            
            for q_idx in range(min(n_quantiles, len(quantile_names))):
                quantile_name = quantile_names[q_idx]
                # Get temporal analysis for this quantile
                temporal_results = self._analyze_quantile_over_time(
                    share_data, valid_shares, q_idx, n_timesteps
                )
                
                # Add panel analysis for this quantile
                panel_results = self._analyze_quantile_panel(
                    share_data, valid_shares, q_idx, n_timesteps
                )
                
                results_per_quantile[quantile_name] = {
                    'temporal': temporal_results,
                    'panel': panel_results
                }
            
            return results_per_quantile, None
        
        else:
            # For scalar metrics, analyze directly
            temporal_results = self._analyze_scalar_metric_over_time(
                share_data, valid_shares, n_timesteps
            )
            
            # Add panel analysis for scalar metrics
            panel_results = self._analyze_scalar_panel(
                share_data, valid_shares, n_timesteps
            )
            
            return {'temporal': temporal_results, 'panel': panel_results}, None
    
    def _analyze_quantile_over_time(self, share_data, valid_shares, quantile_idx, n_timesteps):
        """Analyze temporal sensitivity for a specific quantile"""
        # Initialize results arrays
        temporal_results = {
            'timesteps': np.arange(n_timesteps),
            'linear_beta1': np.full(n_timesteps, np.nan),
            'linear_r2': np.full(n_timesteps, np.nan),
            'quadratic_beta1': np.full(n_timesteps, np.nan),
            'quadratic_beta2': np.full(n_timesteps, np.nan),
            'quadratic_r2': np.full(n_timesteps, np.nan)
        }
        
        # For each timestep, regress metric ~ alpha
        for t in range(n_timesteps):
            # Extract metric values across all alpha levels at timestep t
            metric_values = []
            alpha_values = []
            
            for alpha in valid_shares:
                metric_at_t_alpha = share_data[alpha][t, quantile_idx]
                if not np.isnan(metric_at_t_alpha):
                    metric_values.append(metric_at_t_alpha)
                    alpha_values.append(alpha)
            
            # Need at least 3 points for quadratic regression
            if len(metric_values) < 3:
                continue
                
            metric_values = np.array(metric_values)
            alpha_values = np.array(alpha_values)
            
            # Linear regression: metric ~ alpha
            linear_coeffs = np.polyfit(alpha_values, metric_values, 1)
            y_pred_linear = np.polyval(linear_coeffs, alpha_values)
            ss_res = np.sum((metric_values - y_pred_linear) ** 2)
            ss_tot = np.sum((metric_values - np.mean(metric_values)) ** 2)
            r2_linear = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            temporal_results['linear_beta1'][t] = linear_coeffs[0]  # slope = impact of alpha
            temporal_results['linear_r2'][t] = r2_linear
            
            # Quadratic regression: metric ~ alpha + alpha²
            quad_coeffs = np.polyfit(alpha_values, metric_values, 2)
            y_pred_quad = np.polyval(quad_coeffs, alpha_values)
            ss_res_quad = np.sum((metric_values - y_pred_quad) ** 2)
            r2_quad = 1 - (ss_res_quad / ss_tot) if ss_tot > 0 else 0
            
            temporal_results['quadratic_beta2'][t] = quad_coeffs[0]  # α² coefficient
            temporal_results['quadratic_beta1'][t] = quad_coeffs[1]  # α coefficient  
            temporal_results['quadratic_r2'][t] = r2_quad
        
        return temporal_results
    
    def _analyze_scalar_metric_over_time(self, share_data, valid_shares, n_timesteps):
        """Analyze temporal sensitivity for scalar metrics (non-quantile)"""
        # Initialize results arrays
        temporal_results = {
            'timesteps': np.arange(n_timesteps),
            'linear_beta1': np.full(n_timesteps, np.nan),
            'linear_r2': np.full(n_timesteps, np.nan),
            'quadratic_beta1': np.full(n_timesteps, np.nan),
            'quadratic_beta2': np.full(n_timesteps, np.nan),
            'quadratic_r2': np.full(n_timesteps, np.nan)
        }
        
        # For each timestep, regress metric ~ alpha
        for t in range(n_timesteps):
            # Extract metric values across all alpha levels at timestep t
            metric_values = []
            alpha_values = []
            
            for alpha in valid_shares:
                metric_at_t_alpha = share_data[alpha][t]
                if not np.isnan(metric_at_t_alpha):
                    metric_values.append(metric_at_t_alpha)
                    alpha_values.append(alpha)
            
            # Need at least 3 points for quadratic regression
            if len(metric_values) < 3:
                continue
                
            metric_values = np.array(metric_values)
            alpha_values = np.array(alpha_values)
            
            # Linear regression: metric ~ alpha
            linear_coeffs = np.polyfit(alpha_values, metric_values, 1)
            y_pred_linear = np.polyval(linear_coeffs, alpha_values)
            ss_res = np.sum((metric_values - y_pred_linear) ** 2)
            ss_tot = np.sum((metric_values - np.mean(metric_values)) ** 2)
            r2_linear = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            temporal_results['linear_beta1'][t] = linear_coeffs[0]  # slope = impact of alpha
            temporal_results['linear_r2'][t] = r2_linear
            
            # Quadratic regression: metric ~ alpha + alpha²
            quad_coeffs = np.polyfit(alpha_values, metric_values, 2)
            y_pred_quad = np.polyval(quad_coeffs, alpha_values)
            ss_res_quad = np.sum((metric_values - y_pred_quad) ** 2)
            r2_quad = 1 - (ss_res_quad / ss_tot) if ss_tot > 0 else 0
            
            temporal_results['quadratic_beta2'][t] = quad_coeffs[0]  # α² coefficient
            temporal_results['quadratic_beta1'][t] = quad_coeffs[1]  # α coefficient  
            temporal_results['quadratic_r2'][t] = r2_quad
        
        return temporal_results
    
    def _analyze_quantile_panel(self, share_data, valid_shares, quantile_idx, n_timesteps):
        """Panel analysis for a specific quantile: pool all timesteps and regress metric ~ alpha"""
        panel_data = []
        panel_alphas = []
        
        # Pool all data across timesteps and shares
        for alpha in valid_shares:
            for t in range(n_timesteps):
                metric_value = share_data[alpha][t, quantile_idx]
                if not np.isnan(metric_value):
                    panel_data.append(metric_value)
                    panel_alphas.append(alpha)
        
        if len(panel_data) < 10:  # Need sufficient observations
            return {
                'linear_beta1': np.nan,
                'linear_r2': np.nan,
                'quadratic_beta1': np.nan,
                'quadratic_beta2': np.nan,
                'quadratic_r2': np.nan,
                'n_observations': len(panel_data)
            }
        
        panel_data = np.array(panel_data)
        panel_alphas = np.array(panel_alphas)
        
        # Linear panel regression: metric ~ alpha (pooled across time)
        linear_coeffs = np.polyfit(panel_alphas, panel_data, 1)
        y_pred_linear = np.polyval(linear_coeffs, panel_alphas)
        ss_res = np.sum((panel_data - y_pred_linear) ** 2)
        ss_tot = np.sum((panel_data - np.mean(panel_data)) ** 2)
        linear_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Quadratic panel regression
        quad_coeffs = np.polyfit(panel_alphas, panel_data, 2)
        y_pred_quad = np.polyval(quad_coeffs, panel_alphas)
        ss_res_quad = np.sum((panel_data - y_pred_quad) ** 2)
        quad_r2 = 1 - (ss_res_quad / ss_tot) if ss_tot > 0 else 0
        
        return {
            'linear_beta1': linear_coeffs[0],
            'linear_r2': linear_r2,
            'quadratic_beta1': quad_coeffs[1],  # α coefficient
            'quadratic_beta2': quad_coeffs[0],  # α² coefficient
            'quadratic_r2': quad_r2,
            'n_observations': len(panel_data)
        }
    
    def _analyze_scalar_panel(self, share_data, valid_shares, n_timesteps):
        """Panel analysis for scalar metrics: pool all timesteps and regress metric ~ alpha"""
        panel_data = []
        panel_alphas = []
        
        # Pool all data across timesteps and shares
        for alpha in valid_shares:
            for t in range(n_timesteps):
                metric_value = share_data[alpha][t]
                if not np.isnan(metric_value):
                    panel_data.append(metric_value)
                    panel_alphas.append(alpha)
        
        if len(panel_data) < 10:  # Need sufficient observations
            return {
                'linear_beta1': np.nan,
                'linear_r2': np.nan,
                'quadratic_beta1': np.nan,
                'quadratic_beta2': np.nan,
                'quadratic_r2': np.nan,
                'n_observations': len(panel_data)
            }
        
        panel_data = np.array(panel_data)
        panel_alphas = np.array(panel_alphas)
        
        # Linear panel regression: metric ~ alpha (pooled across time)
        linear_coeffs = np.polyfit(panel_alphas, panel_data, 1)
        y_pred_linear = np.polyval(linear_coeffs, panel_alphas)
        ss_res = np.sum((panel_data - y_pred_linear) ** 2)
        ss_tot = np.sum((panel_data - np.mean(panel_data)) ** 2)
        linear_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Quadratic panel regression
        quad_coeffs = np.polyfit(panel_alphas, panel_data, 2)
        y_pred_quad = np.polyval(quad_coeffs, panel_alphas)
        ss_res_quad = np.sum((panel_data - y_pred_quad) ** 2)
        quad_r2 = 1 - (ss_res_quad / ss_tot) if ss_tot > 0 else 0
        
        return {
            'linear_beta1': linear_coeffs[0],
            'linear_r2': linear_r2,
            'quadratic_beta1': quad_coeffs[1],  # α coefficient
            'quadratic_beta2': quad_coeffs[0],  # α² coefficient
            'quadratic_r2': quad_r2,
            'n_observations': len(panel_data)
        }
    
    def analyze_polynomial_sensitivity(self, param_data):
        """Analyze sensitivity for polynomial estimates (betas)"""
        poly_data = param_data['metrics'].get('poly_estimates_avg', {})
        if not poly_data:
            return None
            
        shares = param_data['shares']
        
        # poly_estimates_avg should have shape (n_timesteps, n_coefficients)
        # We want to analyze β₁ and β₂ separately
        timestep_results = {}
        panel_results = {}
        
        # Get first valid data to determine structure
        first_share = None
        for share in shares:
            if share in poly_data and poly_data[share] is not None:
                first_share = share
                break
        
        if first_share is None:
            return None
            
        poly_array = poly_data[first_share]
        if poly_array.ndim != 2:
            return None
            
        n_timesteps, n_coeffs = poly_array.shape
        
        # Analyze each coefficient (β₁, β₂, etc.)
        for coeff_idx in range(min(2, n_coeffs)):  # Focus on β₁ and β₂
            coeff_name = f'beta_{coeff_idx + 1}'
            
            # Extract coefficient time series for all shares
            coeff_data = []
            valid_shares = []
            
            for share in shares:
                if share in poly_data and poly_data[share] is not None:
                    coeff_data.append(poly_data[share][:, coeff_idx])
                    valid_shares.append(share)
            
            if not coeff_data:
                continue
                
            coeff_data = np.array(coeff_data)  # Shape: (n_shares, n_timesteps)
            valid_shares = np.array(valid_shares)
            
            # Fit sensitivity models for this coefficient
            per_timestep, panel = self._fit_coefficient_sensitivity(
                valid_shares, coeff_data, coeff_name
            )
            
            timestep_results[coeff_name] = per_timestep
            panel_results[coeff_name] = panel
        
        return {'per_timestep': timestep_results, 'panel': panel_results}
    
    def _fit_coefficient_sensitivity(self, shares, coeff_data, coeff_name):
        """Helper function to fit sensitivity models for polynomial coefficients"""
        n_timesteps = coeff_data.shape[1]
        
        # Helper function to extract scalars
        def extract_scalar_local(coeff):
            if hasattr(coeff, 'shape'):
                if coeff.shape == ():
                    return float(coeff)
                elif len(coeff.shape) == 1 and coeff.shape[0] == 1:
                    return float(coeff[0])
                else:
                    return float(coeff.flat[0])  # Take first element
            elif hasattr(coeff, 'item'):
                return coeff.item()
            else:
                return float(coeff)
        
        # Per-timestep analysis
        per_timestep = {
            'timesteps': np.arange(n_timesteps),
            'linear_beta1': np.zeros(n_timesteps),
            'linear_r2': np.zeros(n_timesteps),
            'quadratic_beta1': np.zeros(n_timesteps), 
            'quadratic_beta2': np.zeros(n_timesteps),
            'quadratic_r2': np.zeros(n_timesteps)
        }
        
        for t in range(n_timesteps):
            y = coeff_data[:, t]
            X = shares.reshape(-1, 1)
            
            # Filter out NaN values
            valid_mask = ~np.isnan(y)
            if valid_mask.sum() < 3:  # Need at least 3 points for quadratic fit
                per_timestep['linear_beta1'][t] = np.nan
                per_timestep['linear_r2'][t] = np.nan
                per_timestep['quadratic_beta1'][t] = np.nan
                per_timestep['quadratic_beta2'][t] = np.nan
                per_timestep['quadratic_r2'][t] = np.nan
                continue
            
            y_valid = y[valid_mask]
            X_valid = X[valid_mask].flatten()  # np.polyfit expects 1D arrays
            
            # Linear model using np.polyfit (degree=1)
            linear_coeffs = np.polyfit(X_valid, y_valid, 1)
            per_timestep['linear_beta1'][t] = linear_coeffs[0]  # slope
            
            # Calculate R² for linear model
            y_pred_linear = np.polyval(linear_coeffs, X_valid)
            ss_res_linear = np.sum((y_valid - y_pred_linear) ** 2)
            ss_tot_linear = np.sum((y_valid - np.mean(y_valid)) ** 2)
            per_timestep['linear_r2'][t] = 1 - (ss_res_linear / ss_tot_linear) if ss_tot_linear > 0 else 0
            
            # Quadratic model using np.polyfit (degree=2)
            quad_coeffs = np.polyfit(X_valid, y_valid, 2)
            per_timestep['quadratic_beta2'][t] = quad_coeffs[0]  # α² coefficient
            per_timestep['quadratic_beta1'][t] = quad_coeffs[1]  # α coefficient
            
            # Calculate R² for quadratic model
            y_pred_quad = np.polyval(quad_coeffs, X_valid)
            ss_res_quad = np.sum((y_valid - y_pred_quad) ** 2)
            ss_tot_quad = np.sum((y_valid - np.mean(y_valid)) ** 2)
            per_timestep['quadratic_r2'][t] = 1 - (ss_res_quad / ss_tot_quad) if ss_tot_quad > 0 else 0
        
        # Panel analysis (similar to main sensitivity analysis)
        panel_X = []
        panel_y = []
        panel_time = []
        
        for i, share in enumerate(shares):
            for t in range(n_timesteps):
                if not np.isnan(coeff_data[i, t]):  # Filter out NaN values
                    panel_X.append([share])
                    panel_y.append(coeff_data[i, t])
                    panel_time.append(t)
        
        panel_X = np.array(panel_X)
        panel_y = np.array(panel_y)
        panel_time = np.array(panel_time)
        
        # Check if we have enough valid data points
        if len(panel_y) < 10:  # Need sufficient data for panel analysis
            print(f"Warning: Only {len(panel_y)} valid data points for panel analysis")
            panel = {
                'linear_beta1': np.nan,
                'linear_r2': np.nan,
                'quadratic_beta1': np.nan,
                'quadratic_beta2': np.nan,
                'quadratic_r2': np.nan
            }
        else:
            # Panel models without time fixed effects for better performance
            # Linear panel model
            panel_linear_model = LinearRegression()
            panel_linear_model.fit(panel_X, panel_y)
            
            # Quadratic panel model
            poly_features_panel = PolynomialFeatures(degree=2, include_bias=True)
            X_poly = poly_features_panel.fit_transform(panel_X)
            
            panel_quad_model = LinearRegression()
            panel_quad_model.fit(X_poly, panel_y)
        
            # Helper function to extract scalars
            def extract_scalar_panel(coeff):
                if hasattr(coeff, 'shape'):
                    if coeff.shape == ():
                        return float(coeff)
                    elif len(coeff.shape) == 1 and coeff.shape[0] == 1:
                        return float(coeff[0])
                    else:
                        return float(coeff.flat[0])  # Take first element
                elif hasattr(coeff, 'item'):
                    return coeff.item()
                else:
                    return float(coeff)
                    
            panel = {
                'linear_beta1': float(panel_linear_model.coef_[0]),
                'linear_r2': panel_linear_model.score(panel_X, panel_y),
                'quadratic_beta1': float(panel_quad_model.coef_[1]),  # α term (after intercept)
                'quadratic_beta2': float(panel_quad_model.coef_[2]),  # α² term
                'quadratic_r2': panel_quad_model.score(X_poly, panel_y),
                'n_observations': len(panel_y)
            }
        
        return per_timestep, panel
    
    def analyze_mobility_sensitivity(self, param_data):
        """Analyze sensitivity for rank mobility metrics"""
        # Helper function for mobility coefficients
        def extract_scalar_mobility(coeff):
            if hasattr(coeff, 'shape'):
                if coeff.shape == ():
                    return float(coeff)
                elif len(coeff.shape) == 1 and coeff.shape[0] == 1:
                    return float(coeff[0])
                else:
                    return float(coeff.flat[0])  # Take first element
            elif hasattr(coeff, 'item'):
                return coeff.item()
            else:
                return float(coeff)
        
        mobility_results = {}
        
        for mobility_metric in ['rank_ranges', 'rank_std']:
            mobility_data = param_data['metrics'].get(mobility_metric, {})
            if not mobility_data:
                continue
                
            shares = param_data['shares']
            
            # Aggregate mobility data across experiments for each share
            aggregated_mobility = {}
            for share in shares:
                if share in mobility_data and mobility_data[share] is not None:
                    mob_data = mobility_data[share]

                    # mobility_data[share] has shape (n_experiments, n_markets, n_firms_per_market)
                    # We want to aggregate across all three dimensions to get a single scalar
                    if isinstance(mob_data, np.ndarray) and mob_data.ndim == 3:
                        # Mean across experiments, markets, and firms
                        overall_mean = np.mean(mob_data)
                        aggregated_mobility[share] = overall_mean
                    elif isinstance(mob_data, np.ndarray) and mob_data.ndim == 2:
                        # 2D array: (n_experiments, n_firms) - for backward compatibility
                        mean_per_firm = np.mean(mob_data, axis=0)
                        overall_mean = np.mean(mean_per_firm)
                        aggregated_mobility[share] = overall_mean
                    elif np.isscalar(mob_data) or (isinstance(mob_data, np.ndarray) and mob_data.ndim == 0):
                        # Already aggregated to a scalar
                        aggregated_mobility[share] = float(mob_data)
            
            if not aggregated_mobility:
                continue
                
            # Fit sensitivity models
            valid_shares = np.array(list(aggregated_mobility.keys()))
            mobility_values = np.array(list(aggregated_mobility.values()))
            
            # Linear model using np.polyfit
            linear_coeffs = np.polyfit(valid_shares, mobility_values, 1)
            y_pred_linear = np.polyval(linear_coeffs, valid_shares)
            ss_res_linear = np.sum((mobility_values - y_pred_linear) ** 2)
            ss_tot_linear = np.sum((mobility_values - np.mean(mobility_values)) ** 2)
            linear_r2 = 1 - (ss_res_linear / ss_tot_linear) if ss_tot_linear > 0 else 0
            
            # Quadratic model using np.polyfit
            quad_coeffs = np.polyfit(valid_shares, mobility_values, 2)
            y_pred_quad = np.polyval(quad_coeffs, valid_shares)
            ss_res_quad = np.sum((mobility_values - y_pred_quad) ** 2)
            ss_tot_quad = np.sum((mobility_values - np.mean(mobility_values)) ** 2)
            quad_r2 = 1 - (ss_res_quad / ss_tot_quad) if ss_tot_quad > 0 else 0
            
            mobility_results[mobility_metric] = {
                'shares': valid_shares,
                'values': mobility_values,
                'linear_beta1': linear_coeffs[0],  # slope
                'linear_r2': linear_r2,
                'quadratic_beta1': quad_coeffs[1],  # α coefficient 
                'quadratic_beta2': quad_coeffs[0],  # α² coefficient
                'quadratic_r2': quad_r2
            }
        
        return mobility_results
    
    def run_cross_parametrization_panel_analysis(self, all_data, control_hyperparams=True):
        """
        Run panel analysis pooling data across ALL parametrizations with hyperparameter controls
        
        This allows isolating the pure effect of alpha while controlling for structural differences
        between parametrizations (markets, firms_per_market, etc.)
        """
        print("Running cross-parametrization panel analysis with controls...")
        
        # Collect all data across parametrizations
        panel_data = []
        
        for param_name, param_data in all_data.items():
            shares = param_data['shares']
            hyperparams = param_data['hyperparameters']
            
            # Extract key hyperparameters for controls
            markets = hyperparams.get('markets', 100)
            firms_per_market = hyperparams.get('firms_per_market', 100)
            steps = hyperparams.get('steps', 10000)
            merge_thresh = hyperparams.get('merge_thresh', 0.05)
            break_thresh = hyperparams.get('break_thresh', 0.85)
            lookback = hyperparams.get('lookback', 50)
            cost_type = param_data['cost_type']

            # For each metric, collect panel data
            for metric_name in ['gini_quantiles_avg', 'market_share_quantiles_avg']:
                metric_data = param_data['metrics'].get(metric_name, {})
                
                if metric_data:
                    for alpha in shares:
                        if alpha in metric_data and metric_data[alpha] is not None:
                            data_array = metric_data[alpha]
                            
                            # Handle potential transpose issues
                            if metric_name == 'market_share_quantiles_avg' and data_array.shape[0] < data_array.shape[1]:
                                data_array = data_array.T
                            
                            n_timesteps, n_quantiles = data_array.shape
                            quantile_names = ['10th', '25th', '50th', '75th', '90th', '99th', '100th']
                            
                            # For each quantile and timestep
                            for q_idx, quantile in enumerate(quantile_names[:n_quantiles]):
                                for t in range(n_timesteps):
                                    metric_value = data_array[t, q_idx]
                                    if not np.isnan(metric_value):
                                        panel_data.append({
                                            'metric_name': metric_name,
                                            'quantile': quantile,
                                            'alpha': alpha,
                                            'metric_value': metric_value,
                                            'timestep': t,
                                            'param_name': param_name,
                                            'cost_type': cost_type,
                                            'markets': markets,
                                            'firms_per_market': firms_per_market,
                                            'total_firms': markets * firms_per_market,
                                            'steps': steps,
                                            'merge_thresh': merge_thresh,
                                            'break_thresh': break_thresh,
                                            'lookback': lookback
                                        })

        # Add scalar metrics (pool across timesteps)
        for param_name, param_data in all_data.items():
            shares = param_data['shares']
            hyperparams = param_data['hyperparameters']

            # Extract key hyperparameters for controls
            markets = hyperparams.get('markets', 100)
            firms_per_market = hyperparams.get('firms_per_market', 100)
            steps = hyperparams.get('steps', 10000)
            merge_thresh = hyperparams.get('merge_thresh', 0.05)
            break_thresh = hyperparams.get('break_thresh', 0.85)
            lookback = hyperparams.get('lookback', 50)
            cost_type = param_data['cost_type']

            # Process scalar metrics (mean_members_avg, num_cong_avg, mergers_per_period_avg, exits_per_period_avg)
            for scalar_metric_name in ['mean_members_avg', 'num_cong_avg', 'mergers_per_period_avg', 'exits_per_period_avg']:
                scalar_metric_data = param_data['metrics'].get(scalar_metric_name, {})
                if scalar_metric_data:
                    for alpha in shares:
                        if alpha in scalar_metric_data and scalar_metric_data[alpha] is not None:
                            data_array = scalar_metric_data[alpha]

                            # Scalar metrics have shape (n_timesteps,)
                            if isinstance(data_array, np.ndarray) and data_array.ndim == 1:
                                n_timesteps = len(data_array)

                                for t in range(n_timesteps):
                                    metric_value = data_array[t]
                                    if not np.isnan(metric_value):
                                        panel_data.append({
                                            'metric_name': scalar_metric_name,
                                            'quantile': 'overall',  # No quantiles for scalar metrics
                                            'alpha': alpha,
                                            'metric_value': metric_value,
                                            'timestep': t,
                                            'param_name': param_name,
                                            'cost_type': cost_type,
                                            'markets': markets,
                                            'firms_per_market': firms_per_market,
                                            'total_firms': markets * firms_per_market,
                                            'steps': steps,
                                            'merge_thresh': merge_thresh,
                                            'break_thresh': break_thresh,
                                            'lookback': lookback
                                        })

        # Add mobility metrics (no timestep dimension - already averaged over time)
        for param_name, param_data in all_data.items():
            shares = param_data['shares']
            hyperparams = param_data['hyperparameters']

            # Extract key hyperparameters for controls
            markets = hyperparams.get('markets', 100)
            lookback = hyperparams.get('lookback', 50)
            firms_per_market = hyperparams.get('firms_per_market', 100)
            steps = hyperparams.get('steps', 10000)
            merge_thresh = hyperparams.get('merge_thresh', 0.05)
            break_thresh = hyperparams.get('break_thresh', 0.85)
            cost_type = param_data['cost_type']

            # Check if this parametrization has mobility data
            if 'mobility' in param_data:
                mobility_data = param_data['mobility']

                # Process each mobility metric
                for mobility_metric in ['rank_ranges', 'rank_std']:
                    if mobility_metric in mobility_data:
                        mob = mobility_data[mobility_metric]

                        # mob has 'shares' and 'values' arrays (26 points each)
                        for i, alpha in enumerate(mob['shares']):
                            mobility_value = mob['values'][i]
                            if not np.isnan(mobility_value):
                                panel_data.append({
                                    'metric_name': mobility_metric,
                                    'quantile': 'overall',  # No quantiles for mobility
                                    'alpha': alpha,
                                    'metric_value': mobility_value,
                                    'timestep': None,  # No timestep for mobility
                                    'param_name': param_name,
                                    'cost_type': cost_type,
                                    'markets': markets,
                                    'firms_per_market': firms_per_market,
                                    'total_firms': markets * firms_per_market,
                                    'steps': steps,
                                    'merge_thresh': merge_thresh,
                                    'break_thresh': break_thresh,
                                    'lookback': lookback
                                })

        # Add polynomial metrics (pool across timesteps)
        for param_name, param_data in all_data.items():
            shares = param_data['shares']
            hyperparams = param_data['hyperparameters']

            # Extract key hyperparameters for controls
            markets = hyperparams.get('markets', 100)
            firms_per_market = hyperparams.get('firms_per_market', 100)
            steps = hyperparams.get('steps', 10000)
            merge_thresh = hyperparams.get('merge_thresh', 0.05)
            break_thresh = hyperparams.get('break_thresh', 0.85)
            lookback = hyperparams.get('lookback', 50)
            cost_type = param_data['cost_type']

            # Process polynomial estimates
            poly_metric_data = param_data['metrics'].get('poly_estimates_avg', {})
            if poly_metric_data:
                for alpha in shares:
                    if alpha in poly_metric_data and poly_metric_data[alpha] is not None:
                        poly_array = poly_metric_data[alpha]  # Shape: (n_timesteps, n_coeffs)

                        if isinstance(poly_array, np.ndarray) and poly_array.ndim == 2:
                            n_timesteps, n_coeffs = poly_array.shape

                            # Process beta_1 (coefficient index 1) and beta_2 (coefficient index 0)
                            for coeff_idx, coeff_name in [(1, 'poly_beta_1'), (0, 'poly_beta_2')]:
                                if coeff_idx < n_coeffs:
                                    for t in range(n_timesteps):
                                        beta_value = poly_array[t, coeff_idx]
                                        if not np.isnan(beta_value):
                                            panel_data.append({
                                                'metric_name': coeff_name,
                                                'quantile': 'overall',  # No quantiles for polynomial estimates
                                                'alpha': alpha,
                                                'metric_value': beta_value,
                                                'timestep': t,
                                                'param_name': param_name,
                                                'cost_type': cost_type,
                                                'markets': markets,
                                                'firms_per_market': firms_per_market,
                                                'total_firms': markets * firms_per_market,
                                                'steps': steps,
                                                'merge_thresh': merge_thresh,
                                                'break_thresh': break_thresh,
                                                'lookback': lookback
                                            })

        if not panel_data:
            print("No panel data collected!")
            return None
        
        # Convert to DataFrame for easier regression
        import pandas as pd
        df = pd.DataFrame(panel_data)
        
        print(f"Collected {len(df)} observations across {df['param_name'].nunique()} parametrizations")
        
        # Run controlled regressions for each metric-quantile combination
        results = {}

        for metric_name in df['metric_name'].unique():
            results[metric_name] = {}
            metric_df = df[df['metric_name'] == metric_name]

            for quantile in metric_df['quantile'].unique():
                quantile_df = metric_df[metric_df['quantile'] == quantile]

                if len(quantile_df) < 50:  # Need sufficient observations
                    continue

                # Debug: Check variation in interaction variables (only print once)
                if metric_name == 'mean_members_avg' and quantile == 'overall':
                    print(f"  DEBUG - Unique values in quantile_df:")
                    print(f"    lookback: {quantile_df['lookback'].nunique() if 'lookback' in quantile_df.columns else 'NOT IN COLUMNS'}")
                    print(f"    markets: {quantile_df['markets'].nunique() if 'markets' in quantile_df.columns else 'NOT IN COLUMNS'}")
                    print(f"    firms_per_market: {quantile_df['firms_per_market'].nunique() if 'firms_per_market' in quantile_df.columns else 'NOT IN COLUMNS'}")
                    if 'lookback' in quantile_df.columns:
                        print(f"    lookback values: {sorted(quantile_df['lookback'].unique())}")
                    if 'markets' in quantile_df.columns:
                        print(f"    markets values: {sorted(quantile_df['markets'].unique())}")
                    if 'firms_per_market' in quantile_df.columns:
                        print(f"    firms_per_market values: {sorted(quantile_df['firms_per_market'].unique())}")

                # Run both linear and quadratic regressions with hyperparameter controls
                linear_results = self._run_controlled_regression(quantile_df, control_hyperparams, model_type='linear')
                quadratic_results = self._run_controlled_regression(quantile_df, control_hyperparams, model_type='quadratic')

                # Store both models
                results[metric_name][quantile] = {
                    'linear': linear_results,
                    'quadratic': quadratic_results
                }

                print(f"{metric_name} {quantile}:")
                print(f"  Linear: α coeff = {linear_results['alpha_coeff']:.6f} "
                      f"(t = {linear_results['alpha_tstat']:.2f}, R² = {linear_results['r2']:.4f})")
                print(f"  Quadratic: β₁ = {quadratic_results['alpha_coeff']:.6f}, "
                      f"β₂ = {quadratic_results['alpha_sq_coeff']:.6f} "
                      f"(R² = {quadratic_results['r2']:.4f})")

        return results
    
    def run_controlled_temporal_analysis(self, all_data, control_hyperparams=True):
        """
        Run controlled temporal analysis pooling within cost functions at each timestep
        
        For each (cost_function, timestep, quantile):
        metric_value_t = β₀ + β₁·alpha + β₂·markets + β₃·total_firms + β₄·merge_thresh + ε
        """
        print("Running controlled temporal analysis within cost functions...")
        
        # Collect all data organized by cost function
        temporal_data = []
        
        for param_name, param_data in all_data.items():
            shares = param_data['shares']
            hyperparams = param_data['hyperparameters']
            cost_type = param_data['cost_type']
            
            # Extract hyperparameters for controls
            markets = hyperparams.get('markets', 100)
            firms_per_market = hyperparams.get('firms_per_market', 100)
            total_firms = markets * firms_per_market
            merge_thresh = hyperparams.get('merge_thresh', 0.05)
            break_thresh = hyperparams.get('break_thresh', 0.85)
            
            # For quantile metrics, collect temporal data
            for metric_name in ['gini_quantiles_avg', 'market_share_quantiles_avg']:
                metric_data = param_data['metrics'].get(metric_name, {})

                if metric_data:
                    for alpha in shares:
                        if alpha in metric_data and metric_data[alpha] is not None:
                            data_array = metric_data[alpha]

                            # Handle potential transpose issues
                            if metric_name == 'market_share_quantiles_avg' and data_array.shape[0] < data_array.shape[1]:
                                data_array = data_array.T

                            n_timesteps, n_quantiles = data_array.shape
                            quantile_names = ['10th', '25th', '50th', '75th', '90th', '99th', '100th']

                            # For each quantile and timestep
                            for q_idx, quantile in enumerate(quantile_names[:n_quantiles]):
                                for t in range(n_timesteps):
                                    metric_value = data_array[t, q_idx]
                                    if not np.isnan(metric_value):
                                        temporal_data.append({
                                            'metric_name': metric_name,
                                            'quantile': quantile,
                                            'alpha': alpha,
                                            'metric_value': metric_value,
                                            'timestep': t,
                                            'cost_type': cost_type,
                                            'markets': markets,
                                            'firms_per_market': firms_per_market,
                                            'total_firms': total_firms,
                                            'merge_thresh': merge_thresh,
                                            'break_thresh': break_thresh,
                                            'param_name': param_name
                                        })

            # For scalar metrics, collect temporal data
            for metric_name in ['mean_members_avg', 'num_cong_avg', 'mergers_per_period_avg', 'exits_per_period_avg']:
                metric_data = param_data['metrics'].get(metric_name, {})

                if metric_data:
                    for alpha in shares:
                        if alpha in metric_data and metric_data[alpha] is not None:
                            data_array = metric_data[alpha]

                            # Scalar metrics have shape (n_timesteps,)
                            if isinstance(data_array, np.ndarray) and data_array.ndim == 1:
                                n_timesteps = len(data_array)

                                for t in range(n_timesteps):
                                    metric_value = data_array[t]
                                    if not np.isnan(metric_value):
                                        temporal_data.append({
                                            'metric_name': metric_name,
                                            'quantile': 'overall',  # No quantiles for scalar metrics
                                            'alpha': alpha,
                                            'metric_value': metric_value,
                                            'timestep': t,
                                            'cost_type': cost_type,
                                            'markets': markets,
                                            'firms_per_market': firms_per_market,
                                            'total_firms': total_firms,
                                            'merge_thresh': merge_thresh,
                                            'break_thresh': break_thresh,
                                            'param_name': param_name
                                        })

            # For polynomial estimates (beta_1 and beta_2 from size→market_share regressions)
            # These are stored in metrics['poly_estimates_avg'][alpha] as (timesteps, n_coeffs)
            poly_metric_data = param_data['metrics'].get('poly_estimates_avg', {})

            if poly_metric_data:
                for alpha in shares:
                    if alpha in poly_metric_data and poly_metric_data[alpha] is not None:
                        poly_array = poly_metric_data[alpha]  # Shape: (n_timesteps, n_coeffs)

                        if isinstance(poly_array, np.ndarray) and poly_array.ndim == 2:
                            n_timesteps, n_coeffs = poly_array.shape

                            # Process beta_1 (coefficient index 1) and beta_2 (coefficient index 0)
                            # Note: polyfit returns [beta_2, beta_1, beta_0] for quadratic
                            for coeff_idx, coeff_name in [(1, 'poly_beta_1'), (0, 'poly_beta_2')]:
                                if coeff_idx < n_coeffs:
                                    for t in range(n_timesteps):
                                        beta_value = poly_array[t, coeff_idx]
                                        if not np.isnan(beta_value):
                                            temporal_data.append({
                                                'metric_name': coeff_name,
                                                'quantile': 'overall',  # No quantiles for polynomial estimates
                                                'alpha': alpha,
                                                'metric_value': beta_value,
                                                'timestep': t,
                                                'cost_type': cost_type,
                                                'markets': markets,
                                                'firms_per_market': firms_per_market,
                                                'total_firms': total_firms,
                                                'merge_thresh': merge_thresh,
                                                'break_thresh': break_thresh,
                                                'param_name': param_name
                                            })

        if not temporal_data:
            print("No temporal data collected!")
            return None
        
        # Convert to DataFrame
        import pandas as pd
        df = pd.DataFrame(temporal_data)
        
        print(f"Collected {len(df)} observations across {df['param_name'].nunique()} parametrizations")
        
        # Run controlled regressions for each (cost_type, metric, quantile) at sampled timesteps
        results = {}
        
        for cost_type in df['cost_type'].unique():
            results[cost_type] = {}
            cost_df = df[df['cost_type'] == cost_type]
            
            print(f"Analyzing cost function: {cost_type}")
            
            for metric_name in cost_df['metric_name'].unique():
                results[cost_type][metric_name] = {}
                metric_df = cost_df[cost_df['metric_name'] == metric_name]
                
                for quantile in metric_df['quantile'].unique():
                    quantile_df = metric_df[metric_df['quantile'] == quantile]
                    
                    # Sample timesteps (every 100th to reduce computation)
                    max_timestep = quantile_df['timestep'].max()
                    timestep_sample = list(range(0, max_timestep + 1, 100))
                    
                    # Store results for both linear and quadratic models
                    linear_results = {
                        'timesteps': [],
                        'alpha_coeff': [],
                        'alpha_se': [],  # Standard error for confidence bands
                        'alpha_tstat': [],
                        'r2': [],
                        'n_obs': []
                    }

                    quadratic_results = {
                        'timesteps': [],
                        'alpha_coeff': [],  # β₁ from quadratic model
                        'alpha_se': [],  # Standard error for β₁
                        'alpha_sq_coeff': [],  # β₂ from quadratic model
                        'alpha_sq_se': [],  # Standard error for β₂
                        'alpha_tstat': [],
                        'alpha_sq_tstat': [],
                        'r2': [],
                        'n_obs': []
                    }

                    # Run regression for each sampled timestep
                    for t in timestep_sample:
                        timestep_df = quantile_df[quantile_df['timestep'] == t]

                        # Need sufficient observations and variation in hyperparameters
                        if len(timestep_df) < 10:
                            continue

                        # Check if there's variation in hyperparameters within this cost function
                        if timestep_df['total_firms'].nunique() <= 1 and timestep_df['merge_thresh'].nunique() <= 1:
                            # No hyperparameter variation - skip controls
                            continue

                        try:
                            # Run linear controlled regression
                            linear_reg = self._run_controlled_regression(timestep_df, control_hyperparams, model_type='linear')

                            linear_results['timesteps'].append(t)
                            linear_results['alpha_coeff'].append(linear_reg['alpha_coeff'])
                            linear_results['alpha_se'].append(linear_reg['alpha_se'])
                            linear_results['alpha_tstat'].append(linear_reg['alpha_tstat'])
                            linear_results['r2'].append(linear_reg['r2'])
                            linear_results['n_obs'].append(linear_reg['n_obs'])

                            # Run quadratic controlled regression
                            quad_reg = self._run_controlled_regression(timestep_df, control_hyperparams, model_type='quadratic')

                            quadratic_results['timesteps'].append(t)
                            quadratic_results['alpha_coeff'].append(quad_reg['alpha_coeff'])
                            quadratic_results['alpha_se'].append(quad_reg['alpha_se'])
                            quadratic_results['alpha_sq_coeff'].append(quad_reg['alpha_sq_coeff'])
                            quadratic_results['alpha_sq_se'].append(quad_reg['alpha_sq_se'])
                            quadratic_results['alpha_tstat'].append(quad_reg['alpha_tstat'])
                            quadratic_results['alpha_sq_tstat'].append(quad_reg['alpha_sq_tstat'])
                            quadratic_results['r2'].append(quad_reg['r2'])
                            quadratic_results['n_obs'].append(quad_reg['n_obs'])

                        except Exception as e:
                            # Skip if regression fails
                            continue

                    # Store results if we have any
                    if linear_results['timesteps']:
                        # Convert to numpy arrays
                        for key in linear_results.keys():
                            linear_results[key] = np.array(linear_results[key])
                        for key in quadratic_results.keys():
                            quadratic_results[key] = np.array(quadratic_results[key])

                        # Store both models
                        results[cost_type][metric_name][quantile] = {
                            'linear': linear_results,
                            'quadratic': quadratic_results
                        }

                        print(f"  {metric_name} {quantile}: {len(linear_results['timesteps'])} timesteps analyzed (linear + quadratic)")
        
        return results
    
    def _run_controlled_regression(self, df, control_hyperparams=True, model_type='linear'):
        """Run regression with optional hyperparameter controls

        Args:
            df: DataFrame with regression data
            control_hyperparams: Whether to include hyperparameter controls
            model_type: 'linear' or 'quadratic' - controls alpha polynomial terms
        """
        from sklearn.linear_model import LinearRegression
        from scipy import stats
        import numpy as np

        # Dependent variable
        y = df['metric_value'].values

        # Independent variables - always start with alpha
        X_vars = ['alpha']
        X_data = [df['alpha'].values]

        # Add alpha^2 for quadratic models
        if model_type == 'quadratic':
            X_vars.append('alpha_squared')
            X_data.append(df['alpha'].values ** 2)

        if control_hyperparams:
            # Add hyperparameter controls
            control_vars = ['markets', 'total_firms', 'merge_thresh', 'break_thresh']
            for var in control_vars:
                if var in df.columns and df[var].nunique() > 1:  # Only add if there's variation
                    X_vars.append(var)
                    X_data.append(df[var].values)

            # Add interaction terms: lookback x alpha, markets x alpha, firms_per_market x alpha
            interaction_vars = ['lookback', 'markets', 'firms_per_market']
            for var in interaction_vars:
                if var in df.columns and df[var].nunique() > 1:  # Only add if there's variation
                    interaction_name = f'{var}_x_alpha'
                    X_vars.append(interaction_name)
                    X_data.append(df[var].values * df['alpha'].values)

            # Add cost function fixed effects (dummy variables)
            cost_types = df['cost_type'].unique()
            if len(cost_types) > 1:
                for cost_type in cost_types[1:]:  # Omit first as reference category
                    X_vars.append(f'cost_{cost_type}')
                    X_data.append((df['cost_type'] == cost_type).astype(int).values)

        # Stack into design matrix
        X = np.column_stack(X_data)

        # Add intercept
        X = np.column_stack([np.ones(len(X)), X])
        var_names = ['intercept'] + X_vars

        # Run regression
        model = LinearRegression(fit_intercept=False)  # We added intercept manually
        model.fit(X, y)

        # Calculate standard errors and t-statistics
        y_pred = model.predict(X)
        residuals = y - y_pred
        mse = np.sum(residuals**2) / (len(y) - len(var_names))
        var_covar_matrix = mse * np.linalg.inv(X.T @ X)
        standard_errors = np.sqrt(np.diag(var_covar_matrix))
        t_statistics = model.coef_ / standard_errors

        # Find alpha coefficient indices
        alpha_idx = 1  # Alpha is always first after intercept
        alpha_sq_idx = 2 if model_type == 'quadratic' else None

        result = {
            'coefficients': dict(zip(var_names, model.coef_)),
            'standard_errors': dict(zip(var_names, standard_errors)),
            't_statistics': dict(zip(var_names, t_statistics)),
            'alpha_coeff': model.coef_[alpha_idx],
            'alpha_se': standard_errors[alpha_idx],
            'alpha_tstat': t_statistics[alpha_idx],
            'r2': model.score(X, y),
            'n_obs': len(y),
            'controls': control_hyperparams,
            'control_vars': X_vars[1:] if control_hyperparams else [],
            'model_type': model_type
        }

        # Add quadratic coefficient if applicable
        if model_type == 'quadratic':
            result['alpha_sq_coeff'] = model.coef_[alpha_sq_idx]
            result['alpha_sq_se'] = standard_errors[alpha_sq_idx]
            result['alpha_sq_tstat'] = t_statistics[alpha_sq_idx]

        # Add interaction term coefficients if applicable
        if control_hyperparams:
            for var in ['lookback', 'markets', 'firms_per_market']:
                interaction_name = f'{var}_x_alpha'
                if interaction_name in var_names:
                    idx = var_names.index(interaction_name)
                    result[f'{interaction_name}_coeff'] = model.coef_[idx]
                    result[f'{interaction_name}_se'] = standard_errors[idx]
                    result[f'{interaction_name}_tstat'] = t_statistics[idx]

        return result

    def create_polynomial_panel_table(self, results, save_path=None):
        """Create table summarizing polynomial panel regression results

        Shows how profit-sharing (α) affects the polynomial relationship between
        conglomerate size and market share.

        Args:
            results: Results dictionary containing controlled_panel data
            save_path: Optional path to save the table (supports .csv, .tex, .txt)

        Returns:
            DataFrame with polynomial panel regression summary
        """
        import pandas as pd

        panel_results = results.get('cross_parametrization', {}).get('controlled_panel', {})

        if not panel_results:
            print("No controlled panel results found!")
            return None

        # Extract polynomial metrics results
        table_rows = []

        for metric_name in ['poly_beta_1', 'poly_beta_2']:
            if metric_name not in panel_results:
                continue

            metric_results = panel_results[metric_name].get('overall', {})

            if not metric_results:
                continue

            linear_res = metric_results.get('linear', {})
            quadratic_res = metric_results.get('quadratic', {})

            # Build row for this metric
            row = {
                'Metric': metric_name,
                'Linear β₁': linear_res.get('alpha_coeff', np.nan),
                'Linear SE': linear_res.get('alpha_se', np.nan),
                'Linear t-stat': linear_res.get('alpha_tstat', np.nan),
                'Quadratic β₁': quadratic_res.get('alpha_coeff', np.nan),
                'Quadratic β₂': quadratic_res.get('alpha_sq_coeff', np.nan),
                'Quadratic SE(β₁)': quadratic_res.get('alpha_se', np.nan),
                'Quadratic SE(β₂)': quadratic_res.get('alpha_sq_se', np.nan),
                'Quadratic t(β₁)': quadratic_res.get('alpha_tstat', np.nan),
                'Quadratic t(β₂)': quadratic_res.get('alpha_sq_tstat', np.nan),
                'R² (Quadratic)': quadratic_res.get('r2', np.nan),
                'N': quadratic_res.get('n_obs', 0)
            }

            table_rows.append(row)

        if not table_rows:
            print("No polynomial metrics found in panel results!")
            return None

        # Create DataFrame
        df = pd.DataFrame(table_rows)

        # Format numerical columns
        for col in df.columns:
            if col not in ['Metric', 'N']:
                df[col] = df[col].apply(lambda x: f"{x:.4f}" if not np.isnan(x) else "—")

        # Print to console
        print("\n" + "="*120)
        print("POLYNOMIAL PANEL REGRESSION RESULTS")
        print("Dependent Variable: Polynomial coefficient value")
        print("Independent Variables: α (profit-sharing), α² (quadratic term), + hyperparameter controls")
        print("="*120)
        print(df.to_string(index=False))
        print("="*120)
        print(f"\nNote: Results pool observations across timesteps and parametrizations.")
        print(f"Controls include: markets, firms_per_market, merge_thresh, cost_type fixed effects")
        print("="*120 + "\n")

        # Save if requested
        if save_path:
            save_path = Path(save_path)

            if save_path.suffix == '.csv':
                df.to_csv(save_path, index=False)
                print(f"Table saved to {save_path}")

            elif save_path.suffix == '.tex':
                # LaTeX formatting
                latex_str = df.to_latex(
                    index=False,
                    caption="Polynomial panel regression results showing how profit-sharing (α) affects the size-market share relationship",
                    label="tab:polynomial_panel",
                    escape=False,
                    column_format='l' + 'r'*(len(df.columns)-1)
                )
                with open(save_path, 'w') as f:
                    f.write(latex_str)
                print(f"LaTeX table saved to {save_path}")

            else:  # Default to text
                with open(save_path, 'w') as f:
                    f.write("="*120 + "\n")
                    f.write("POLYNOMIAL PANEL REGRESSION RESULTS\n")
                    f.write("="*120 + "\n\n")
                    f.write(df.to_string(index=False))
                    f.write("\n\n" + "="*120 + "\n")
                    f.write("Note: Results pool observations across timesteps and parametrizations.\n")
                    f.write("Controls include: markets, firms_per_market, merge_thresh, cost_type fixed effects\n")
                    f.write("="*120 + "\n")
                print(f"Text table saved to {save_path}")

        return df

    def create_scalar_panel_table(self, results, save_path=None):
        """Create table summarizing scalar metric panel regression results

        Shows how profit-sharing (α) affects average conglomerate size and number of conglomerates.

        Args:
            results: Results dictionary containing controlled_panel data
            save_path: Optional path to save the table (supports .csv, .tex, .txt)

        Returns:
            DataFrame with scalar panel regression summary
        """
        import pandas as pd

        panel_results = results.get('cross_parametrization', {}).get('controlled_panel', {})

        if not panel_results:
            print("No controlled panel results found!")
            return None

        # Extract scalar metrics results
        table_rows = []

        for metric_name in ['mean_members_avg', 'num_cong_avg', 'mergers_per_period_avg', 'exits_per_period_avg']:
            if metric_name not in panel_results:
                continue

            metric_results = panel_results[metric_name].get('overall', {})

            if not metric_results:
                continue

            linear_res = metric_results.get('linear', {})
            quadratic_res = metric_results.get('quadratic', {})

            # Build row for this metric
            row = {
                'Metric': metric_name,
                'Linear β₁': linear_res.get('alpha_coeff', np.nan),
                'Linear SE': linear_res.get('alpha_se', np.nan),
                'Linear t-stat': linear_res.get('alpha_tstat', np.nan),
                'Quadratic β₁': quadratic_res.get('alpha_coeff', np.nan),
                'Quadratic β₂': quadratic_res.get('alpha_sq_coeff', np.nan),
                'Quadratic SE(β₁)': quadratic_res.get('alpha_se', np.nan),
                'Quadratic SE(β₂)': quadratic_res.get('alpha_sq_se', np.nan),
                'Quadratic t(β₁)': quadratic_res.get('alpha_tstat', np.nan),
                'Quadratic t(β₂)': quadratic_res.get('alpha_sq_tstat', np.nan),
                'Lookback×α': quadratic_res.get('lookback_x_alpha_coeff', np.nan),
                'Lookback×α SE': quadratic_res.get('lookback_x_alpha_se', np.nan),
                'Lookback×α t': quadratic_res.get('lookback_x_alpha_tstat', np.nan),
                'Markets×α': quadratic_res.get('markets_x_alpha_coeff', np.nan),
                'Markets×α SE': quadratic_res.get('markets_x_alpha_se', np.nan),
                'Markets×α t': quadratic_res.get('markets_x_alpha_tstat', np.nan),
                'Firms/Mkt×α': quadratic_res.get('firms_per_market_x_alpha_coeff', np.nan),
                'Firms/Mkt×α SE': quadratic_res.get('firms_per_market_x_alpha_se', np.nan),
                'Firms/Mkt×α t': quadratic_res.get('firms_per_market_x_alpha_tstat', np.nan),
                'R² (Quadratic)': quadratic_res.get('r2', np.nan),
                'N': quadratic_res.get('n_obs', 0)
            }

            table_rows.append(row)

        if not table_rows:
            print("No scalar metrics found in panel results!")
            return None

        # Create DataFrame
        df = pd.DataFrame(table_rows)

        # Format numerical columns
        for col in df.columns:
            if col not in ['Metric', 'N']:
                df[col] = df[col].apply(lambda x: f"{x:.4f}" if not np.isnan(x) else "—")

        # Print to console
        print("\n" + "="*140)
        print("SCALAR METRICS PANEL REGRESSION RESULTS")
        print("Dependent Variable: Metric value (pooled across timesteps)")
        print("Independent Variables: α (profit-sharing), α² (quadratic term), interaction terms, + hyperparameter controls")
        print("="*140)
        print(df.to_string(index=False))
        print("="*140)
        print(f"\nNote: Results pool observations across timesteps and parametrizations.")
        print(f"Controls include: markets, firms_per_market, merge_thresh, cost_type fixed effects")
        print(f"Interaction terms: lookback×α, markets×α, firms_per_market×α")
        print("="*140 + "\n")

        # Save if requested
        if save_path:
            save_path = Path(save_path)

            if save_path.suffix == '.csv':
                df.to_csv(save_path, index=False)
                print(f"Table saved to {save_path}")

            elif save_path.suffix == '.tex':
                # Custom LaTeX formatting (bypasses jinja2 dependency)
                latex_str = self._create_scalar_table_latex(panel_results)
                with open(save_path, 'w') as f:
                    f.write(latex_str)
                print(f"LaTeX table saved to {save_path}")

            else:  # Default to text
                with open(save_path, 'w') as f:
                    f.write("="*140 + "\n")
                    f.write("SCALAR METRICS PANEL REGRESSION RESULTS\n")
                    f.write("="*140 + "\n\n")
                    f.write(df.to_string(index=False))
                    f.write("\n\n" + "="*140 + "\n")
                    f.write("Note: Results pool observations across timesteps and parametrizations.\n")
                    f.write("Controls include: markets, firms_per_market, merge_thresh, cost_type fixed effects\n")
                    f.write("Interaction terms: lookback×α, markets×α, firms_per_market×α\n")
                    f.write("="*140 + "\n")
                print(f"Text table saved to {save_path}")

        return df

    def _create_scalar_table_latex(self, panel_results):
        """Create custom LaTeX table for scalar metrics (bypasses jinja2 dependency)

        Creates table in exact format matching user's template with 4 columns:
        Mean Members, Number Conglomerates, Periodic Mergers, Periodic Exits
        """
        # Helper function to add significance stars
        def add_stars(t_stat):
            """Add significance stars based on t-statistic"""
            abs_t = abs(t_stat)
            if abs_t >= 2.576:  # p < 0.01
                return '***'
            elif abs_t >= 1.96:  # p < 0.05
                return '**'
            elif abs_t >= 1.645:  # p < 0.1
                return '*'
            else:
                return ''

        # Metric names and their display labels
        metrics = [
            ('mean_members_avg', 'Mean Members'),
            ('num_cong_avg', 'Number Conglomerates'),
            ('mergers_per_period_avg', 'Periodic Mergers'),
            ('exits_per_period_avg', 'Periodic Exits')
        ]

        # Extract quadratic regression results for each metric
        results_data = []
        n_obs = None
        r2_values = []

        for metric_name, display_name in metrics:
            if metric_name in panel_results:
                metric_results = panel_results[metric_name].get('overall', {})
                quad_res = metric_results.get('quadratic', {})

                alpha_coeff = quad_res.get('alpha_coeff', np.nan)
                alpha_se = quad_res.get('alpha_se', np.nan)
                alpha_tstat = quad_res.get('alpha_tstat', np.nan)
                alpha_sq_coeff = quad_res.get('alpha_sq_coeff', np.nan)
                alpha_sq_se = quad_res.get('alpha_sq_se', np.nan)
                alpha_sq_tstat = quad_res.get('alpha_sq_tstat', np.nan)
                r2 = quad_res.get('r2', np.nan)

                # Extract interaction term coefficients
                lookback_x_alpha_coeff = quad_res.get('lookback_x_alpha_coeff', np.nan)
                lookback_x_alpha_se = quad_res.get('lookback_x_alpha_se', np.nan)
                lookback_x_alpha_tstat = quad_res.get('lookback_x_alpha_tstat', np.nan)
                markets_x_alpha_coeff = quad_res.get('markets_x_alpha_coeff', np.nan)
                markets_x_alpha_se = quad_res.get('markets_x_alpha_se', np.nan)
                markets_x_alpha_tstat = quad_res.get('markets_x_alpha_tstat', np.nan)
                firms_per_market_x_alpha_coeff = quad_res.get('firms_per_market_x_alpha_coeff', np.nan)
                firms_per_market_x_alpha_se = quad_res.get('firms_per_market_x_alpha_se', np.nan)
                firms_per_market_x_alpha_tstat = quad_res.get('firms_per_market_x_alpha_tstat', np.nan)

                if n_obs is None:
                    n_obs = quad_res.get('n_obs', 0)

                results_data.append({
                    'display_name': display_name,
                    'alpha_coeff': alpha_coeff,
                    'alpha_se': alpha_se,
                    'alpha_tstat': alpha_tstat,
                    'alpha_sq_coeff': alpha_sq_coeff,
                    'alpha_sq_se': alpha_sq_se,
                    'alpha_sq_tstat': alpha_sq_tstat,
                    'lookback_x_alpha_coeff': lookback_x_alpha_coeff,
                    'lookback_x_alpha_se': lookback_x_alpha_se,
                    'lookback_x_alpha_tstat': lookback_x_alpha_tstat,
                    'markets_x_alpha_coeff': markets_x_alpha_coeff,
                    'markets_x_alpha_se': markets_x_alpha_se,
                    'markets_x_alpha_tstat': markets_x_alpha_tstat,
                    'firms_per_market_x_alpha_coeff': firms_per_market_x_alpha_coeff,
                    'firms_per_market_x_alpha_se': firms_per_market_x_alpha_se,
                    'firms_per_market_x_alpha_tstat': firms_per_market_x_alpha_tstat,
                    'r2': r2
                })
                r2_values.append(r2)
            else:
                # Missing metric - fill with empty values
                results_data.append({
                    'display_name': display_name,
                    'alpha_coeff': np.nan,
                    'alpha_se': np.nan,
                    'alpha_tstat': np.nan,
                    'alpha_sq_coeff': np.nan,
                    'alpha_sq_se': np.nan,
                    'alpha_sq_tstat': np.nan,
                    'lookback_x_alpha_coeff': np.nan,
                    'lookback_x_alpha_se': np.nan,
                    'lookback_x_alpha_tstat': np.nan,
                    'markets_x_alpha_coeff': np.nan,
                    'markets_x_alpha_se': np.nan,
                    'markets_x_alpha_tstat': np.nan,
                    'firms_per_market_x_alpha_coeff': np.nan,
                    'firms_per_market_x_alpha_se': np.nan,
                    'firms_per_market_x_alpha_tstat': np.nan,
                    'r2': np.nan
                })
                r2_values.append(np.nan)

        # Build LaTeX table string
        latex = []
        latex.append(r'\begin{table}')
        latex.append(r'    \begin{tabular}{lcccc}')
        latex.append(r'        \hline\hline \\')
        latex.append(r'        [-1.8ex]')
        latex.append('')

        # Column headers (1-4)
        col_numbers = ' & '.join([f'({i+1})' for i in range(4)])
        latex.append(f'        & {col_numbers} \\\\')
        latex.append('')

        # Column names
        col_names = ' & '.join([r['display_name'] for r in results_data])
        latex.append(f'        & {col_names} \\\\')
        latex.append('')
        latex.append(r'        \hline')
        latex.append(r'        \\[-1.8ex]')
        latex.append('')

        # Alpha coefficient row
        alpha_coeffs = []
        for r in results_data:
            if not np.isnan(r['alpha_coeff']):
                stars = add_stars(r['alpha_tstat'])
                alpha_coeffs.append(f"{r['alpha_coeff']:.2f}{stars}")
            else:
                alpha_coeffs.append('---')

        alpha_row = ' & '.join(alpha_coeffs)
        latex.append(f'        $\\alpha$ & {alpha_row} \\\\')
        latex.append('')

        # Alpha standard errors row (in parentheses)
        alpha_ses = []
        for r in results_data:
            if not np.isnan(r['alpha_se']):
                alpha_ses.append(f"({r['alpha_se']:.2f})")
            else:
                alpha_ses.append('---')

        alpha_se_row = ' & '.join(alpha_ses)
        latex.append(f'        & {alpha_se_row} \\\\')
        latex.append('')

        # Alpha squared coefficient row
        alpha_sq_coeffs = []
        for r in results_data:
            if not np.isnan(r['alpha_sq_coeff']):
                stars = add_stars(r['alpha_sq_tstat'])
                alpha_sq_coeffs.append(f"{r['alpha_sq_coeff']:.2f}{stars}")
            else:
                alpha_sq_coeffs.append('---')

        alpha_sq_row = ' & '.join(alpha_sq_coeffs)
        latex.append(f'        $\\alpha^2$ & {alpha_sq_row} \\\\')
        latex.append('')

        # Alpha squared standard errors row (in parentheses)
        alpha_sq_ses = []
        for r in results_data:
            if not np.isnan(r['alpha_sq_se']):
                alpha_sq_ses.append(f"({r['alpha_sq_se']:.2f})")
            else:
                alpha_sq_ses.append('---')

        alpha_sq_se_row = ' & '.join(alpha_sq_ses)
        latex.append(f'        & {alpha_sq_se_row} \\\\')
        latex.append('')

        # Lookback x Alpha interaction coefficient row
        lookback_x_alpha_coeffs = []
        for r in results_data:
            if not np.isnan(r['lookback_x_alpha_coeff']):
                stars = add_stars(r['lookback_x_alpha_tstat'])
                lookback_x_alpha_coeffs.append(f"{r['lookback_x_alpha_coeff']:.4f}{stars}")
            else:
                lookback_x_alpha_coeffs.append('---')

        lookback_x_alpha_row = ' & '.join(lookback_x_alpha_coeffs)
        latex.append(f'        Lookback $\\times$ $\\alpha$ & {lookback_x_alpha_row} \\\\')
        latex.append('')

        # Lookback x Alpha standard errors row
        lookback_x_alpha_ses = []
        for r in results_data:
            if not np.isnan(r['lookback_x_alpha_se']):
                lookback_x_alpha_ses.append(f"({r['lookback_x_alpha_se']:.4f})")
            else:
                lookback_x_alpha_ses.append('---')

        lookback_x_alpha_se_row = ' & '.join(lookback_x_alpha_ses)
        latex.append(f'        & {lookback_x_alpha_se_row} \\\\')
        latex.append('')

        # Markets x Alpha interaction coefficient row
        markets_x_alpha_coeffs = []
        for r in results_data:
            if not np.isnan(r['markets_x_alpha_coeff']):
                stars = add_stars(r['markets_x_alpha_tstat'])
                markets_x_alpha_coeffs.append(f"{r['markets_x_alpha_coeff']:.4f}{stars}")
            else:
                markets_x_alpha_coeffs.append('---')

        markets_x_alpha_row = ' & '.join(markets_x_alpha_coeffs)
        latex.append(f'        Markets $\\times$ $\\alpha$ & {markets_x_alpha_row} \\\\')
        latex.append('')

        # Markets x Alpha standard errors row
        markets_x_alpha_ses = []
        for r in results_data:
            if not np.isnan(r['markets_x_alpha_se']):
                markets_x_alpha_ses.append(f"({r['markets_x_alpha_se']:.4f})")
            else:
                markets_x_alpha_ses.append('---')

        markets_x_alpha_se_row = ' & '.join(markets_x_alpha_ses)
        latex.append(f'        & {markets_x_alpha_se_row} \\\\')
        latex.append('')

        # Firms per Market x Alpha interaction coefficient row
        firms_x_alpha_coeffs = []
        for r in results_data:
            if not np.isnan(r['firms_per_market_x_alpha_coeff']):
                stars = add_stars(r['firms_per_market_x_alpha_tstat'])
                firms_x_alpha_coeffs.append(f"{r['firms_per_market_x_alpha_coeff']:.4f}{stars}")
            else:
                firms_x_alpha_coeffs.append('---')

        firms_x_alpha_row = ' & '.join(firms_x_alpha_coeffs)
        latex.append(f'        Firms/Market $\\times$ $\\alpha$ & {firms_x_alpha_row} \\\\')
        latex.append('')

        # Firms per Market x Alpha standard errors row
        firms_x_alpha_ses = []
        for r in results_data:
            if not np.isnan(r['firms_per_market_x_alpha_se']):
                firms_x_alpha_ses.append(f"({r['firms_per_market_x_alpha_se']:.4f})")
            else:
                firms_x_alpha_ses.append('---')

        firms_x_alpha_se_row = ' & '.join(firms_x_alpha_ses)
        latex.append(f'        & {firms_x_alpha_se_row} \\\\')
        latex.append('')

        # Separator
        latex.append(r'        \hline')
        latex.append(r'        \\[-1.8ex]')

        # Observations row
        if n_obs is not None:
            obs_row = ' & '.join([str(n_obs) for _ in range(4)])
            latex.append(f'        Observations & {obs_row} \\\\')

        # R² row
        r2_row = []
        for r2 in r2_values:
            if not np.isnan(r2):
                r2_row.append(f"{r2:.2f}")
            else:
                r2_row.append('---')
        r2_row_str = ' & '.join(r2_row)
        latex.append(f'        Adjusted R$^{{2}}$ & {r2_row_str} \\\\')

        # Notes
        latex.append(r'        \multicolumn{5}{l}{\footnotesize \textit{Note:} Standard errors in parentheses.} \\')
        latex.append(r'        \multicolumn{5}{l}{\footnotesize $^{*}$p$<$0.1; $^{**}$p$<$0.05; $^{***}$p$<$0.01} \\')
        latex.append('        ')
        latex.append(r'    \end{tabular}')
        latex.append('')
        latex.append(r'    \caption{Scalar metrics panel regression results showing how pooling affects conglomerate size and count}')
        latex.append(r'\end{table}')

        return '\n'.join(latex)

    def run_comprehensive_analysis(self):
        """Run sensitivity analysis across all parametrizations and all metrics"""
        print("Starting comprehensive analysis...")
        all_data = self.load_all_parametrizations()
        
        if not all_data:
            print("No data loaded!")
            return None
        
        results = {
            'parametrizations': {},
            'cross_parametrization': {
                'by_cost_function': {},
                'aggregate_stats': {}
            }
        }
        
        # Analyze each parametrization individually
        print(f"Analyzing {len(all_data)} parametrizations...")
        
        for i, (param_name, param_data) in enumerate(all_data.items()):
            print(f"[{i+1}/{len(all_data)}] Processing: {param_name}")
            
            param_results = {
                'cost_type': param_data['cost_type'],
                'hyperparameters': param_data['hyperparameters'],
                'metrics': {}
            }
            
            # Standard metrics sensitivity analysis
            for metric_name in ['gini_quantiles_avg', 'market_share_quantiles_avg', 'mean_members_avg', 'num_cong_avg', 'mergers_per_period_avg', 'exits_per_period_avg']:
                temporal_results, _ = self.fit_alpha_sensitivity_models(param_data, metric_name)
                if temporal_results:
                    param_results['metrics'][metric_name] = temporal_results
            
            # Polynomial estimates sensitivity
            poly_results = self.analyze_polynomial_sensitivity(param_data)
            if poly_results:
                param_results['polynomial_estimates'] = poly_results
            
            # Mobility sensitivity
            mobility_results = self.analyze_mobility_sensitivity(param_data)
            if mobility_results:
                param_results['mobility'] = mobility_results
                # Also add to all_data so controlled panel analysis can access it
                all_data[param_name]['mobility'] = mobility_results

            results['parametrizations'][param_name] = param_results
        
        # Aggregate by cost function type
        self._aggregate_by_cost_function(results)
        
        # Add cross-parametrization panel analysis with controls
        print("\nRunning robustness analysis with hyperparameter controls...")
        controlled_results = self.run_cross_parametrization_panel_analysis(all_data, control_hyperparams=True)
        if controlled_results:
            results['cross_parametrization']['controlled_panel'] = controlled_results
        
        # Add controlled per-timestep analysis within cost functions
        print("\nRunning controlled temporal analysis within cost functions...")
        controlled_temporal = self.run_controlled_temporal_analysis(all_data, control_hyperparams=True)
        if controlled_temporal:
            results['cross_parametrization']['controlled_temporal'] = controlled_temporal
        
        print("Analysis completed!")
        return results, all_data
    
    def _aggregate_by_cost_function(self, results):
        """Aggregate temporal results by cost function type for cross-parametrization analysis"""
        cost_aggregates = {}
        
        for param_name, param_results in results['parametrizations'].items():
            cost_type = param_results['cost_type']
            
            if cost_type not in cost_aggregates:
                cost_aggregates[cost_type] = {
                    'parametrizations': [],
                    'temporal_coefficients': {},
                    'panel_coefficients': {}
                }
            
            cost_aggregates[cost_type]['parametrizations'].append(param_name)
            
            # Collect temporal and panel coefficients for aggregation
            for metric_name, metric_data in param_results.get('metrics', {}).items():
                if isinstance(metric_data, dict):
                    # Handle quantile-based metrics (multiple quantiles per metric)
                    if any(q in metric_data for q in ['10th', '25th', '50th', '75th', '90th', '99th', '100th']):
                        for quantile_name, quantile_data in metric_data.items():
                            if isinstance(quantile_data, dict):
                                # Handle temporal coefficients
                                if 'temporal' in quantile_data:
                                    temporal_data = quantile_data['temporal']
                                    for coeff_name in ['linear_beta1', 'quadratic_beta1', 'quadratic_beta2']:
                                        if coeff_name in temporal_data:
                                            key = f"{metric_name}_{quantile_name}_temporal_{coeff_name}"
                                            if key not in cost_aggregates[cost_type]['temporal_coefficients']:
                                                cost_aggregates[cost_type]['temporal_coefficients'][key] = []
                                            cost_aggregates[cost_type]['temporal_coefficients'][key].append(
                                                temporal_data[coeff_name]
                                            )
                                
                                # Handle panel coefficients
                                if 'panel' in quantile_data:
                                    panel_data = quantile_data['panel']
                                    for coeff_name in ['linear_beta1', 'quadratic_beta1', 'quadratic_beta2']:
                                        if coeff_name in panel_data:
                                            key = f"{metric_name}_{quantile_name}_panel_{coeff_name}"
                                            if key not in cost_aggregates[cost_type]['panel_coefficients']:
                                                cost_aggregates[cost_type]['panel_coefficients'][key] = []
                                            cost_aggregates[cost_type]['panel_coefficients'][key].append(
                                                panel_data[coeff_name]
                                            )
                    else:
                        # Handle scalar metrics with temporal/panel structure
                        if 'temporal' in metric_data:
                            temporal_data = metric_data['temporal']
                            for coeff_name in ['linear_beta1', 'quadratic_beta1', 'quadratic_beta2']:
                                if coeff_name in temporal_data:
                                    key = f"{metric_name}_temporal_{coeff_name}"
                                    if key not in cost_aggregates[cost_type]['temporal_coefficients']:
                                        cost_aggregates[cost_type]['temporal_coefficients'][key] = []
                                    cost_aggregates[cost_type]['temporal_coefficients'][key].append(
                                        temporal_data[coeff_name]
                                    )
                        
                        if 'panel' in metric_data:
                            panel_data = metric_data['panel']
                            for coeff_name in ['linear_beta1', 'quadratic_beta1', 'quadratic_beta2']:
                                if coeff_name in panel_data:
                                    key = f"{metric_name}_panel_{coeff_name}"
                                    if key not in cost_aggregates[cost_type]['panel_coefficients']:
                                        cost_aggregates[cost_type]['panel_coefficients'][key] = []
                                    cost_aggregates[cost_type]['panel_coefficients'][key].append(
                                        panel_data[coeff_name]
                                    )
        
        # Calculate means and standard deviations across parametrizations
        for cost_type, cost_data in cost_aggregates.items():
            cost_data['temporal_stats'] = {}
            cost_data['panel_stats'] = {}

            # Process temporal coefficients (arrays)
            for key, coefficient_arrays in cost_data['temporal_coefficients'].items():
                if coefficient_arrays:
                    # Stack arrays and compute statistics across parametrizations
                    stacked = np.array(coefficient_arrays)  # Shape: (n_parametrizations, n_timesteps)

                    # Filter out NaN values for mean/std calculation
                    valid_mask = ~np.isnan(stacked)
                    n_valid_per_timestep = np.sum(valid_mask, axis=0)

                    # Calculate mean and std, handling NaN values
                    mean_series = np.nanmean(stacked, axis=0)
                    std_series = np.nanstd(stacked, axis=0)

                    cost_data['temporal_stats'][key] = {
                        'mean': mean_series,
                        'std': std_series,
                        'n_valid': n_valid_per_timestep,
                        'total_parametrizations': stacked.shape[0]
                    }

            # Process panel coefficients (scalars)
            for key, coefficient_scalars in cost_data['panel_coefficients'].items():
                if coefficient_scalars:
                    # Panel coefficients are scalars, just compute mean and std
                    scalars = np.array(coefficient_scalars)  # Shape: (n_parametrizations,)

                    # Filter out NaN values
                    valid_mask = ~np.isnan(scalars)
                    n_valid = np.sum(valid_mask)

                    # Calculate mean and std
                    mean_value = np.nanmean(scalars)
                    std_value = np.nanstd(scalars)

                    cost_data['panel_stats'][key] = {
                        'mean': mean_value,
                        'std': std_value,
                        'n_valid': n_valid,
                        'total_parametrizations': len(scalars)
                    }

        results['cross_parametrization']['by_cost_function'] = cost_aggregates
    
    def visualize_panel_coefficients(self, results, metric_name='gini_quantiles_avg', 
                                   coefficient='linear_beta1', save_path=None):
        """Visualize panel coefficients across cost functions with error bars"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        cost_functions = []
        means = []
        stds = []
        colors = []
        
        cost_data = results['cross_parametrization']['by_cost_function']
        key = f"{metric_name}_{coefficient}"
        
        for cost_type, data in cost_data.items():
            if key in data.get('panel_stats', {}):
                cost_functions.append(cost_type)
                stats = data['panel_stats'][key]
                means.append(stats['mean'])
                stds.append(stats['std'])
                colors.append(self.cost_function_colors.get(cost_type, '#333333'))
        
        if not means:
            print(f"No data available for {metric_name} {coefficient}")
            return fig, ax
        
        # Create bar plot with error bars
        bars = ax.bar(cost_functions, means, yerr=stds, color=colors, 
                     alpha=0.7, capsize=5, edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Cost Function Type')
        ax.set_ylabel('Coefficient')
        # No title or grid to match counterfactuals.py style
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std,
                   f'{mean:.4f}±{std:.4f}',
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig, ax
    
    def visualize_time_varying_coefficients(self, results, metric_name='gini_quantiles_avg',
                                          coefficient='linear_beta1', save_path=None):
        """Visualize time-varying coefficients with confidence bands"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        cost_data = results['cross_parametrization']['by_cost_function']
        key = f"{metric_name}_{coefficient}"
        
        for cost_type, data in cost_data.items():
            if key in data.get('per_timestep_stats', {}):
                stats = data['per_timestep_stats'][key]
                mean_coeff = stats['mean']
                std_coeff = stats['std']
                timesteps = np.arange(len(mean_coeff))
                
                color = self.cost_function_colors.get(cost_type, '#333333')
                
                # Plot mean line
                ax.plot(timesteps, mean_coeff, color=color, label=cost_type, 
                       linewidth=2)
                
                # Add confidence band
                ax.fill_between(timesteps, 
                              mean_coeff - std_coeff, 
                              mean_coeff + std_coeff,
                              color=color, alpha=0.2)
        
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Coefficient')
        ax.legend()
        # No title or grid to match counterfactuals.py style
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig, ax
    
    def visualize_temporal_alpha_impact(self, results, metric_name='gini_quantiles_avg', 
                                       quantile='50th', coefficient='linear_beta1', save_path=None):
        """Visualize temporal impact of alpha with confidence bands across cost functions"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        cost_data = results['cross_parametrization']['by_cost_function']
        
        # For quantile-based metrics, construct the key
        if quantile:
            key = f"{metric_name}_{quantile}_temporal_{coefficient}"
            title_suffix = f"{quantile} percentile"
        else:
            key = f"{metric_name}_temporal_{coefficient}"
            title_suffix = ""
        
        for cost_type, data in cost_data.items():
            if key in data.get('temporal_stats', {}):
                stats = data['temporal_stats'][key]
                mean_coeff = stats['mean']
                std_coeff = stats['std']
                timesteps = np.arange(len(mean_coeff))
                
                color = self.cost_function_colors.get(cost_type, '#333333')
                
                # Plot mean line
                ax.plot(timesteps, mean_coeff, color=color, label=cost_type, 
                       linewidth=2, alpha=0.8)
                
                # Add confidence band (±1 std)
                ax.fill_between(timesteps, 
                              mean_coeff - std_coeff, 
                              mean_coeff + std_coeff,
                              color=color, alpha=0.2)
        
        ax.set_xlabel('Timestep')
        ax.set_ylabel(f'Impact of α on {metric_name.replace("_", " ").title()}')
        
        ax.legend()
        # No title or grid to match counterfactuals.py style
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig, ax

    def visualize_metric_quantiles_faceted(self, results, metric_name='gini_quantiles_avg', 
                                          coefficient='linear_beta1', save_path=None):
        """Create faceted plot showing all quantiles for a metric in subplots"""
        quantiles = ['10th', '25th', '50th', '75th', '90th', '99th', '100th']
        
        # Create subplot grid (2 rows x 4 columns) with shared x-axis
        fig, axes_grid = plt.subplots(2, 4, figsize=(24, 12), sharex=True)
        axes = axes_grid.flatten()
        
        # Remove the last (empty) subplot
        axes[7].remove()
        
        cost_data = results['cross_parametrization']['by_cost_function']
        
        for i, quantile in enumerate(quantiles):
            ax = axes[i]
            key = f"{metric_name}_{quantile}_temporal_{coefficient}"
            
            # Plot each cost function
            for cost_type, data in cost_data.items():
                if key in data.get('temporal_stats', {}):
                    stats = data['temporal_stats'][key]
                    mean_coeff = stats['mean']
                    std_coeff = stats['std']
                    timesteps = np.arange(len(mean_coeff))
                    
                    color = self.cost_function_colors.get(cost_type, '#333333')
                    
                    # Plot mean line
                    ax.plot(timesteps, mean_coeff, color=color, label=cost_type, 
                           linewidth=1.5, alpha=0.8)
                    
                    # Add confidence band (±1 std)
                    ax.fill_between(timesteps, 
                                  mean_coeff - std_coeff, 
                                  mean_coeff + std_coeff,
                                  color=color, alpha=0.15)
            
            # No title or grid to match counterfactuals.py style
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=0.8)
            
            # Set labels only for edge subplots with larger font size
            if i >= 4:  # Bottom row only
                ax.set_xlabel('Timestep', fontsize=42)
            if i % 4 == 0:  # Left column
                ax.set_ylabel('Impact of α', fontsize=42)
            
            # Increase tick label font size
            ax.tick_params(axis='both', which='major', labelsize=36)
        
        # Add legend to the empty space (position 8)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:  # Only create legend if we have plot handles
            # Create legend in the empty bottom-right position
            legend_ax = plt.subplot(2, 4, 8)
            legend_ax.legend(handles, labels, loc='center', fontsize=38)
            legend_ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig, axes

    def visualize_metric_quantiles_single(self, results, metric_name='gini_quantiles_avg', 
                                         coefficient='linear_beta1', save_path=None):
        """Create single plot with all quantiles as separate lines per cost function"""
        quantiles = ['50th', '75th', '90th', '99th']  # Focus on key quantiles
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        cost_data = results['cross_parametrization']['by_cost_function']
        
        for cost_type, data in cost_data.items():
            base_color = self.cost_function_colors.get(cost_type, '#333333')
            
            for i, quantile in enumerate(quantiles):
                key = f"{metric_name}_{quantile}_temporal_{coefficient}"
                
                if key in data.get('temporal_stats', {}):
                    stats = data['temporal_stats'][key]
                    mean_coeff = stats['mean']
                    std_coeff = stats['std']
                    timesteps = np.arange(len(mean_coeff))
                    
                    # Create different line styles for quantiles
                    line_styles = ['-', '--', '-.', ':']
                    line_style = line_styles[i % len(line_styles)]
                    
                    # Plot mean line
                    label = f"{cost_type} ({quantile})"
                    ax.plot(timesteps, mean_coeff, color=base_color, linestyle=line_style,
                           label=label, linewidth=2, alpha=0.8)
                    
                    # Add subtle confidence band
                    ax.fill_between(timesteps, 
                                  mean_coeff - std_coeff, 
                                  mean_coeff + std_coeff,
                                  color=base_color, alpha=0.1)
        
        ax.set_xlabel('Timestep', fontsize=12)
        ax.set_ylabel(f'Impact of α on {metric_name.replace("_", " ").title()}', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        # No title or grid to match counterfactuals.py style
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig, ax

    def visualize_panel_coefficients_by_quantile(self, results, metric_name='gini_quantiles_avg',
                                                coefficient='linear_beta1', save_path=None):
        """Create faceted plot showing panel coefficients across quantiles and cost functions

        Each quantile gets its own subplot with independent y-axis scaling.
        Shows both β₁ (linear) and β₂ (quadratic) as grouped bars by cost function.
        """
        # Use only 5 key quantiles like the temporal plot
        quantiles = ['10th', '50th', '90th', '99th', '100th']

        cost_data = results['cross_parametrization']['by_cost_function']
        cost_types = sorted(cost_data.keys())

        # Create vertically stacked subplots - one per quantile
        fig, axes = plt.subplots(len(quantiles), 1, figsize=(12, 15), sharex=True)

        for q_idx, quantile in enumerate(quantiles):
            ax = axes[q_idx]

            # Collect β₁ (linear) and β₂ (quadratic) for each cost function
            beta1_means = []
            beta2_means = []
            labels = []

            for cost_type in cost_types:
                # Get β₁ (linear)
                key_beta1 = f"{metric_name}_{quantile}_panel_quadratic_beta1"
                # Get β₂ (quadratic)
                key_beta2 = f"{metric_name}_{quantile}_panel_quadratic_beta2"

                if (key_beta1 in cost_data[cost_type].get('panel_stats', {}) and
                    key_beta2 in cost_data[cost_type].get('panel_stats', {})):

                    stats_beta1 = cost_data[cost_type]['panel_stats'][key_beta1]
                    stats_beta2 = cost_data[cost_type]['panel_stats'][key_beta2]

                    beta1_means.append(float(stats_beta1['mean']))
                    beta2_means.append(float(stats_beta2['mean']))
                    labels.append(cost_type)

            if beta1_means:
                # Create grouped bar plot
                x = np.arange(len(labels))  # Cost function positions
                width = 0.35  # Bar width
                gap = 0.1     # Gap between bars within a group

                # Plot β₁ bars
                bars1 = ax.bar(x - width/2 - gap/2, beta1_means, width,
                             label='β₁ (linear)', alpha=0.7, color='#0072B2')

                # Plot β₂ bars
                bars2 = ax.bar(x + width/2 + gap/2, beta2_means, width,
                             label='β₂ (quadratic)', alpha=0.7, color='#D55E00', hatch='//')

                # Add value labels on bars
                for bar, val in zip(bars1, beta1_means):
                    height = bar.get_height()
                    y_pos = height + 0.005 * (ax.get_ylim()[1] - ax.get_ylim()[0]) if height > 0 else height - 0.005 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                    va = 'bottom' if height > 0 else 'top'
                    ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                           f'{val:.3f}', ha='center', va=va, fontsize=9)

                for bar, val in zip(bars2, beta2_means):
                    height = bar.get_height()
                    y_pos = height + 0.005 * (ax.get_ylim()[1] - ax.get_ylim()[0]) if height > 0 else height - 0.005 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                    va = 'bottom' if height > 0 else 'top'
                    ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                           f'{val:.3f}', ha='center', va=va, fontsize=9)

                # Format subplot
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=0.8)
                ax.set_xticks(x)
                ax.set_xticklabels(labels, fontsize=11)
                ax.set_ylabel('Coefficient', fontsize=11)
                ax.tick_params(axis='y', which='major', labelsize=10)

                # Add quantile label
                ax.text(0.02, 0.95, f'Quantile: {quantile}',
                       transform=ax.transAxes, fontsize=11, fontweight='bold',
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                # Add legend to first subplot only
                if q_idx == 0:
                    ax.legend(fontsize=10, loc='upper right')

        # Add x-label only to bottom plot
        axes[-1].set_xlabel('Cost Function', fontsize=12)

        plt.suptitle(f'{metric_name.replace("_", " ").title()}\nPanel Estimates by Quantile (β₁ and β₂)',
                    fontsize=13, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig, axes

    def visualize_controlled_panel_results(self, results, save_path=None):
        """Create separate plots for gini and market_share showing quadratic controlled panel regressions

        Each metric gets its own figure showing the quadratic model:
        y = β₀ + β₁·α + β₂·α² + controls

        Displays grouped bars for β₁ (solid) and β₂ (hatched)

        Returns a dictionary with separate figures for each metric
        """
        if 'controlled_panel' not in results.get('cross_parametrization', {}):
            print("No controlled panel results found!")
            return {}

        controlled_data = results['cross_parametrization']['controlled_panel']

        metrics = ['gini_quantiles_avg', 'market_share_quantiles_avg']
        figures = {}

        for metric_name in metrics:
            if metric_name not in controlled_data:
                continue

            # Create separate figure for this metric with more width for bar spacing
            fig, ax = plt.subplots(1, 1, figsize=(16, 16))

            quantiles = []
            alpha_coeffs = []
            alpha_ses = []
            alpha_tstats = []
            alpha_sq_coeffs = []
            alpha_sq_ses = []
            alpha_sq_tstats = []

            # Only use quadratic model results
            for quantile, models in controlled_data[metric_name].items():
                quad_results = models['quadratic']

                quantiles.append(quantile)
                alpha_coeffs.append(quad_results['alpha_coeff'])
                alpha_ses.append(quad_results['alpha_se'])
                alpha_tstats.append(quad_results['alpha_tstat'])
                alpha_sq_coeffs.append(quad_results['alpha_sq_coeff'])
                alpha_sq_ses.append(quad_results['alpha_sq_se'])
                alpha_sq_tstats.append(quad_results['alpha_sq_tstat'])

            # Increase spacing between quantile groups
            x = np.arange(len(quantiles)) * 2.5  # 2.5x spacing between groups

            # Grouped bars for β₁ and β₂ with gap between them
            width = 0.35  # Bar width
            gap = 0.5     # Gap between bars within a group
            colors_beta1 = ['green' if abs(t) > 2.0 else 'orange' for t in alpha_tstats]
            colors_beta2 = ['darkgreen' if abs(t) > 2.0 else 'darkorange' for t in alpha_sq_tstats]

            bars1 = ax.bar(x - width/2 - gap/2, alpha_coeffs, width, color=colors_beta1,
                          alpha=0.7, label='β₁ (linear)')
            bars2 = ax.bar(x + width/2 + gap/2, alpha_sq_coeffs, width, color=colors_beta2,
                          alpha=0.7, label='β₂ (quadratic)', hatch='//')

            # Add β estimates with SE above/below bars
            for j, (bar1, bar2, beta1, se1, beta2, se2) in enumerate(zip(bars1, bars2, alpha_coeffs, alpha_ses, alpha_sq_coeffs, alpha_sq_ses)):
                # β₁ estimate with SE
                h1 = bar1.get_height()
                y_pos1 = h1 + 0.01 if h1 > 0 else h1 - 0.01  # Increased spacing
                va1 = 'bottom' if h1 > 0 else 'top'
                ax.text(bar1.get_x() + bar1.get_width()/2., y_pos1,
                       f'{beta1:.4f}\n({se1:.4f})', ha='center', va=va1, fontsize=18)

                # β₂ estimate with SE
                h2 = bar2.get_height()
                y_pos2 = h2 + 0.01 if h2 > 0 else h2 - 0.01  # Increased spacing
                va2 = 'bottom' if h2 > 0 else 'top'
                ax.text(bar2.get_x() + bar2.get_width()/2., y_pos2,
                       f'{beta2:.4f}\n({se2:.4f})', ha='center', va=va2, fontsize=18)

            # Format axes
            ax.set_xticks(x)
            ax.set_xticklabels(quantiles, rotation=45, fontsize=16)
            ax.set_xlabel('Quantile', fontsize=20)
            ax.set_ylabel('Controlled Coefficient', fontsize=20)
            ax.set_title(f'{metric_name.replace("_", " ").title()}\n(quadratic model with controls)',
                        fontsize=20)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.legend(fontsize=16)
            ax.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            figures[metric_name] = fig

        return figures

    def visualize_mobility_controlled_panel(self, results, save_path=None):
        """Create plot of controlled panel results for mobility metrics"""
        if 'controlled_panel' not in results.get('cross_parametrization', {}):
            print("No controlled panel results found!")
            return None, None

        controlled_data = results['cross_parametrization']['controlled_panel']

        # Check if we have mobility metrics
        mobility_metrics = []
        for metric_name in ['rank_ranges', 'rank_std']:
            if metric_name in controlled_data:
                mobility_metrics.append(metric_name)

        if not mobility_metrics:
            print("No mobility metrics found in controlled panel results!")
            return None, None

        # Create figure with one subplot per mobility metric
        fig, axes = plt.subplots(1, len(mobility_metrics), figsize=(8 * len(mobility_metrics), 16))
        if len(mobility_metrics) == 1:
            axes = [axes]

        for i, metric_name in enumerate(mobility_metrics):
            ax = axes[i]

            # Mobility metrics have only 'overall' quantile
            models = controlled_data[metric_name].get('overall', {})

            if models:
                # Access quadratic model results
                quad_results = models['quadratic']

                beta1 = quad_results['alpha_coeff']
                beta1_se = quad_results['alpha_se']
                beta1_tstat = quad_results['alpha_tstat']
                beta2 = quad_results['alpha_sq_coeff']
                beta2_se = quad_results['alpha_sq_se']
                beta2_tstat = quad_results['alpha_sq_tstat']

                # Create grouped bars for β₁ and β₂ with gap between them
                width = 0.35
                gap = 0.45     # Gap between bars
                x = np.array([0])

                # β₁ bar
                color1 = 'green' if abs(beta1_tstat) > 2.0 else 'orange'
                bar1 = ax.bar(x - width/2 - gap/2, [beta1], width, color=color1, alpha=0.7, label='β₁ (linear)')

                # β₂ bar (hatched)
                color2 = 'darkgreen' if abs(beta2_tstat) > 2.0 else 'darkorange'
                bar2 = ax.bar(x + width/2 + gap/2, [beta2], width, color=color2, alpha=0.7, label='β₂ (quadratic)', hatch='//')

                # Add β estimates with SE above/below bars
                # For β₁
                h1 = beta1
                y_pos1 = h1 + 0.005 if h1 > 0 else h1 - 0.005
                va1 = 'bottom' if h1 > 0 else 'top'
                ax.text(x[0] - width/2 - gap/2, y_pos1,
                       f'{beta1:.4f}\n({beta1_se:.4f})', ha='center', va=va1, fontsize=14)

                # For β₂
                h2 = beta2
                y_pos2 = h2 + 0.005 if h2 > 0 else h2 - 0.005
                va2 = 'bottom' if h2 > 0 else 'top'
                ax.text(x[0] + width/2 + gap/2, y_pos2,
                       f'{beta2:.4f}\n({beta2_se:.4f})', ha='center', va=va2, fontsize=14)

                ax.set_xlim(-0.6, 0.6)
                ax.set_xticks([])
                ax.set_ylabel('Controlled Coefficient', fontsize=16)
                ax.legend(fontsize=14)

                # Format title
                metric_title = metric_name.replace('_', ' ').title()
                ax.set_title(f'{metric_title}\n(quadratic model with controls)', fontsize=14, weight='bold')
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, format='pdf', bbox_inches='tight')

        return fig, axes

    def visualize_controlled_panel_faceted(self, results, metric_name='gini_quantiles_avg', save_path=None):
        """Create faceted plot showing controlled panel estimates across quantiles

        Shows time-pooled regression coefficients β₁ and β₂ from:
        y = β₀ + β₁·α + β₂·α² + controls

        Faceted by quantile (vertically stacked), grouped by cost function.

        Args:
            results: Results dictionary containing controlled_panel data
            metric_name: Metric to visualize ('gini_quantiles_avg' or 'market_share_quantiles_avg')
            save_path: Optional path to save the figure

        Returns:
            (fig, axes): Matplotlib figure and axes objects
        """
        if 'controlled_panel' not in results.get('cross_parametrization', {}):
            print("No controlled panel results found!")
            return None, None

        controlled_panel = results['cross_parametrization']['controlled_panel']

        if metric_name not in controlled_panel:
            print(f"Metric {metric_name} not found in controlled panel results!")
            return None, None

        # Select quantiles to display (5 quantiles like the temporal plot)
        quantiles = ['10th', '50th', '90th', '99th', '100th']

        # Get cost function data from controlled_temporal to know which cost types we have
        # (controlled_panel pools across cost types, so we need to look elsewhere)
        if 'controlled_temporal' in results.get('cross_parametrization', {}):
            cost_types = list(results['cross_parametrization']['controlled_temporal'].keys())
        else:
            # Fallback: use default cost types
            cost_types = ['linear', 'quadratic', 'exponential', 'power_law']

        # Create subplot grid - stacked vertically (5 rows x 1 column)
        fig, axes = plt.subplots(len(quantiles), 1, figsize=(10, 12), sharex=True)

        from matplotlib.lines import Line2D

        for i, quantile in enumerate(quantiles):
            ax = axes[i]

            if quantile not in controlled_panel[metric_name]:
                # Skip if this quantile doesn't have data
                continue

            quantile_data = controlled_panel[metric_name][quantile]
            quad_results = quantile_data.get('quadratic', {})

            # Extract coefficients
            beta1 = quad_results.get('alpha_coeff', np.nan)
            beta1_se = quad_results.get('alpha_se', np.nan)
            beta1_tstat = quad_results.get('alpha_tstat', np.nan)
            beta2 = quad_results.get('alpha_sq_coeff', np.nan)
            beta2_se = quad_results.get('alpha_sq_se', np.nan)
            beta2_tstat = quad_results.get('alpha_sq_tstat', np.nan)

            # Create grouped bars for β₁ and β₂
            # Since panel results pool across cost types, we show single bars
            x = np.array([0, 1.5])  # Two positions: one for β₁, one for β₂
            heights = [beta1, beta2]
            errors = [beta1_se, beta2_se]
            colors = ['#0072B2', '#D55E00']  # Blue for β₁, orange for β₂
            labels = ['β₁ (linear)', 'β₂ (quadratic)']

            bars = ax.bar(x, heights, width=0.6, color=colors, alpha=0.7, yerr=errors, capsize=5)

            # Add value labels on bars
            for bar, val, se, tstat in zip(bars, heights, errors, [beta1_tstat, beta2_tstat]):
                if not np.isnan(val):
                    h = bar.get_height()
                    y_pos = h + se + 0.01 if h > 0 else h - se - 0.01
                    va = 'bottom' if h > 0 else 'top'

                    # Add significance stars
                    stars = ''
                    if abs(tstat) >= 2.576:
                        stars = '***'
                    elif abs(tstat) >= 1.96:
                        stars = '**'
                    elif abs(tstat) >= 1.645:
                        stars = '*'

                    ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                           f'{val:.3f}{stars}\n({se:.3f})',
                           ha='center', va=va, fontsize=10)

            # Format subplot
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=11)
            ax.set_ylabel('Coefficient', fontsize=11)
            ax.tick_params(axis='y', which='major', labelsize=10)

            # Add quantile label
            ax.text(0.02, 0.95, f'Quantile: {quantile}',
                   transform=ax.transAxes, fontsize=11, fontweight='bold',
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Add x-label only to bottom plot
        axes[-1].set_xlabel('Coefficient Type', fontsize=12)

        plt.suptitle(f'{metric_name.replace("_", " ").title()}\nControlled Panel Regression (pooled across time and cost functions)',
                    fontsize=13, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig, axes

    def visualize_controlled_temporal_faceted(self, results, metric_name='gini_quantiles_avg', save_path=None):
        """Create faceted plot showing controlled temporal evolution across quantiles

        Plots β₁ and β₂ from the quadratic model on the same plot:
        - Solid lines: β₁ (linear coefficient of α)
        - Dashed lines: β₂ (quadratic coefficient of α²)

        Both from quadratic model: y = β₀ + β₁·α + β₂·α² + controls
        """
        if 'controlled_temporal' not in results.get('cross_parametrization', {}):
            print("No controlled temporal results found!")
            return None, None

        controlled_temporal = results['cross_parametrization']['controlled_temporal']
        # Remove 25th and 75th quantiles, keep only 10th, 50th, 90th, 99th, 100th
        quantiles = ['10th', '50th', '90th', '99th', '100th']

        # Create subplot grid - stacked vertically (5 rows x 1 column)
        fig, axes = plt.subplots(5, 1, figsize=(12, 20), sharex=True)

        for i, quantile in enumerate(quantiles):
            ax = axes[i]
            plotted_any = False

            # Plot each cost function
            for cost_type, cost_data in controlled_temporal.items():
                if metric_name in cost_data:
                    metric_data = cost_data[metric_name]
                    if quantile in metric_data:
                        quantile_data = metric_data[quantile]

                        color = self.cost_function_colors.get(cost_type, '#333333')

                        # Plot quadratic model coefficients with confidence bands
                        if 'quadratic' in quantile_data:
                            quad_data = quantile_data['quadratic']
                            timesteps = quad_data['timesteps']

                            # β₁ (linear coefficient) - solid line with confidence band
                            alpha_coeff = quad_data['alpha_coeff']
                            alpha_se = quad_data['alpha_se']

                            ax.plot(timesteps, alpha_coeff, color=color, linestyle='-',
                                   linewidth=1.5, alpha=0.8)
                            # Add ±1 SE confidence band
                            ax.fill_between(timesteps, alpha_coeff - alpha_se, alpha_coeff + alpha_se,
                                          color=color, alpha=0.15, linewidth=0)
                            plotted_any = True

                            # β₂ (quadratic coefficient) - dashed line with confidence band
                            alpha_sq_coeff = quad_data['alpha_sq_coeff']
                            alpha_sq_se = quad_data['alpha_sq_se']

                            ax.plot(timesteps, alpha_sq_coeff, color=color, linestyle='--',
                                   linewidth=1.5, alpha=0.8)
                            # Add ±1 SE confidence band
                            ax.fill_between(timesteps, alpha_sq_coeff - alpha_sq_se, alpha_sq_coeff + alpha_sq_se,
                                          color=color, alpha=0.15, linewidth=0)

            if plotted_any:
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=0.8)

                # Add quantile label above the facet
                ax.text(0.02, 0.95, f'Quantile: {quantile}',
                       transform=ax.transAxes, fontsize=14, fontweight='bold',
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                # Only add x-label to bottom plot
                if i == len(quantiles) - 1:
                    ax.set_xlabel('Timestep', fontsize=14)

                # Add y-label to all plots
                ax.set_ylabel('Coefficient', fontsize=14)

                # Tick label font size
                ax.tick_params(axis='both', which='major', labelsize=12)

        # Add legend to the bottom plot
        from matplotlib.lines import Line2D

        # Get unique cost types
        cost_types = list(controlled_temporal.keys())
        legend_elements = []

        # Add cost function colors
        for cost_type in cost_types:
            color = self.cost_function_colors.get(cost_type, '#333333')
            legend_elements.append(Line2D([0], [0], color=color, linewidth=2, label=cost_type))

        # Add separator
        legend_elements.append(Line2D([0], [0], color='none'))

        # Add line style meanings
        legend_elements.append(Line2D([0], [0], color='gray', linestyle='-', linewidth=2, label='β₁ (linear)'))
        legend_elements.append(Line2D([0], [0], color='gray', linestyle='--', linewidth=2, label='β₂ (quadratic)'))

        # Place legend outside the bottom plot
        axes[-1].legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                       fontsize=12, ncol=3, frameon=True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig, axes

    def visualize_controlled_temporal_scalar(self, results, metric_name='mean_members_avg', save_path=None):
        """Create plot showing controlled temporal evolution for scalar metrics

        Plots β₁ and β₂ from the quadratic model on the same plot:
        - Solid lines: β₁ (linear coefficient of α)
        - Dashed lines: β₂ (quadratic coefficient of α²)

        Both from quadratic model: y = β₀ + β₁·α + β₂·α² + controls
        """
        if 'controlled_temporal' not in results.get('cross_parametrization', {}):
            print("No controlled temporal results found!")
            return None, None

        controlled_temporal = results['cross_parametrization']['controlled_temporal']

        fig, ax = plt.subplots(figsize=(12, 8))
        plotted_any = False

        # Plot each cost function
        for cost_type, cost_data in controlled_temporal.items():
            if metric_name in cost_data:
                metric_data = cost_data[metric_name]
                # Scalar metrics have 'overall' quantile
                if 'overall' in metric_data:
                    quantile_data = metric_data['overall']

                    color = self.cost_function_colors.get(cost_type, '#333333')

                    # Plot quadratic model coefficients with confidence bands
                    if 'quadratic' in quantile_data:
                        quad_data = quantile_data['quadratic']
                        timesteps = quad_data['timesteps']

                        # β₁ (linear coefficient) - solid line with confidence band
                        alpha_coeff = quad_data['alpha_coeff']
                        alpha_se = quad_data['alpha_se']

                        ax.plot(timesteps, alpha_coeff, color=color,
                               linewidth=2, alpha=0.8, linestyle='-')
                        # Add ±1 SE confidence band
                        ax.fill_between(timesteps, alpha_coeff - alpha_se, alpha_coeff + alpha_se,
                                      color=color, alpha=0.15, linewidth=0)
                        plotted_any = True

                        # β₂ (quadratic coefficient) - dashed line with confidence band
                        alpha_sq_coeff = quad_data['alpha_sq_coeff']
                        alpha_sq_se = quad_data['alpha_sq_se']

                        ax.plot(timesteps, alpha_sq_coeff, color=color,
                               linewidth=2, alpha=0.8, linestyle='--')
                        # Add ±1 SE confidence band
                        ax.fill_between(timesteps, alpha_sq_coeff - alpha_sq_se, alpha_sq_coeff + alpha_sq_se,
                                      color=color, alpha=0.15, linewidth=0)

        if plotted_any:
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=0.8)
            ax.set_xlabel('Timestep', fontsize=14)
            ax.set_ylabel('Controlled Coefficient', fontsize=14)

            # Create custom legend matching faceted plot style
            from matplotlib.lines import Line2D

            legend_elements = []

            # Add cost function colors
            for cost_type in controlled_temporal.keys():
                color = self.cost_function_colors.get(cost_type, '#333333')
                legend_elements.append(Line2D([0], [0], color=color, linewidth=2, label=cost_type))

            # Add separator
            legend_elements.append(Line2D([0], [0], color='none'))

            # Add line style meanings
            legend_elements.append(Line2D([0], [0], color='gray', linestyle='-', linewidth=2, label='β₁ (linear)'))
            legend_elements.append(Line2D([0], [0], color='gray', linestyle='--', linewidth=2, label='β₂ (quadratic)'))

            ax.legend(handles=legend_elements, fontsize=12)
            ax.tick_params(axis='both', which='major', labelsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig, ax

    def visualize_temporal_polynomial_estimates(self, all_data, save_path=None):
        """Visualize temporal polynomial estimates (β₁ and β₂) evolution over time

        Shows how polynomial coefficients from market_share ~ poly(size, 2) evolve
        over time for different α values, faceted by cost function.

        Aggregates across parametrizations by showing median with IQR bands.
        This is DESCRIPTIVE visualization showing patterns, not inferential statistics.

        Args:
            all_data: Dictionary of all parametrization data (from load_data())
            save_path: Optional path to save the figure

        Returns:
            (fig, axes): Matplotlib figure and axes objects
        """
        # Group data by cost function type
        cost_groups = {}
        for param_name, param_data in all_data.items():
            cost_type = param_data['cost_type']
            if cost_type not in cost_groups:
                cost_groups[cost_type] = []
            cost_groups[cost_type].append((param_name, param_data))

        # Create faceted plot - one row per cost function, one column per coefficient
        n_cost_types = len(cost_groups)
        fig, axes = plt.subplots(n_cost_types, 2, figsize=(16, 5 * n_cost_types))

        # Ensure axes is 2D even with single cost type
        if n_cost_types == 1:
            axes = axes.reshape(1, -1)

        plotted_any = False

        for row_idx, (cost_type, param_list) in enumerate(sorted(cost_groups.items())):
            ax_beta1 = axes[row_idx, 0]
            ax_beta2 = axes[row_idx, 1]

            # Collect all data for this cost function grouped by alpha
            # Structure: {alpha: {timestep: [values from different parametrizations]}}
            beta1_by_alpha_time = {}
            beta2_by_alpha_time = {}

            for param_name, param_data in param_list:
                shares = param_data['shares']

                # Get temporal polynomial estimates
                temporal_poly_avg = param_data['metrics'].get('temporal_poly_estimates_avg', {})

                if not temporal_poly_avg:
                    continue

                # Collect data for each alpha value
                for alpha in shares:
                    if alpha not in temporal_poly_avg:
                        continue

                    poly_avg = temporal_poly_avg[alpha]  # Shape: (n_timesteps, 3) - [β₂, β₁, β₀]

                    if poly_avg is None or not isinstance(poly_avg, np.ndarray):
                        continue

                    if alpha not in beta1_by_alpha_time:
                        beta1_by_alpha_time[alpha] = {}
                        beta2_by_alpha_time[alpha] = {}

                    # Add data for each timestep
                    for t in range(poly_avg.shape[0]):
                        if t not in beta1_by_alpha_time[alpha]:
                            beta1_by_alpha_time[alpha][t] = []
                            beta2_by_alpha_time[alpha][t] = []

                        beta1_by_alpha_time[alpha][t].append(poly_avg[t, 1])  # β₁ is index 1
                        beta2_by_alpha_time[alpha][t].append(poly_avg[t, 0])  # β₂ is index 0

            # Now plot aggregated data for each alpha
            alpha_sorted = sorted(beta1_by_alpha_time.keys())

            for alpha in alpha_sorted:
                # Get all timesteps for this alpha
                timesteps = sorted(beta1_by_alpha_time[alpha].keys())

                if len(timesteps) == 0:
                    continue

                # Compute median and IQR for β₁
                beta1_median = []
                beta1_lower = []
                beta1_upper = []

                for t in timesteps:
                    values = np.array(beta1_by_alpha_time[alpha][t])
                    if len(values) > 0:
                        beta1_median.append(np.median(values))
                        beta1_lower.append(np.percentile(values, 25))
                        beta1_upper.append(np.percentile(values, 75))

                # Compute median and IQR for β₂
                beta2_median = []
                beta2_lower = []
                beta2_upper = []

                for t in timesteps:
                    values = np.array(beta2_by_alpha_time[alpha][t])
                    if len(values) > 0:
                        beta2_median.append(np.median(values))
                        beta2_lower.append(np.percentile(values, 25))
                        beta2_upper.append(np.percentile(values, 75))

                if len(beta1_median) == 0:
                    continue

                # Color by alpha value (darker = higher alpha)
                color = plt.cm.viridis(alpha)

                # Plot β₁
                timesteps_arr = np.array(timesteps)
                ax_beta1.plot(timesteps_arr, beta1_median, color=color, linewidth=2,
                            label=f'α={alpha:.2f}')
                ax_beta1.fill_between(timesteps_arr, beta1_lower, beta1_upper,
                                    color=color, alpha=0.2, linewidth=0)

                # Plot β₂
                ax_beta2.plot(timesteps_arr, beta2_median, color=color, linewidth=2,
                            label=f'α={alpha:.2f}')
                ax_beta2.fill_between(timesteps_arr, beta2_lower, beta2_upper,
                                    color=color, alpha=0.2, linewidth=0)

                plotted_any = True

            # Formatting for β₁ subplot
            ax_beta1.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=0.8)
            ax_beta1.set_xlabel('Timestep', fontsize=12)
            ax_beta1.set_ylabel('β₁ (Linear Coefficient)', fontsize=12)
            ax_beta1.set_title(f'{cost_type} - β₁ (median ± IQR)', fontsize=14, fontweight='bold')
            ax_beta1.grid(True, alpha=0.3)

            # Formatting for β₂ subplot
            ax_beta2.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=0.8)
            ax_beta2.set_xlabel('Timestep', fontsize=12)
            ax_beta2.set_ylabel('β₂ (Quadratic Coefficient)', fontsize=12)
            ax_beta2.set_title(f'{cost_type} - β₂ (median ± IQR)', fontsize=14, fontweight='bold')
            ax_beta2.grid(True, alpha=0.3)

            # Add legend to the first row only (to avoid clutter)
            if row_idx == 0:
                ax_beta2.legend(fontsize=9, loc='best', ncol=2)

        if plotted_any:
            plt.suptitle('Temporal Polynomial Estimates: market_share ~ poly(size, 2) (Aggregated Across Parametrizations)',
                        fontsize=16, fontweight='bold', y=0.995)
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved temporal polynomial estimates plot: {save_path}")
        else:
            print("No temporal polynomial estimates data found to plot!")

        return fig, axes

    def visualize_panel_polynomial_estimates(self, all_data, save_path=None):
        """Visualize panel polynomial estimates (β₁ and β₂) across α values

        Line plots with α on x-axis, separate lines for each cost function,
        with confidence bands showing within-group variation (25th-75th percentiles).

        This is DESCRIPTIVE visualization showing patterns, not inferential statistics.

        Args:
            all_data: Dictionary of all parametrization data (from load_data())
            save_path: Optional path to save the figure

        Returns:
            (fig, axes): Matplotlib figure and axes objects
        """
        # Group data by cost function type
        cost_groups = {}
        for param_name, param_data in all_data.items():
            cost_type = param_data['cost_type']
            if cost_type not in cost_groups:
                cost_groups[cost_type] = []
            cost_groups[cost_type].append((param_name, param_data))

        # Create figure with two subplots (one for β₁, one for β₂)
        fig, (ax_beta1, ax_beta2) = plt.subplots(1, 2, figsize=(16, 6))

        plotted_any = False

        # Define colors for cost functions
        cost_colors = {
            'linear': '#1f77b4',
            'quadratic': '#ff7f0e',
            'power_law': '#2ca02c',
            'exponential': '#d62728'
        }

        # Prepare data for LaTeX export - separate by cost type
        cost_type_codes = {}
        beta1_by_cost = {}  # {cost_type_id: [[alpha, median, q25, q75], ...]}
        beta2_by_cost = {}
        cost_idx = 0

        for cost_type, param_list in sorted(cost_groups.items()):
            # Assign numeric code for this cost type
            cost_type_codes[cost_type] = cost_idx
            beta1_by_cost[cost_idx] = []
            beta2_by_cost[cost_idx] = []

            # Collect data by alpha value for this cost function
            beta1_by_alpha = {}  # {alpha: [values...]}
            beta2_by_alpha = {}

            # Collect all data across parametrizations in this cost function
            for param_name, param_data in param_list:
                shares = param_data['shares']

                # Get panel polynomial estimates
                panel_poly_all = param_data['metrics'].get('panel_poly_estimates_all', {})

                if not panel_poly_all:
                    continue

                # Collect for each alpha value
                for alpha in shares:
                    if alpha not in panel_poly_all:
                        continue

                    poly_all = panel_poly_all[alpha]  # Shape: (n_realizations, 3) - [β₂, β₁, β₀]

                    if poly_all is None or not isinstance(poly_all, np.ndarray):
                        continue

                    # Check if we have any data (non-empty array with proper shape)
                    if poly_all.size == 0 or poly_all.shape[0] == 0:
                        continue

                    # Extract coefficients
                    beta1_values = poly_all[:, 1]  # Linear coefficient
                    beta2_values = poly_all[:, 0]  # Quadratic coefficient

                    # Remove NaNs
                    beta1_values = beta1_values[~np.isnan(beta1_values)]
                    beta2_values = beta2_values[~np.isnan(beta2_values)]

                    # Collect by alpha
                    if alpha not in beta1_by_alpha:
                        beta1_by_alpha[alpha] = []
                        beta2_by_alpha[alpha] = []

                    beta1_by_alpha[alpha].extend(beta1_values)
                    beta2_by_alpha[alpha].extend(beta2_values)

                    plotted_any = True

            # Sort alpha values
            alpha_sorted = sorted(beta1_by_alpha.keys())

            if not alpha_sorted:
                cost_idx += 1
                continue

            # Compute statistics for β₁ (only for alphas with data)
            beta1_means = []
            beta1_lower = []
            beta1_upper = []
            valid_alphas_beta1 = []

            for alpha in alpha_sorted:
                values = np.array(beta1_by_alpha[alpha])
                if len(values) > 0:  # Only compute stats if we have data
                    median_val = np.median(values)
                    q25_val = np.percentile(values, 25)
                    q75_val = np.percentile(values, 75)

                    beta1_means.append(median_val)
                    beta1_lower.append(q25_val)
                    beta1_upper.append(q75_val)
                    valid_alphas_beta1.append(alpha)

                    # Export row: alpha, median, q25, q75
                    beta1_by_cost[cost_idx].append([alpha, median_val, q25_val, q75_val])

            # Compute statistics for β₂ (only for alphas with data)
            beta2_means = []
            beta2_lower = []
            beta2_upper = []
            valid_alphas_beta2 = []

            for alpha in alpha_sorted:
                values = np.array(beta2_by_alpha[alpha])
                if len(values) > 0:  # Only compute stats if we have data
                    median_val = np.median(values)
                    q25_val = np.percentile(values, 25)
                    q75_val = np.percentile(values, 75)

                    beta2_means.append(median_val)
                    beta2_lower.append(q25_val)
                    beta2_upper.append(q75_val)
                    valid_alphas_beta2.append(alpha)

                    # Export row: alpha, median, q25, q75
                    beta2_by_cost[cost_idx].append([alpha, median_val, q25_val, q75_val])

            # Get color for this cost function
            color = cost_colors.get(cost_type, '#333333')

            # Plot β₁ (only if we have valid data)
            if len(valid_alphas_beta1) > 0:
                ax_beta1.plot(valid_alphas_beta1, beta1_means, marker='o', linewidth=2,
                             label=cost_type, color=color)
                ax_beta1.fill_between(valid_alphas_beta1, beta1_lower, beta1_upper,
                                     alpha=0.2, color=color)

            # Plot β₂ (only if we have valid data)
            if len(valid_alphas_beta2) > 0:
                ax_beta2.plot(valid_alphas_beta2, beta2_means, marker='o', linewidth=2,
                             label=cost_type, color=color)
                ax_beta2.fill_between(valid_alphas_beta2, beta2_lower, beta2_upper,
                                     alpha=0.2, color=color)

            cost_idx += 1

        # Formatting for β₁ subplot
        ax_beta1.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=0.8)
        ax_beta1.set_xlabel('α (Profit Sharing)', fontsize=24)
        ax_beta1.set_ylabel('β₁ (Linear Coefficient)', fontsize=24)
        ax_beta1.tick_params(axis='both', which='major', labelsize=20)

        # Formatting for β₂ subplot
        ax_beta2.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=0.8)
        ax_beta2.set_xlabel('α (Profit Sharing)', fontsize=24)
        ax_beta2.set_ylabel('β₂ (Quadratic Coefficient)', fontsize=24)
        ax_beta2.tick_params(axis='both', which='major', labelsize=20)

        if plotted_any:
            # Create a single legend above the subplots
            handles, labels = ax_beta1.get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05),
                      ncol=len(labels), fontsize=20, frameon=False)

            # No suptitle to match counterfactuals.py style
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for legend above

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved panel polynomial estimates plot: {save_path}")

                # Export data for LaTeX
                out_dir = Path(save_path).parent
                stem = Path(save_path).stem

                # Create plots subdirectory
                plots_dir = out_dir / "plots"
                plots_dir.mkdir(exist_ok=True)

                # Export separate CSV files for each cost type
                for cost_id in range(cost_idx):
                    # Beta1 data
                    if beta1_by_cost[cost_id]:
                        beta1_array = np.array(beta1_by_cost[cost_id])
                        np.savetxt(plots_dir / f"poly_beta1_cost{cost_id}.csv",
                                  beta1_array, delimiter=",",
                                  header="alpha,median,q25,q75", comments="")

                    # Beta2 data
                    if beta2_by_cost[cost_id]:
                        beta2_array = np.array(beta2_by_cost[cost_id])
                        np.savetxt(plots_dir / f"poly_beta2_cost{cost_id}.csv",
                                  beta2_array, delimiter=",",
                                  header="alpha,median,q25,q75", comments="")

                print(f"  Exported polynomial data to {plots_dir}")

                # Export cost type codes
                with open(out_dir / f"{stem}_cost_type_codes.txt", "w") as f:
                    for name, code in cost_type_codes.items():
                        f.write(f"{code}: {name}\n")

        else:
            print("No panel polynomial estimates data found to plot!")

        return fig, (ax_beta1, ax_beta2)

    def visualize_market_share_by_quantile(self, all_data, save_path=None):
        """Create faceted plot showing average market share by quantile across α levels

        Shows descriptive patterns of how market share concentration varies with cooperation.
        Each quantile is a separate facet, with α levels shown as different colored lines.

        Args:
            all_data: Dictionary of all parametrization data (from load_data())
            save_path: Optional path to save the figure

        Returns:
            (fig, axes): Matplotlib figure and axes objects
        """
        # Select quantiles to display
        quantiles = ['10th', '50th', '90th', '99th', '100th']
        quantile_indices = [0, 2, 4, 5, 6]  # Indices in the 7-quantile array
        export_rows = []

        save_path = Path(save_path)
        out_dir = save_path.parent
        stem = save_path.stem

        # Create vertically stacked subplots
        fig, axes = plt.subplots(len(quantiles), 1, figsize=(12, 18), sharex=True)

        # Group data by cost function type
        cost_groups = {}
        for param_name, param_data in all_data.items():
            cost_type = param_data['cost_type']
            if cost_type not in cost_groups:
                cost_groups[cost_type] = []
            cost_groups[cost_type].append((param_name, param_data))

        quantile_codes = {q: i for i, q in enumerate(quantiles)}
        cost_type_codes = {cost: i for i, cost in enumerate(sorted(cost_groups.keys()))}

        # For each cost function, collect market share data
        for cost_type, param_list in sorted(cost_groups.items()):
            color = self.cost_function_colors.get(cost_type, '#333333')

            # Collect data aggregated by alpha
            # Structure: {alpha: {quantile_idx: [values from different scenarios]}}
            market_share_by_alpha_quantile = {}

            for param_name, param_data in param_list:
                shares = param_data['shares']
                market_share_data = param_data['metrics'].get('market_share_quantiles_avg', {})

                if not market_share_data:
                    continue

                # For each alpha value
                for alpha in shares:
                    if alpha not in market_share_data or market_share_data[alpha] is None:
                        continue

                    data_array = market_share_data[alpha]

                    # Handle transpose if needed
                    if data_array.shape[0] < data_array.shape[1]:
                        data_array = data_array.T

                    # data_array shape: (n_timesteps, n_quantiles)
                    if alpha not in market_share_by_alpha_quantile:
                        market_share_by_alpha_quantile[alpha] = {q_idx: [] for q_idx in quantile_indices}

                    # Take time average for each quantile
                    for q_idx in quantile_indices:
                        time_avg = np.nanmean(data_array[:, q_idx])
                        market_share_by_alpha_quantile[alpha][q_idx].append(time_avg)

            # Plot for each quantile subplot
            for subplot_idx, (quantile_name, q_idx) in enumerate(zip(quantiles, quantile_indices)):
                ax = axes[subplot_idx]

                # Collect data for this quantile
                alphas_sorted = sorted(market_share_by_alpha_quantile.keys())

                medians = []
                lower_bounds = []
                upper_bounds = []

                for alpha in alphas_sorted:
                    values = np.array(market_share_by_alpha_quantile[alpha][q_idx])
                    if len(values) > 0:
                        median = np.median(values)
                        q25 = np.percentile(values, 25)
                        q75 = np.percentile(values, 75)

                        medians.append(median)
                        lower_bounds.append(q25)
                        upper_bounds.append(q75)

                        export_rows.append([quantile_codes[quantile_name],  # quantile_id
                            cost_type_codes[cost_type],  # cost_type_id
                            alpha, median, q25, q75, ])

                if medians:
                    # Plot line with confidence band
                    ax.plot(alphas_sorted, medians, color=color, linewidth=2,
                           label=cost_type, marker='o', markersize=4)
                    ax.fill_between(alphas_sorted, lower_bounds, upper_bounds,
                                   color=color, alpha=0.2)

        # Format subplots
        for subplot_idx, quantile_name in enumerate(quantiles):
            ax = axes[subplot_idx]

            # Add quantile label
            ax.text(0.02, 0.95, f'Quantile: {quantile_name}',
                   transform=ax.transAxes, fontsize=14, fontweight='bold',
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.set_ylabel('Market Share', fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.grid(True, alpha=0.3)

            # Add legend to first subplot
            if subplot_idx == 0:
                ax.legend(fontsize=12, loc='best')

        # X-label only on bottom subplot
        axes[-1].set_xlabel('α (Profit Sharing)', fontsize=16)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            # Export combined CSV (keep this for reference/debugging)
            export_array = np.array(export_rows)
            np.savetxt(out_dir / f"{stem}.csv", export_array, delimiter=",",
                       header="quantile_id,cost_type_id,alpha,median,q25,q75", comments="")
            print(f"Saved combined market share data: {out_dir / f'{stem}.csv'}")

            # Create subdirectory for individual CSV files
            plots_dir = out_dir / "plots"
            plots_dir.mkdir(exist_ok=True)

            # Export separate CSV files for each quantile-cost combination (for LaTeX)
            unique_quantiles = np.unique(export_array[:, 0])  # Column 0 is quantile_id
            unique_cost_types = np.unique(export_array[:, 1])  # Column 1 is cost_type_id

            print(f"Creating {len(unique_quantiles)} × {len(unique_cost_types)} separate CSV files for LaTeX...")

            for q_id in unique_quantiles:
                for c_id in unique_cost_types:
                    # Filter rows for this quantile-cost combination
                    mask = (export_array[:, 0] == q_id) & (export_array[:, 1] == c_id)
                    subset = export_array[mask]

                    if len(subset) > 0:
                        # Save as q{quantile}_cost{cost_type}.csv in plots subdirectory
                        filename = plots_dir / f"q{int(q_id)}_cost{int(c_id)}.csv"

                        np.savetxt(filename, subset, delimiter=",",
                                   header="quantile_id,cost_type_id,alpha,median,q25,q75", comments="")

            print(f"Created {len(unique_quantiles) * len(unique_cost_types)} LaTeX-ready CSV files in {plots_dir}")

            # Save the code mappings
            with open(out_dir / f"{stem}_quantile_codes.txt", "w") as f:
                for name, code in quantile_codes.items():
                    f.write(f"{code}: {name}\n")

            with open(out_dir / f"{stem}_cost_type_codes.txt", "w") as f:
                for name, code in cost_type_codes.items():
                    f.write(f"{code}: {name}\n")

            print(f"Saved market share by quantile plot: {save_path}")

        return fig, axes

    def visualize_conglomerate_dynamics(self, all_data, save_path=None):
        """Create faceted plot showing conglomerate dynamics metrics across α levels

        Plots four scalar metrics: average conglomerate size, number of conglomerates,
        mergers per period, and exits per period.

        Args:
            all_data: Dictionary of all parametrization data (from load_data())
            save_path: Optional path to save the figure

        Returns:
            (fig, axes): Matplotlib figure and axes objects
        """
        # Define metrics to plot
        metrics_info = [
            ('mean_members_avg', 'Average Conglomerate Size', 'Members'),
            ('num_cong_avg', 'Number of Conglomerates', 'Count'),
            ('mergers_per_period_avg', 'Mergers per Period', 'Mergers'),
            ('exits_per_period_avg', 'Exits per Period', 'Exits')
        ]

        export_rows = []

        save_path = Path(save_path)
        out_dir = save_path.parent
        stem = save_path.stem

        # Create vertically stacked subplots
        fig, axes = plt.subplots(len(metrics_info), 1, figsize=(12, 18), sharex=True)

        # Group data by cost function type
        cost_groups = {}
        for param_name, param_data in all_data.items():
            cost_type = param_data['cost_type']
            if cost_type not in cost_groups:
                cost_groups[cost_type] = []
            cost_groups[cost_type].append((param_name, param_data))

        metric_codes = {metric_key: i for i, (metric_key, _, _) in enumerate(metrics_info)}
        cost_type_codes = {cost: i for i, cost in enumerate(sorted(cost_groups.keys()))}

        # For each cost function, collect data for each metric
        for cost_type, param_list in sorted(cost_groups.items()):
            color = self.cost_function_colors.get(cost_type, '#333333')

            # Collect data aggregated by alpha and metric
            # Structure: {metric_key: {alpha: [values from different scenarios]}}
            metrics_by_alpha = {metric_key: {} for metric_key, _, _ in metrics_info}

            for param_name, param_data in param_list:
                shares = param_data['shares']

                for metric_key, _, _ in metrics_info:
                    metric_data = param_data['metrics'].get(metric_key, {})

                    if not metric_data:
                        continue

                    # For each alpha value
                    for alpha in shares:
                        if alpha not in metric_data or metric_data[alpha] is None:
                            continue

                        data_array = metric_data[alpha]

                        # data_array shape: (n_timesteps,) - scalar time series
                        if alpha not in metrics_by_alpha[metric_key]:
                            metrics_by_alpha[metric_key][alpha] = []

                        # Take time average
                        time_avg = np.nanmean(data_array)
                        metrics_by_alpha[metric_key][alpha].append(time_avg)

            # Plot for each metric subplot
            for subplot_idx, (metric_key, metric_name, y_label) in enumerate(metrics_info):
                ax = axes[subplot_idx]

                # Collect data for this metric
                alpha_data = metrics_by_alpha[metric_key]

                if not alpha_data:
                    continue

                alphas_sorted = sorted(alpha_data.keys())

                medians = []
                lower_bounds = []
                upper_bounds = []

                for alpha in alphas_sorted:
                    values = np.array(alpha_data[alpha])
                    if len(values) > 0:
                        median = np.median(values)
                        q25 = np.percentile(values, 25)
                        q75 = np.percentile(values, 75)

                        medians.append(median)
                        lower_bounds.append(q25)
                        upper_bounds.append(q75)

                        export_rows.append([
                            metric_codes[metric_key],  # metric_id
                            cost_type_codes[cost_type],  # cost_type_id
                            alpha,
                            median,
                            q25,
                            q75
                        ])

                if medians:
                    # Plot line with confidence band
                    ax.plot(alphas_sorted, medians, color=color, linewidth=2,
                           label=cost_type, marker='o', markersize=4)
                    ax.fill_between(alphas_sorted, lower_bounds, upper_bounds,
                                   color=color, alpha=0.2)

        # Format subplots
        for subplot_idx, (metric_key, metric_name, y_label) in enumerate(metrics_info):
            ax = axes[subplot_idx]

            # Add metric label
            ax.text(0.02, 0.95, metric_name,
                   transform=ax.transAxes, fontsize=14, fontweight='bold',
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.set_ylabel(y_label, fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.grid(True, alpha=0.3)

            # Add legend to first subplot
            if subplot_idx == 0:
                ax.legend(fontsize=12, loc='best')

        # X-label only on bottom subplot
        axes[-1].set_xlabel('α (Profit Sharing)', fontsize=16)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

            # Export combined CSV (keep this for reference/debugging)
            export_array = np.array(export_rows)
            np.savetxt(out_dir / f"{stem}.csv", export_array, delimiter=",",
                       header="metric_id,cost_type_id,alpha,median,q25,q75", comments="")
            print(f"Saved combined conglomerate dynamics data: {out_dir / f'{stem}.csv'}")

            # Create subdirectory for individual CSV files
            plots_dir = out_dir / "plots"
            plots_dir.mkdir(exist_ok=True)

            # Export separate CSV files for each metric-cost combination (for LaTeX)
            unique_metrics = np.unique(export_array[:, 0])  # Column 0 is metric_id
            unique_cost_types = np.unique(export_array[:, 1])  # Column 1 is cost_type_id

            print(f"Creating {len(unique_metrics)} × {len(unique_cost_types)} separate CSV files for LaTeX...")

            for m_id in unique_metrics:
                for c_id in unique_cost_types:
                    # Filter rows for this metric-cost combination
                    mask = (export_array[:, 0] == m_id) & (export_array[:, 1] == c_id)
                    subset = export_array[mask]

                    if len(subset) > 0:
                        # Save as m{metric}_cost{cost_type}.csv in plots subdirectory
                        filename = plots_dir / f"m{int(m_id)}_cost{int(c_id)}.csv"

                        np.savetxt(filename, subset, delimiter=",",
                                   header="metric_id,cost_type_id,alpha,median,q25,q75", comments="")

            print(f"Created {len(unique_metrics) * len(unique_cost_types)} LaTeX-ready CSV files in {plots_dir}")

            # Save the code mappings
            with open(out_dir / f"{stem}_metric_codes.txt", "w") as f:
                for name, code in metric_codes.items():
                    f.write(f"{code}: {name}\n")

            with open(out_dir / f"{stem}_cost_type_codes.txt", "w") as f:
                for name, code in cost_type_codes.items():
                    f.write(f"{code}: {name}\n")

            print(f"Saved conglomerate dynamics plot: {save_path}")

        return fig, axes

    def export_panel_polynomial_statistics(self, all_data, save_path=None):
        """Export panel polynomial estimate statistics to CSV and TXT files

        Computes median and IQR (25th-75th percentiles) for β₁ and β₂
        across α values, grouped by cost function.

        Args:
            all_data: Dictionary of all parametrization data (from load_data())
            save_path: Base path for saving files (will create .csv and .txt versions)
                      If None, defaults to 'panel_polynomial_statistics'

        Returns:
            DataFrame with statistics
        """
        if save_path is None:
            save_path = 'panel_polynomial_statistics'

        # Group data by cost function type
        cost_groups = {}
        for param_name, param_data in all_data.items():
            cost_type = param_data['cost_type']
            if cost_type not in cost_groups:
                cost_groups[cost_type] = []
            cost_groups[cost_type].append((param_name, param_data))

        # Collect statistics for each cost function and alpha
        rows = []

        for cost_type, param_list in sorted(cost_groups.items()):
            # Collect data by alpha value for this cost function
            beta1_by_alpha = {}  # {alpha: [values...]}
            beta2_by_alpha = {}

            # Collect all data across parametrizations in this cost function
            for param_name, param_data in param_list:
                shares = param_data['shares']
                panel_poly_all = param_data['metrics'].get('panel_poly_estimates_all', {})

                if not panel_poly_all:
                    continue

                # Collect for each alpha value
                for alpha in shares:
                    if alpha not in panel_poly_all:
                        continue

                    poly_all = panel_poly_all[alpha]  # Shape: (n_realizations, 3) - [β₂, β₁, β₀]

                    if poly_all is None or not isinstance(poly_all, np.ndarray):
                        continue

                    if poly_all.size == 0 or poly_all.shape[0] == 0:
                        continue

                    # Extract coefficients
                    beta1_values = poly_all[:, 1]  # Linear coefficient
                    beta2_values = poly_all[:, 0]  # Quadratic coefficient

                    # Remove NaNs
                    beta1_values = beta1_values[~np.isnan(beta1_values)]
                    beta2_values = beta2_values[~np.isnan(beta2_values)]

                    # Collect by alpha
                    if alpha not in beta1_by_alpha:
                        beta1_by_alpha[alpha] = []
                        beta2_by_alpha[alpha] = []

                    beta1_by_alpha[alpha].extend(beta1_values)
                    beta2_by_alpha[alpha].extend(beta2_values)

            # Compute statistics for each alpha
            for alpha in sorted(beta1_by_alpha.keys()):
                beta1_vals = np.array(beta1_by_alpha[alpha])
                beta2_vals = np.array(beta2_by_alpha[alpha])

                if len(beta1_vals) > 0 and len(beta2_vals) > 0:
                    row = {
                        'cost_function': cost_type,
                        'alpha': alpha,
                        'beta1_median': np.median(beta1_vals),
                        'beta1_p25': np.percentile(beta1_vals, 25),
                        'beta1_p75': np.percentile(beta1_vals, 75),
                        'beta1_n': len(beta1_vals),
                        'beta2_median': np.median(beta2_vals),
                        'beta2_p25': np.percentile(beta2_vals, 25),
                        'beta2_p75': np.percentile(beta2_vals, 75),
                        'beta2_n': len(beta2_vals)
                    }
                    rows.append(row)

        if not rows:
            print("No panel polynomial statistics to export!")
            return None

        # Create DataFrame
        df = pd.DataFrame(rows)

        # Save CSV
        csv_path = f"{save_path}.csv"
        df.to_csv(csv_path, index=False, float_format='%.6f')
        print(f"Saved panel polynomial statistics to {csv_path}")

        # Save formatted TXT
        txt_path = f"{save_path}.txt"
        with open(txt_path, 'w') as f:
            f.write("="*120 + "\n")
            f.write("PANEL POLYNOMIAL ESTIMATE STATISTICS\n")
            f.write("market_share ~ poly(size, 2) pooled across time\n")
            f.write("="*120 + "\n\n")

            # Group by cost function for readable output
            for cost_type in sorted(df['cost_function'].unique()):
                cost_df = df[df['cost_function'] == cost_type]

                f.write(f"\n{cost_type.upper()} Cost Function\n")
                f.write("-"*120 + "\n")
                f.write(f"{'Alpha':>8}  {'β₁ Median':>12}  {'β₁ [P25, P75]':>25}  {'N':>8}  ")
                f.write(f"{'β₂ Median':>12}  {'β₂ [P25, P75]':>25}  {'N':>8}\n")
                f.write("-"*120 + "\n")

                for _, row in cost_df.iterrows():
                    f.write(f"{row['alpha']:8.2f}  ")
                    f.write(f"{row['beta1_median']:12.6f}  ")
                    f.write(f"[{row['beta1_p25']:10.6f}, {row['beta1_p75']:10.6f}]  ")
                    f.write(f"{int(row['beta1_n']):8d}  ")
                    f.write(f"{row['beta2_median']:12.6f}  ")
                    f.write(f"[{row['beta2_p25']:10.6f}, {row['beta2_p75']:10.6f}]  ")
                    f.write(f"{int(row['beta2_n']):8d}\n")

            f.write("\n" + "="*120 + "\n")
            f.write("Note: Median values shown with interquartile range [P25, P75]\n")
            f.write("N = number of observations (realizations × parametrizations)\n")
            f.write("="*120 + "\n")

        print(f"Saved formatted statistics to {txt_path}")

        return df

    def visualize_mobility_ridge_plots(self, all_data, save_path=None):
        """Create ridge plots showing mobility distributions across α levels

        Combines all cost types into a single figure with overlaid distributions.
        Each ridge represents one α level, with different cost types shown in different colors.

        Args:
            all_data: Dictionary of all parametrization data (from load_data())
            save_path: Optional path to save the combined figure

        Returns:
            fig: matplotlib figure object
        """
        from scipy.stats import gaussian_kde
        import matplotlib.pyplot as plt

        # Define cost type colors (matching your LaTeX figure)
        cost_colors = {'linear': '#1f77b4',  # blue
            'quadratic': '#ff7f0e',  # orange
            'power_law': '#2ca02c',  # green
            'exponential': '#d62728'  # red
        }

        # Group data by cost function type
        cost_groups = {}
        for param_name, param_data in all_data.items():
            cost_type = param_data['cost_type']
            if cost_type not in cost_groups:
                cost_groups[cost_type] = []
            cost_groups[cost_type].append((param_name, param_data))

        # Collect mobility data by alpha AND cost type
        # Structure: {alpha: {cost_type: [all firm observations]}}
        rank_ranges_by_alpha_cost = {}
        rank_std_by_alpha_cost = {}

        print("\nCollecting mobility data across all cost types...")

        for cost_type, param_list in sorted(cost_groups.items()):
            print(f"  Processing {cost_type} cost function...")

            for param_name, param_data in param_list:
                shares = param_data['shares']
                rank_ranges_data = param_data['metrics'].get('rank_ranges', {})
                rank_std_data = param_data['metrics'].get('rank_std', {})

                for alpha in shares:
                    # Initialize nested dict if needed
                    if alpha not in rank_ranges_by_alpha_cost:
                        rank_ranges_by_alpha_cost[alpha] = {}
                    if alpha not in rank_std_by_alpha_cost:
                        rank_std_by_alpha_cost[alpha] = {}

                    # Collect rank ranges
                    if alpha in rank_ranges_data and rank_ranges_data[alpha] is not None:
                        firm_ranges = rank_ranges_data[alpha].flatten()
                        firm_ranges = firm_ranges[~np.isnan(firm_ranges)]

                        if cost_type not in rank_ranges_by_alpha_cost[alpha]:
                            rank_ranges_by_alpha_cost[alpha][cost_type] = []
                        rank_ranges_by_alpha_cost[alpha][cost_type].extend(firm_ranges)

                    # Collect rank std
                    if alpha in rank_std_data and rank_std_data[alpha] is not None:
                        firm_std = rank_std_data[alpha].flatten()
                        firm_std = firm_std[~np.isnan(firm_std)]

                        if cost_type not in rank_std_by_alpha_cost[alpha]:
                            rank_std_by_alpha_cost[alpha][cost_type] = []
                        rank_std_by_alpha_cost[alpha][cost_type].extend(firm_std)

        if not rank_ranges_by_alpha_cost and not rank_std_by_alpha_cost:
            print("No mobility data found, skipping...")
            return None

        # Create single figure with 2 subplots (ranges and std)
        fig, axes = plt.subplots(1, 2, figsize=(16, 10), sharey=True)

        # Ridge plot parameters
        overlap = 1.0  # Height of each ridge
        offset = 1.5  # Vertical spacing between ridges

        # Get sorted alpha values
        alphas = sorted(rank_ranges_by_alpha_cost.keys())

        print(f"\nCreating combined ridge plot for {len(alphas)} α levels...")

        # LEFT PANEL: Rank Ranges
        ax_ranges = axes[0]

        for i, alpha in enumerate(alphas):
            y_base = i * offset

            # Plot each cost type at this alpha level
            for cost_type in sorted(cost_colors.keys()):
                if cost_type not in rank_ranges_by_alpha_cost[alpha]:
                    continue

                ranges_data = np.array(rank_ranges_by_alpha_cost[alpha][cost_type])

                if len(ranges_data) < 10:
                    continue

                color = cost_colors[cost_type]

                try:
                    # Compute histogram
                    hist, bin_edges = np.histogram(ranges_data, bins=150, density=True)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                    # Scale and offset for ridge effect
                    y = hist / hist.max() * overlap
                    y = y + y_base

                    # Fill between to create ridge
                    ax_ranges.fill_between(bin_centers, y, y_base, alpha=0.5, color=color, linewidth=1.5,
                                           edgecolor=color)

                except Exception as e:
                    print(f"  Warning: Could not plot {cost_type} ranges α={alpha:.2f}: {e}")
                    continue

            # Add alpha label on the left
            ax_ranges.text(-0.02, y_base + overlap / 2, f'α={alpha:.2f}', transform=ax_ranges.get_yaxis_transform(),
                           ha='right', va='center', fontsize=12, fontweight='bold')

        # RIGHT PANEL: Rank Std Dev
        ax_std = axes[1]

        for i, alpha in enumerate(alphas):
            y_base = i * offset

            # Plot each cost type at this alpha level
            for cost_type in sorted(cost_colors.keys()):
                if cost_type not in rank_std_by_alpha_cost[alpha]:
                    continue

                std_data = np.array(rank_std_by_alpha_cost[alpha][cost_type])

                if len(std_data) < 10:
                    continue

                color = cost_colors[cost_type]

                try:
                    # Compute histogram
                    hist, bin_edges = np.histogram(std_data, bins=150, density=True)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                    # Scale and offset for ridge effect
                    y = hist / hist.max() * overlap
                    y = y + y_base

                    # Fill between to create ridge
                    ax_std.fill_between(bin_centers, y, y_base, alpha=0.5, color=color, linewidth=1.5, edgecolor=color)

                except Exception as e:
                    print(f"  Warning: Could not plot {cost_type} std α={alpha:.2f}: {e}")
                    continue

        # Format left panel
        ax_ranges.set_yticks([])
        ax_ranges.set_xlabel('Rank Range', fontsize=20)
        ax_ranges.set_title('Distribution of Rank Ranges by α', fontsize=22, fontweight='bold')
        ax_ranges.tick_params(axis='x', which='major', labelsize=18)
        ax_ranges.spines['left'].set_visible(False)
        ax_ranges.spines['top'].set_visible(False)
        ax_ranges.spines['right'].set_visible(False)

        # Format right panel
        ax_std.set_yticks([])
        ax_std.set_xlabel('Rank Standard Deviation', fontsize=20)
        ax_std.set_title('Distribution of Rank Volatility by α', fontsize=22, fontweight='bold')
        ax_std.tick_params(axis='x', which='major', labelsize=18)
        ax_std.spines['left'].set_visible(False)
        ax_std.spines['top'].set_visible(False)
        ax_std.spines['right'].set_visible(False)

        # Create legend for cost types
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=cost_colors['linear'], alpha=0.5, edgecolor=cost_colors['linear'], label='Linear'),
            Patch(facecolor=cost_colors['quadratic'], alpha=0.5, edgecolor=cost_colors['quadratic'], label='Quadratic'),
            Patch(facecolor=cost_colors['power_law'], alpha=0.5, edgecolor=cost_colors['power_law'], label='Power Law'),
            Patch(facecolor=cost_colors['exponential'], alpha=0.5, edgecolor=cost_colors['exponential'],
                  label='Exponential')]

        fig.legend(handles=legend_elements, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=18,
                   frameon=True)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.08)  # Make room for legend

        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nSaved combined ridge plot: {save_path}")

        print(f"\nCreated combined figure with {len(alphas)} α levels across 4 cost types")

        return fig

    def export_mobility_ridge_data_latex(self, all_data, out_dir='../latex/figures', n_bins=150):
        """Export histogram data for ridge plots in LaTeX format

        Creates one CSV per alpha level per panel with all cost types in columns.
        This is more efficient than separate files per cost type.

        Args:
            all_data: Dictionary of all parametrization data (from load_data())
            out_dir: Output directory for CSV files
            n_bins: Number of histogram bins
        """
        print("\nExporting mobility ridge plot data for LaTeX...")

        # Define cost type colors and mapping
        cost_type_order = ['linear', 'quadratic', 'power_law', 'exponential']

        # Group data by cost function type
        cost_groups = {}
        for param_name, param_data in all_data.items():
            cost_type = param_data['cost_type']
            if cost_type not in cost_groups:
                cost_groups[cost_type] = []
            cost_groups[cost_type].append((param_name, param_data))

        # Collect mobility data by alpha AND cost type
        rank_ranges_by_alpha_cost = {}
        rank_std_by_alpha_cost = {}

        for cost_type, param_list in sorted(cost_groups.items()):
            for param_name, param_data in param_list:
                shares = param_data['shares']
                rank_ranges_data = param_data['metrics'].get('rank_ranges', {})
                rank_std_data = param_data['metrics'].get('rank_std', {})

                for alpha in shares:
                    if alpha not in rank_ranges_by_alpha_cost:
                        rank_ranges_by_alpha_cost[alpha] = {}
                    if alpha not in rank_std_by_alpha_cost:
                        rank_std_by_alpha_cost[alpha] = {}

                    # Collect rank ranges
                    if alpha in rank_ranges_data and rank_ranges_data[alpha] is not None:
                        firm_ranges = rank_ranges_data[alpha].flatten()
                        firm_ranges = firm_ranges[~np.isnan(firm_ranges)]

                        if cost_type not in rank_ranges_by_alpha_cost[alpha]:
                            rank_ranges_by_alpha_cost[alpha][cost_type] = []
                        rank_ranges_by_alpha_cost[alpha][cost_type].extend(firm_ranges)

                    # Collect rank std
                    if alpha in rank_std_data and rank_std_data[alpha] is not None:
                        firm_std = rank_std_data[alpha].flatten()
                        firm_std = firm_std[~np.isnan(firm_std)]

                        if cost_type not in rank_std_by_alpha_cost[alpha]:
                            rank_std_by_alpha_cost[alpha][cost_type] = []
                        rank_std_by_alpha_cost[alpha][cost_type].extend(firm_std)

        # Create output directory
        out_path = Path(out_dir) / "plots"
        out_path.mkdir(parents=True, exist_ok=True)

        # Get sorted alpha values
        alphas = sorted(rank_ranges_by_alpha_cost.keys())

        # Find global x-range for each metric to ensure consistent bins
        all_range_data = []
        all_std_data = []

        for alpha in alphas:
            for cost_type in cost_type_order:
                if cost_type in rank_ranges_by_alpha_cost.get(alpha, {}):
                    all_range_data.extend(rank_ranges_by_alpha_cost[alpha][cost_type])
                if cost_type in rank_std_by_alpha_cost.get(alpha, {}):
                    all_std_data.extend(rank_std_by_alpha_cost[alpha][cost_type])

        # Define consistent bins across all ridges
        if all_range_data:
            range_bins = np.linspace(np.min(all_range_data), np.max(all_range_data), n_bins + 1)
        if all_std_data:
            std_bins = np.linspace(np.min(all_std_data), np.max(all_std_data), n_bins + 1)

        # Export data for each alpha level
        for i, alpha in enumerate(alphas):
            # RANGES panel
            if alpha in rank_ranges_by_alpha_cost and all_range_data:
                # Compute bin centers
                bin_centers = (range_bins[:-1] + range_bins[1:]) / 2

                # Prepare data array: bin_center, density_cost0, density_cost1, density_cost2, density_cost3
                ridge_data = np.zeros((len(bin_centers), 5))
                ridge_data[:, 0] = bin_centers

                for j, cost_type in enumerate(cost_type_order):
                    if cost_type in rank_ranges_by_alpha_cost[alpha]:
                        ranges_data = np.array(rank_ranges_by_alpha_cost[alpha][cost_type])

                        if len(ranges_data) >= 10:
                            # Compute histogram with consistent bins
                            hist, _ = np.histogram(ranges_data, bins=range_bins, density=True)
                            # Normalize to max=1
                            if hist.max() > 0:
                                hist = hist / hist.max()
                            ridge_data[:, j + 1] = hist

                # Save to CSV
                filename = out_path / f"ridge_ranges_alpha{i:02d}.csv"
                np.savetxt(filename, ridge_data, delimiter=",",
                          header="x,cost0,cost1,cost2,cost3", comments="")
                print(f"  Exported {filename}")

            # STD panel
            if alpha in rank_std_by_alpha_cost and all_std_data:
                # Compute bin centers
                bin_centers = (std_bins[:-1] + std_bins[1:]) / 2

                # Prepare data array
                ridge_data = np.zeros((len(bin_centers), 5))
                ridge_data[:, 0] = bin_centers

                for j, cost_type in enumerate(cost_type_order):
                    if cost_type in rank_std_by_alpha_cost[alpha]:
                        std_data = np.array(rank_std_by_alpha_cost[alpha][cost_type])

                        if len(std_data) >= 10:
                            # Compute histogram with consistent bins
                            hist, _ = np.histogram(std_data, bins=std_bins, density=True)
                            # Normalize to max=1
                            if hist.max() > 0:
                                hist = hist / hist.max()
                            ridge_data[:, j + 1] = hist

                # Save to CSV
                filename = out_path / f"ridge_std_alpha{i:02d}.csv"
                np.savetxt(filename, ridge_data, delimiter=",",
                          header="x,cost0,cost1,cost2,cost3", comments="")
                print(f"  Exported {filename}")

        # Export alpha values mapping
        alpha_file = out_path / "ridge_alpha_values.csv"
        alpha_data = np.column_stack([np.arange(len(alphas)), alphas])
        np.savetxt(alpha_file, alpha_data, delimiter=",",
                  header="index,alpha", comments="")
        print(f"  Exported {alpha_file}")

        print(f"\nExported ridge plot data for {len(alphas)} α levels")
        print(f"Files saved to: {out_path}")

    def create_sensitivity_dashboard(self, results, all_data, save_dir='sensitivity_plots'):
        """Create comprehensive temporal visualization dashboard with faceted plots

        Args:
            results: Dictionary containing regression analysis results
            all_data: Dictionary containing raw parametrization data (for descriptive polynomial plots)
            save_dir: Directory to save all plots
        """
        os.makedirs(save_dir, exist_ok=True)

        # Quantile-based metrics and coefficients to analyze
        quantile_metrics = ['gini_quantiles_avg', 'market_share_quantiles_avg']
        scalar_metrics = ['mean_members_avg', 'num_cong_avg']
        poly_metrics = ['poly_beta_1', 'poly_beta_2']  # Separate for table display
        coefficients = ['linear_beta1', 'quadratic_beta1', 'quadratic_beta2']

        # Panel coefficient plots (time-averaged, uncontrolled - shows variability across cost functions)
        for metric in quantile_metrics:
            for coeff in coefficients:
                # Panel coefficient plot - time-averaged summary across quantiles
                fig, ax = self.visualize_panel_coefficients_by_quantile(results, metric, coeff)
                save_path = os.path.join(save_dir, f'panel_{metric}_{coeff}.pdf')
                plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"Saved panel plot: {save_path}")

        # Controlled panel regressions (time-pooled, with hyperparameter controls)
        if 'controlled_panel' in results.get('cross_parametrization', {}):
            # Gini and market share panel regressions - now returns separate figures
            figures_dict = self.visualize_controlled_panel_results(results)
            for metric_name, fig in figures_dict.items():
                if fig is not None:
                    # Save each metric as a separate file
                    save_path = os.path.join(save_dir, f'controlled_panel_{metric_name}.pdf')
                    fig.savefig(save_path, format='pdf', bbox_inches='tight')
                    plt.close(fig)
                    print(f"Saved controlled panel plot: {save_path}")

            # Mobility panel regressions
            fig, axes = self.visualize_mobility_controlled_panel(results)
            if fig is not None:
                save_path = os.path.join(save_dir, 'controlled_panel_mobility.pdf')
                plt.savefig(save_path, format='pdf', bbox_inches='tight')
                plt.close(fig)
                print(f"Saved mobility controlled panel plot: {save_path}")

            # Polynomial panel regressions - display as table
            poly_table_csv = os.path.join(save_dir, 'polynomial_panel_results.csv')
            poly_table_tex = os.path.join(save_dir, 'polynomial_panel_results.tex')
            poly_table_txt = os.path.join(save_dir, 'polynomial_panel_results.txt')

            # Generate table in all formats
            df = self.create_polynomial_panel_table(results, save_path=poly_table_csv)
            if df is not None:
                # Also save LaTeX and text versions
                self.create_polynomial_panel_table(results, save_path=poly_table_tex)
                self.create_polynomial_panel_table(results, save_path=poly_table_txt)
                print(f"Saved polynomial panel table in 3 formats (CSV, LaTeX, TXT)")

            # Scalar metrics panel regressions - display as table
            scalar_table_csv = os.path.join(save_dir, 'scalar_panel_results.csv')
            scalar_table_tex = os.path.join(save_dir, 'scalar_panel_results.tex')
            scalar_table_txt = os.path.join(save_dir, 'scalar_panel_results.txt')

            # Generate table in all formats
            df = self.create_scalar_panel_table(results, save_path=scalar_table_csv)
            if df is not None:
                # Also save LaTeX and text versions
                self.create_scalar_panel_table(results, save_path=scalar_table_tex)
                self.create_scalar_panel_table(results, save_path=scalar_table_txt)
                print(f"Saved scalar metrics panel table in 3 formats (CSV, LaTeX, TXT)")

        # Controlled temporal evolution plots (timestep-level, with controls)
        # NOTE: poly_beta_1, poly_beta_2, mean_members_avg, and num_cong_avg are NOT included
        # in temporal plots. They are reported in panel regression tables instead (stable over time)
        if 'controlled_temporal' in results.get('cross_parametrization', {}):
            # Quantile metrics (faceted) - only Gini and market share have temporal plots
            for metric in ['gini_quantiles_avg', 'market_share_quantiles_avg']:
                fig, axes = self.visualize_controlled_temporal_faceted(results, metric)
                if fig is not None:
                    save_path = os.path.join(save_dir, f'controlled_temporal_{metric}.pdf')
                    plt.savefig(save_path, format='pdf', bbox_inches='tight')
                    plt.close(fig)
                    print(f"Saved controlled temporal plot: {save_path}")

            # Scalar metrics temporal plots
            # mergers_per_period_avg has temporal evolution, so we plot it
            # (mean_members_avg and num_cong_avg are stable over time - only in tables)
            for metric in ['mergers_per_period_avg']:
                fig, ax = self.visualize_controlled_temporal_scalar(results, metric)
                if fig is not None:
                    save_path = os.path.join(save_dir, f'controlled_temporal_{metric}.pdf')
                    plt.savefig(save_path, format='pdf', bbox_inches='tight')
                    plt.close(fig)
                    print(f"Saved controlled temporal plot: {save_path}")

        # Descriptive polynomial estimate visualizations
        # These show raw patterns of polynomial coefficients across α values
        # WITHOUT regression-on-regression (avoiding double regression problem)
        print("\nGenerating descriptive polynomial estimate visualizations...")

        # Temporal polynomial estimates (how β₁ and β₂ evolve over time for different α)
        fig, axes = self.visualize_temporal_polynomial_estimates(
            all_data,
            save_path=os.path.join(save_dir, 'descriptive_temporal_polynomial.pdf')
        )
        if fig is not None:
            plt.close(fig)

        # Market share by quantile and alpha (descriptive visualization)
        fig, axes = self.visualize_market_share_by_quantile(
            all_data,
            save_path=os.path.join(save_dir, 'descriptive_market_share_by_quantile.pdf')
        )
        if fig is not None:
            plt.close(fig)

        # Panel polynomial estimates (distribution of β₁ and β₂ across realizations)
        fig, axes = self.visualize_panel_polynomial_estimates(
            all_data,
            save_path=os.path.join(save_dir, 'descriptive_panel_polynomial.pdf')
        )
        if fig is not None:
            plt.close(fig)

        # Export panel polynomial statistics to CSV and TXT
        print("\nExporting panel polynomial statistics...")
        self.export_panel_polynomial_statistics(
            all_data,
            save_path=os.path.join(save_dir, 'panel_polynomial_statistics')
        )

        # Conglomerate dynamics (average size, number, mergers, exits)
        print("\nGenerating conglomerate dynamics plots...")
        fig, axes = self.visualize_conglomerate_dynamics(
            all_data,
            save_path=os.path.join(save_dir, 'descriptive_conglomerate_dynamics.pdf')
        )
        if fig is not None:
            plt.close(fig)

        # Ridge plots for mobility distributions (firm-level distributions across α)
        print("\nGenerating mobility ridge plots...")
        ridge_fig = self.visualize_mobility_ridge_plots(
            all_data,
            save_path=os.path.join(save_dir, 'mobility_ridge_plot.pdf')
        )
        if ridge_fig is not None:
            plt.close(ridge_fig)

        # Export ridge plot data for LaTeX
        print("\nExporting ridge plot data for LaTeX...")
        self.export_mobility_ridge_data_latex(all_data, out_dir=save_dir, n_bins=150)

        print(f"\nSensitivity dashboard saved to {save_dir}")
        print("\nPlot types created:")
        print("  - panel_*: Time-averaged coefficients for quantile metrics (bar charts, uncontrolled)")
        print("  - controlled_panel_regression.pdf: Time-pooled regression with hyperparameter controls")
        print("  - controlled_panel_mobility.pdf: Mobility metrics with hyperparameter controls")
        print("  - controlled_temporal_*.pdf: Timestep-level evolution WITH hyperparameter controls (ALL metrics)")
        print("  - polynomial_panel_results.*: Regression results for polynomial coefficients (CSV, LaTeX, TXT)")
        print("\nControlled plots account for:")
        print("  • Market count variations")
        print("  • Firm count variations")
        print("  • Merge threshold variations")
        print("  • Cost function type differences")

def main():
    """Example usage and comprehensive analysis"""
    import argparse

    parser = argparse.ArgumentParser(description='Cross-parametrization sensitivity analysis')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Base directory containing parametrization results')
    parser.add_argument('--test_only', action='store_true',
                       help='Run test analysis on first parametrization only')
    parser.add_argument('--save_results', type=str, default='sensitivity_analysis_results.pkl',
                       help='File to save comprehensive analysis results')
    parser.add_argument('--dashboard_dir', type=str, default='sensitivity_plots',
                       help='Directory to save dashboard plots')
    parser.add_argument('--plot_only', action='store_true',
                       help='Only generate plots from previously saved results (skip expensive analysis)')
    parser.add_argument('--load_results', type=str, default='sensitivity_analysis_results.pkl',
                       help='File to load results from (used with --plot_only)')

    args = parser.parse_args()

    comparator = ParametrizationComparator(args.results_dir)

    # Handle plot-only mode
    if args.plot_only:
        print(f"Loading results from {args.load_results}...")
        try:
            with open(args.load_results, 'rb') as f:
                results = pickle.load(f)
            print("Results loaded successfully!")

            # Print summary
            print(f"\nAnalysis Summary:")
            if 'cross_parametrization' in results and 'by_cost_function' in results['cross_parametrization']:
                cost_data = results['cross_parametrization']['by_cost_function']
                for cost_type, data in cost_data.items():
                    n_params = len(data['parametrizations'])
                    print(f"  {cost_type}: {n_params} parametrizations")

            # Load all_data for descriptive polynomial plots
            print(f"\nLoading parametrization data for polynomial visualizations...")
            all_data = comparator.load_all_parametrizations()

            # Create dashboard
            print(f"\nCreating visualization dashboard...")
            comparator.create_sensitivity_dashboard(results, all_data, args.dashboard_dir)
            print("\nDone! All plots generated without re-running analysis.")
            return
        except FileNotFoundError:
            print(f"Error: Results file '{args.load_results}' not found!")
            print("Please run the analysis first without --plot-only flag.")
            return
        except Exception as e:
            print(f"Error loading results: {e}")
            return

    # Discover parametrizations
    parametrizations = comparator.discover_parametrizations()
    if not parametrizations:
        print("No parametrizations found!")
        return

    print(f"Found {len(parametrizations)} parametrizations:")
    for name, info in parametrizations.items():
        print(f"  {name} (cost: {info['cost_type']})")

    if args.test_only:
        # Test with first parametrization
        first_param = list(parametrizations.keys())[0]
        param_data = comparator.load_parametrization_data(first_param)
        
        print(f"\nTesting with parametrization: {first_param}")
        print(f"Cost type: {param_data['cost_type']}")
        print(f"Available shares: {param_data['shares']}")
        print(f"Available metrics: {list(param_data['metrics'].keys())}")
        
        # Test sensitivity analysis
        per_timestep, panel = comparator.fit_alpha_sensitivity_models(
            param_data, 'gini_quantiles_avg'
        )
        
        if panel:
            print(f"\nPanel regression results for Gini quantiles:")
            print(f"Linear β₁: {panel['linear_beta1']:.6f} (R²: {panel['linear_r2']:.4f})")
            print(f"Quadratic β₁: {panel['quadratic_beta1']:.6f}, β₂: {panel['quadratic_beta2']:.6f} (R²: {panel['quadratic_r2']:.4f})")
            print(f"Observations: {panel['n_observations']}")
            
        # Test mobility analysis
        mobility_results = comparator.analyze_mobility_sensitivity(param_data)
        if mobility_results:
            print(f"\nMobility analysis results:")
            for metric, results in mobility_results.items():
                print(f"  {metric}:")
                print(f"    Linear β₁: {results['linear_beta1']:.6f} (R²: {results['linear_r2']:.4f})")
                print(f"    Quadratic β₁: {results['quadratic_beta1']:.6f}, β₂: {results['quadratic_beta2']:.6f} (R²: {results['quadratic_r2']:.4f})")
    
    else:
        # Run comprehensive analysis
        print("Running comprehensive cross-parametrization analysis...")
        results, all_data = comparator.run_comprehensive_analysis()

        if results:
            # Save results
            with open(args.save_results, 'wb') as f:
                pickle.dump(results, f)
            print(f"Results saved to {args.save_results}")

            # Print summary
            print(f"\nAnalysis Summary:")
            cost_data = results['cross_parametrization']['by_cost_function']
            for cost_type, data in cost_data.items():
                n_params = len(data['parametrizations'])
                print(f"  {cost_type}: {n_params} parametrizations")
                print(f"    Temporal analysis completed for quantile-based metrics")

            # Create dashboard by default
            print(f"\nCreating visualization dashboard...")
            comparator.create_sensitivity_dashboard(results, all_data, args.dashboard_dir)
        else:
            print("No results generated!")

if __name__ == "__main__":
    main()
