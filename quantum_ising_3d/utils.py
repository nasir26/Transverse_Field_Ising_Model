"""
Utility Functions for TFIM Simulations

Provides helper functions for:
- Data I/O
- Statistical analysis
- Parameter sweeps
- Benchmarking
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
import json
import time
import os
from functools import wraps
from dataclasses import dataclass, asdict


@dataclass
class SimulationParameters:
    """Container for simulation parameters."""
    L: int = 2
    J: float = 1.0
    Gamma: float = 1.0
    beta: float = 1.0
    periodic: bool = True
    method: str = 'exact'  # 'exact', 'qmc', 'annealing'


def timer(func: Callable) -> Callable:
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} completed in {elapsed:.3f} seconds")
        return result
    return wrapper


def save_results(data: Dict[str, Any], filename: str, directory: str = './results'):
    """
    Save simulation results to JSON file.
    
    Handles numpy arrays by converting to lists.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    filepath = os.path.join(directory, filename)
    
    # Convert numpy arrays to lists
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, complex):
            return {'real': obj.real, 'imag': obj.imag}
        return obj
    
    data_converted = convert(data)
    
    with open(filepath, 'w') as f:
        json.dump(data_converted, f, indent=2)
    
    print(f"Results saved to {filepath}")


def load_results(filename: str, directory: str = './results') -> Dict[str, Any]:
    """Load simulation results from JSON file."""
    filepath = os.path.join(directory, filename)
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Convert lists back to numpy arrays where appropriate
    def convert(obj):
        if isinstance(obj, list) and len(obj) > 0:
            if isinstance(obj[0], (int, float)):
                return np.array(obj)
            return [convert(v) for v in obj]
        elif isinstance(obj, dict):
            if 'real' in obj and 'imag' in obj:
                return complex(obj['real'], obj['imag'])
            return {k: convert(v) for k, v in obj.items()}
        return obj
    
    return convert(data)


def bootstrap_error(data: np.ndarray, n_bootstrap: int = 1000,
                    statistic: Callable = np.mean) -> Tuple[float, float]:
    """
    Compute bootstrap error estimate.
    
    Parameters
    ----------
    data : np.ndarray
        Sample data
    n_bootstrap : int
        Number of bootstrap samples
    statistic : Callable
        Statistic to compute (default: mean)
        
    Returns
    -------
    mean : float
        Statistic value
    error : float
        Bootstrap error estimate
    """
    n = len(data)
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        sample = data[np.random.randint(0, n, n)]
        bootstrap_stats.append(statistic(sample))
    
    mean = statistic(data)
    error = np.std(bootstrap_stats)
    
    return mean, error


def binning_error(data: np.ndarray, n_bins: int = 20) -> Tuple[float, float]:
    """
    Compute error using binning analysis.
    
    Useful for correlated data from Monte Carlo simulations.
    """
    n = len(data)
    bin_size = n // n_bins
    
    if bin_size < 1:
        return np.mean(data), np.std(data) / np.sqrt(n)
    
    bin_means = []
    for i in range(n_bins):
        bin_data = data[i * bin_size:(i + 1) * bin_size]
        bin_means.append(np.mean(bin_data))
    
    mean = np.mean(data)
    error = np.std(bin_means) / np.sqrt(n_bins)
    
    return mean, error


def autocorrelation_time(data: np.ndarray, max_lag: int = 100) -> float:
    """
    Estimate integrated autocorrelation time.
    
    τ_int = 1/2 + Σ_{t=1}^{t_max} ρ(t)
    
    where ρ(t) is the autocorrelation function.
    """
    n = len(data)
    mean = np.mean(data)
    var = np.var(data)
    
    if var < 1e-15:
        return 0.0
    
    data_centered = data - mean
    
    tau = 0.5
    for t in range(1, min(max_lag, n // 2)):
        rho_t = np.mean(data_centered[:-t] * data_centered[t:]) / var
        if rho_t < 0.01:  # Cut off when correlation becomes negligible
            break
        tau += rho_t
    
    return tau


def effective_sample_size(data: np.ndarray) -> float:
    """
    Compute effective sample size accounting for correlations.
    
    N_eff = N / (2 τ_int)
    """
    n = len(data)
    tau = autocorrelation_time(data)
    return n / (2 * tau) if tau > 0 else n


def parameter_sweep_1d(param_name: str, param_values: np.ndarray,
                        run_function: Callable,
                        base_params: Optional[Dict] = None,
                        **kwargs) -> Dict[str, np.ndarray]:
    """
    Perform 1D parameter sweep.
    
    Parameters
    ----------
    param_name : str
        Name of parameter to sweep
    param_values : np.ndarray
        Values to sweep over
    run_function : Callable
        Function to run simulation
    base_params : Dict, optional
        Base parameters
    **kwargs
        Additional arguments for run_function
        
    Returns
    -------
    Dict with results for each parameter value
    """
    if base_params is None:
        base_params = {}
    
    results = {param_name: param_values}
    
    for i, val in enumerate(param_values):
        print(f"\n{param_name} = {val} ({i+1}/{len(param_values)})")
        
        params = base_params.copy()
        params[param_name] = val
        
        result = run_function(**params, **kwargs)
        
        # Store results
        for key, value in result.items():
            if key not in results:
                results[key] = []
            results[key].append(value)
    
    # Convert to arrays
    for key in results:
        if isinstance(results[key], list):
            results[key] = np.array(results[key])
    
    return results


def parameter_sweep_2d(param1_name: str, param1_values: np.ndarray,
                        param2_name: str, param2_values: np.ndarray,
                        run_function: Callable,
                        base_params: Optional[Dict] = None,
                        **kwargs) -> Dict[str, np.ndarray]:
    """
    Perform 2D parameter sweep.
    
    Returns results as 2D arrays indexed by [i1, i2].
    """
    if base_params is None:
        base_params = {}
    
    n1 = len(param1_values)
    n2 = len(param2_values)
    
    results = {
        param1_name: param1_values,
        param2_name: param2_values
    }
    
    for i1, val1 in enumerate(param1_values):
        for i2, val2 in enumerate(param2_values):
            print(f"\n{param1_name}={val1}, {param2_name}={val2} "
                  f"({i1*n2 + i2 + 1}/{n1*n2})")
            
            params = base_params.copy()
            params[param1_name] = val1
            params[param2_name] = val2
            
            result = run_function(**params, **kwargs)
            
            # Store results
            for key, value in result.items():
                if key not in results:
                    results[key] = np.zeros((n1, n2))
                results[key][i1, i2] = value
    
    return results


def estimate_memory_usage(L: int, method: str = 'exact') -> str:
    """
    Estimate memory usage for simulation.
    
    Parameters
    ----------
    L : int
        Linear lattice size
    method : str
        'exact' or 'qmc'
        
    Returns
    -------
    str
        Human-readable memory estimate
    """
    N = L ** 3
    dim = 2 ** N
    
    if method == 'exact':
        # Sparse matrix: roughly nnz * 16 bytes per element
        # For TFIM, nnz ≈ N * dim (off-diagonal) + dim (diagonal)
        nnz_estimate = N * dim + dim
        bytes_sparse = nnz_estimate * 16  # complex + indices
        
        # State vector
        bytes_state = dim * 16  # complex
        
        total = bytes_sparse + bytes_state
    else:
        # QMC: scales polynomially with N
        total = N * 1000 * 8  # Rough estimate
    
    # Convert to human-readable
    if total < 1024:
        return f"{total} B"
    elif total < 1024 ** 2:
        return f"{total / 1024:.1f} KB"
    elif total < 1024 ** 3:
        return f"{total / 1024**2:.1f} MB"
    else:
        return f"{total / 1024**3:.1f} GB"


def check_feasibility(L: int, method: str = 'exact') -> Dict[str, Any]:
    """
    Check if simulation is feasible with given parameters.
    
    Returns
    -------
    Dict with feasibility assessment
    """
    N = L ** 3
    dim = 2 ** N
    
    result = {
        'L': L,
        'N': N,
        'dim': dim,
        'method': method,
        'memory_estimate': estimate_memory_usage(L, method),
        'feasible': True,
        'warnings': []
    }
    
    if method == 'exact':
        if dim > 2 ** 20:  # ~1 million states
            result['warnings'].append(
                f"Large Hilbert space (dim = {dim}). Consider using QMC.")
        if dim > 2 ** 25:
            result['feasible'] = False
            result['warnings'].append(
                "Hilbert space too large for exact diagonalization.")
    
    # Print summary
    print("Feasibility Check")
    print("-" * 40)
    print(f"Lattice: {L}x{L}x{L} = {N} sites")
    print(f"Hilbert space dimension: {dim}")
    print(f"Memory estimate: {result['memory_estimate']}")
    print(f"Feasible: {'Yes' if result['feasible'] else 'No'}")
    for warning in result['warnings']:
        print(f"Warning: {warning}")
    
    return result


def format_scientific(value: float, precision: int = 4) -> str:
    """Format number in scientific notation."""
    if abs(value) < 1e-10:
        return "0"
    elif abs(value) < 0.01 or abs(value) > 1000:
        return f"{value:.{precision}e}"
    else:
        return f"{value:.{precision}f}"


def create_lattice_connectivity(L: int, dim: int = 3,
                                 periodic: bool = True) -> np.ndarray:
    """
    Create adjacency matrix for d-dimensional hypercubic lattice.
    
    Parameters
    ----------
    L : int
        Linear size
    dim : int
        Spatial dimension
    periodic : bool
        Periodic boundaries
        
    Returns
    -------
    np.ndarray
        Adjacency matrix
    """
    N = L ** dim
    adj = np.zeros((N, N), dtype=int)
    
    for idx in range(N):
        # Convert to d-dimensional coordinates
        coords = []
        temp = idx
        for d in range(dim):
            coords.append(temp % L)
            temp //= L
        
        # Neighbors in each direction
        for d in range(dim):
            for delta in [-1, 1]:
                new_coords = coords.copy()
                new_coords[d] = coords[d] + delta
                
                if periodic:
                    new_coords[d] = new_coords[d] % L
                elif new_coords[d] < 0 or new_coords[d] >= L:
                    continue
                
                # Convert back to linear index
                neighbor_idx = sum(c * (L ** i) for i, c in enumerate(new_coords))
                adj[idx, neighbor_idx] = 1
    
    return adj


class ProgressTracker:
    """Track progress of long simulations."""
    
    def __init__(self, total: int, description: str = "Progress"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
    
    def update(self, n: int = 1):
        """Update progress by n steps."""
        self.current += n
        self._print_progress()
    
    def _print_progress(self):
        """Print progress bar."""
        frac = self.current / self.total
        elapsed = time.time() - self.start_time
        
        if self.current > 0:
            eta = elapsed * (self.total - self.current) / self.current
        else:
            eta = 0
        
        bar_length = 40
        filled = int(bar_length * frac)
        bar = '=' * filled + '-' * (bar_length - filled)
        
        print(f"\r{self.description}: [{bar}] {100*frac:.1f}% "
              f"(ETA: {eta:.0f}s)", end='', flush=True)
        
        if self.current >= self.total:
            print()  # New line at completion
    
    def complete(self):
        """Mark as complete."""
        self.current = self.total
        self._print_progress()
        total_time = time.time() - self.start_time
        print(f"Total time: {total_time:.2f}s")
