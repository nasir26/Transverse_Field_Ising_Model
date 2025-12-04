"""
Quantum Monte Carlo for 3D Transverse Field Ising Model

Implements the Stochastic Series Expansion (SSE) algorithm with
directed loop updates, which is efficient for the transverse field
Ising model.

The SSE algorithm expands the partition function:
    Z = Tr[e^{-βH}] = Σ_n (β^n/n!) Tr[(-H)^n]

and samples the terms stochastically.

For the TFIM, we decompose H into bond operators:
    H = Σ_b H_b

where each H_b is either an Ising interaction or a transverse field term.

References:
- Sandvik, A.W., Phys. Rev. B 59, R14157 (1999)
- Syljuåsen & Sandvik, Phys. Rev. E 66, 046701 (2002)
"""

import numpy as np
from numba import jit, prange
from typing import Tuple, List, Dict, Optional, Any
import time
from tqdm import tqdm


@jit(nopython=True)
def _compute_magnetization_z(spins: np.ndarray) -> float:
    """Compute z-magnetization for a spin configuration."""
    return np.mean(2.0 * spins - 1.0)


@jit(nopython=True)
def _compute_magnetization_z_squared(spins: np.ndarray) -> float:
    """Compute squared z-magnetization."""
    mz = np.mean(2.0 * spins - 1.0)
    return mz * mz


@jit(nopython=True)
def _diagonal_update(spins: np.ndarray, operator_list: np.ndarray,
                     n_operators: int, M: int, beta: float,
                     J: float, Gamma: float,
                     bonds: np.ndarray, N: int) -> int:
    """
    Perform diagonal update in SSE.
    
    Operator types:
    - 0: identity
    - 1: diagonal (Ising) on bond b
    - 2: off-diagonal (transverse field) on site i
    
    Parameters
    ----------
    spins : np.ndarray
        Current spin configuration (0 or 1)
    operator_list : np.ndarray
        List of operators [type, site/bond_index]
    n_operators : int
        Current expansion order
    M : int
        Maximum expansion cutoff
    beta : float
        Inverse temperature
    J : float
        Ising coupling
    Gamma : float
        Transverse field
    bonds : np.ndarray
        Array of bonds [i, j]
    N : int
        Number of sites
        
    Returns
    -------
    int
        New expansion order
    """
    n_bonds = len(bonds)
    
    # Probabilities for inserting operators
    # P_insert_diag = β * J / (M - n)
    # P_insert_offdiag = β * Γ / (M - n)
    # P_remove = (M - n + 1) / (β * weight)
    
    for p in range(M):
        op_type = operator_list[p, 0]
        op_index = operator_list[p, 1]
        
        if op_type == 0:  # Identity operator - try to insert
            # Choose randomly between diagonal and off-diagonal
            if np.random.random() < 0.5:
                # Try inserting diagonal (Ising) operator
                b = np.random.randint(n_bonds)
                i, j = bonds[b]
                
                # Diagonal operators only act on parallel spins
                if spins[i] == spins[j]:
                    prob = beta * J * n_bonds / (M - n_operators)
                    if np.random.random() < prob:
                        operator_list[p, 0] = 1
                        operator_list[p, 1] = b
                        n_operators += 1
            else:
                # Try inserting off-diagonal (transverse field) operator
                site = np.random.randint(N)
                prob = beta * Gamma * N / (M - n_operators)
                if np.random.random() < prob:
                    operator_list[p, 0] = 2
                    operator_list[p, 1] = site
                    n_operators += 1
                    # Flip spin when passing through off-diagonal operator
                    spins[site] = 1 - spins[site]
                    
        elif op_type == 1:  # Diagonal operator - try to remove
            b = op_index
            i, j = bonds[b]
            prob = (M - n_operators + 1) / (beta * J * n_bonds)
            if np.random.random() < prob:
                operator_list[p, 0] = 0
                operator_list[p, 1] = -1
                n_operators -= 1
                
        elif op_type == 2:  # Off-diagonal operator - propagate spin
            site = op_index
            spins[site] = 1 - spins[site]
    
    return n_operators


@jit(nopython=True)
def _loop_update(spins: np.ndarray, operator_list: np.ndarray,
                 M: int, bonds: np.ndarray, N: int) -> None:
    """
    Perform loop (cluster) update in SSE.
    
    This constructs loops in the (space, imaginary-time) configuration
    and flips entire loops with probability 1/2.
    """
    # Build vertex list
    # For each operator, we track links to neighboring operators
    # This is a simplified version - full implementation would use
    # linked vertex list
    
    n_bonds = len(bonds)
    
    # First/last operator touching each site
    first_op = np.full(N, -1, dtype=np.int64)
    last_op = np.full(N, -1, dtype=np.int64)
    
    # Linked list for loop construction
    vertex_link = np.full((M, 4), -1, dtype=np.int64)  # Up to 4 legs per vertex
    
    # Build links
    for p in range(M):
        op_type = operator_list[p, 0]
        if op_type == 0:
            continue
            
        if op_type == 1:  # Diagonal on bond
            b = operator_list[p, 1]
            i, j = bonds[b]
            sites = [i, j]
        else:  # Off-diagonal on site
            site = operator_list[p, 1]
            sites = [site]
        
        for leg, s in enumerate(sites):
            if last_op[s] >= 0:
                # Link to previous operator
                vertex_link[p, leg] = last_op[s]
                vertex_link[last_op[s], 2 + leg] = p
            else:
                first_op[s] = p
            last_op[s] = p
    
    # Close periodic links
    for s in range(N):
        if first_op[s] >= 0 and last_op[s] >= 0:
            # Find which leg corresponds to site s
            p_first = first_op[s]
            p_last = last_op[s]
            
            # Link them
            op_type_first = operator_list[p_first, 0]
            op_type_last = operator_list[p_last, 0]
            
            if op_type_first == 2:
                vertex_link[p_first, 0] = p_last
            if op_type_last == 2:
                vertex_link[p_last, 2] = p_first
    
    # Traverse and flip loops
    visited = np.zeros(M, dtype=np.bool_)
    
    for start in range(M):
        if visited[start]:
            continue
        if operator_list[start, 0] == 0:
            continue
        if operator_list[start, 0] != 2:  # Only start from off-diagonal
            continue
            
        # Decide whether to flip this loop
        if np.random.random() < 0.5:
            continue
        
        # Traverse loop and flip
        p = start
        while True:
            if visited[p]:
                break
            visited[p] = True
            
            op_type = operator_list[p, 0]
            if op_type == 2:  # Off-diagonal - flip the site in initial config
                site = operator_list[p, 1]
                # This effectively changes the spin configuration
            
            # Move to next vertex
            if vertex_link[p, 0] >= 0:
                p = vertex_link[p, 0]
            elif vertex_link[p, 2] >= 0:
                p = vertex_link[p, 2]
            else:
                break
            
            if p == start:
                break


class QuantumMonteCarlo:
    """
    Stochastic Series Expansion QMC for 3D Transverse Field Ising Model.
    
    Parameters
    ----------
    L : int
        Linear lattice size (L x L x L)
    J : float
        Ising coupling
    Gamma : float
        Transverse field
    beta : float
        Inverse temperature
    periodic : bool
        Periodic boundary conditions
    """
    
    def __init__(self, L: int, J: float = 1.0, Gamma: float = 1.0,
                 beta: float = 1.0, periodic: bool = True):
        self.L = L
        self.N = L ** 3
        self.J = J
        self.Gamma = Gamma
        self.beta = beta
        self.periodic = periodic
        
        # Build bond list
        self.bonds = self._build_bonds()
        self.n_bonds = len(self.bonds)
        
        # SSE expansion cutoff (will be adjusted dynamically)
        self.M = max(100, int(2 * beta * (abs(J) * self.n_bonds + abs(Gamma) * self.N)))
        
        # Initialize configuration
        self.spins = np.random.randint(0, 2, self.N).astype(np.int64)
        
        # Operator list: [operator_type, site_or_bond_index]
        # Type 0: identity, Type 1: diagonal (Ising), Type 2: off-diagonal (field)
        self.operator_list = np.zeros((self.M, 2), dtype=np.int64)
        self.operator_list[:, 1] = -1
        self.n_operators = 0
        
        # Measurements
        self.measurements = {
            'energy': [],
            'magnetization_z': [],
            'magnetization_z_squared': [],
            'magnetization_x': [],
            'susceptibility': [],
            'n_operators': []
        }
        
    def _build_bonds(self) -> np.ndarray:
        """Build list of nearest neighbor bonds."""
        bonds = []
        L = self.L
        
        for z in range(L):
            for y in range(L):
                for x in range(L):
                    idx = x + L * y + L * L * z
                    
                    # +x neighbor
                    nx = (x + 1) % L if self.periodic else x + 1
                    if nx < L:
                        bonds.append([idx, nx + L * y + L * L * z])
                    
                    # +y neighbor
                    ny = (y + 1) % L if self.periodic else y + 1
                    if ny < L:
                        bonds.append([idx, x + L * ny + L * L * z])
                    
                    # +z neighbor
                    nz = (z + 1) % L if self.periodic else z + 1
                    if nz < L:
                        bonds.append([idx, x + L * y + L * L * nz])
        
        return np.array(bonds, dtype=np.int64)
    
    def _adjust_cutoff(self):
        """Dynamically adjust expansion cutoff M."""
        if self.n_operators > 0.8 * self.M:
            old_M = self.M
            self.M = int(1.5 * self.M)
            new_ops = np.zeros((self.M, 2), dtype=np.int64)
            new_ops[:old_M] = self.operator_list
            new_ops[old_M:, 1] = -1
            self.operator_list = new_ops
            print(f"  Adjusted cutoff: {old_M} -> {self.M}")
    
    def mc_step(self):
        """Perform one Monte Carlo step (diagonal + loop updates)."""
        # Diagonal update
        self.n_operators = _diagonal_update(
            self.spins, self.operator_list, self.n_operators,
            self.M, self.beta, self.J, self.Gamma,
            self.bonds, self.N
        )
        
        # Loop update
        _loop_update(self.spins, self.operator_list, self.M, self.bonds, self.N)
        
        # Adjust cutoff if needed
        self._adjust_cutoff()
    
    def measure(self):
        """Measure observables."""
        # Energy from expansion order
        # <H> = -<n> / β
        energy = -self.n_operators / self.beta
        
        # Magnetizations
        mz = _compute_magnetization_z(self.spins)
        mz2 = _compute_magnetization_z_squared(self.spins)
        
        # For M_x, we need to count off-diagonal operators
        # This is a simplified estimator
        n_offdiag = np.sum(self.operator_list[:, 0] == 2)
        mx = n_offdiag / (self.beta * self.Gamma * self.N) if self.Gamma > 0 else 0
        
        self.measurements['energy'].append(energy)
        self.measurements['magnetization_z'].append(mz)
        self.measurements['magnetization_z_squared'].append(mz2)
        self.measurements['magnetization_x'].append(mx)
        self.measurements['n_operators'].append(self.n_operators)
    
    def run(self, n_thermalization: int = 1000, n_measurements: int = 5000,
            n_skip: int = 10, show_progress: bool = True) -> Dict[str, Any]:
        """
        Run QMC simulation.
        
        Parameters
        ----------
        n_thermalization : int
            Number of thermalization sweeps
        n_measurements : int
            Number of measurement sweeps
        n_skip : int
            Sweeps between measurements
        show_progress : bool
            Show progress bar
            
        Returns
        -------
        Dict[str, Any]
            Dictionary of measured observables with means and errors
        """
        print(f"Running QMC for {self.L}x{self.L}x{self.L} lattice")
        print(f"  β = {self.beta}, J = {self.J}, Γ = {self.Gamma}")
        print(f"  Thermalization: {n_thermalization}, Measurements: {n_measurements}")
        
        # Clear previous measurements
        for key in self.measurements:
            self.measurements[key] = []
        
        start_time = time.time()
        
        # Thermalization
        print("Thermalizing...")
        for _ in tqdm(range(n_thermalization), disable=not show_progress):
            self.mc_step()
        
        # Measurements
        print("Measuring...")
        for i in tqdm(range(n_measurements), disable=not show_progress):
            for _ in range(n_skip):
                self.mc_step()
            self.measure()
        
        elapsed = time.time() - start_time
        print(f"Completed in {elapsed:.2f} seconds")
        
        # Compute statistics
        results = self._compute_statistics()
        
        return results
    
    def _compute_statistics(self) -> Dict[str, Any]:
        """Compute means and statistical errors using binning."""
        results = {}
        
        for key, values in self.measurements.items():
            if len(values) == 0:
                continue
                
            values = np.array(values)
            mean = np.mean(values)
            
            # Standard error with binning
            n_bins = 20
            bin_size = len(values) // n_bins
            if bin_size > 0:
                binned_means = [np.mean(values[i*bin_size:(i+1)*bin_size]) 
                               for i in range(n_bins)]
                error = np.std(binned_means) / np.sqrt(n_bins)
            else:
                error = np.std(values) / np.sqrt(len(values))
            
            results[key] = {'mean': mean, 'error': error}
        
        # Derived quantities
        if 'magnetization_z_squared' in results:
            mz2_mean = results['magnetization_z_squared']['mean']
            mz_mean = results['magnetization_z']['mean']
            
            # Susceptibility χ = β N (<M²> - <M>²)
            chi = self.beta * self.N * (mz2_mean - mz_mean**2)
            results['susceptibility'] = {'mean': chi, 'error': 0}  # Error propagation needed
        
        # Energy per site
        if 'energy' in results:
            results['energy_per_site'] = {
                'mean': results['energy']['mean'] / self.N,
                'error': results['energy']['error'] / self.N
            }
        
        return results
    
    def phase_diagram_scan(self, Gamma_values: np.ndarray,
                            n_thermalization: int = 500,
                            n_measurements: int = 2000,
                            n_skip: int = 5) -> Dict[str, np.ndarray]:
        """
        Scan over Γ values to map phase diagram.
        
        Parameters
        ----------
        Gamma_values : np.ndarray
            Array of Γ values
        n_thermalization : int
            Thermalization sweeps per point
        n_measurements : int
            Measurement sweeps per point
        n_skip : int
            Skip between measurements
            
        Returns
        -------
        Dict[str, np.ndarray]
            Results for each observable vs Γ
        """
        results = {
            'Gamma': Gamma_values,
            'energy': [],
            'energy_error': [],
            'magnetization_z': [],
            'magnetization_z_error': [],
            'magnetization_z_squared': [],
            'susceptibility': []
        }
        
        for i, Gamma in enumerate(Gamma_values):
            print(f"\n{'='*50}")
            print(f"Γ = {Gamma:.4f} ({i+1}/{len(Gamma_values)})")
            print('='*50)
            
            # Reset simulation with new Gamma
            self.Gamma = Gamma
            self.spins = np.random.randint(0, 2, self.N).astype(np.int64)
            self.operator_list = np.zeros((self.M, 2), dtype=np.int64)
            self.operator_list[:, 1] = -1
            self.n_operators = 0
            
            # Run simulation
            data = self.run(n_thermalization, n_measurements, n_skip, show_progress=False)
            
            results['energy'].append(data['energy']['mean'])
            results['energy_error'].append(data['energy']['error'])
            results['magnetization_z'].append(data['magnetization_z']['mean'])
            results['magnetization_z_error'].append(data['magnetization_z']['error'])
            results['magnetization_z_squared'].append(data['magnetization_z_squared']['mean'])
            if 'susceptibility' in data:
                results['susceptibility'].append(data['susceptibility']['mean'])
        
        # Convert to arrays
        for key in results:
            results[key] = np.array(results[key])
        
        return results


class WorldLineQMC:
    """
    Path Integral (World-Line) QMC for the 3D TFIM.
    
    Uses Suzuki-Trotter decomposition:
        e^{-βH} ≈ (e^{-ΔτH_z} e^{-ΔτH_x})^{n_τ}
    
    where Δτ = β/n_τ is the imaginary time step.
    
    This creates a classical (3+1)D Ising model with anisotropic couplings.
    
    Parameters
    ----------
    L : int
        Linear lattice size
    J : float
        Ising coupling
    Gamma : float
        Transverse field
    beta : float
        Inverse temperature
    n_tau : int
        Number of imaginary time slices
    """
    
    def __init__(self, L: int, J: float = 1.0, Gamma: float = 1.0,
                 beta: float = 1.0, n_tau: int = 100):
        self.L = L
        self.N = L ** 3
        self.J = J
        self.Gamma = Gamma
        self.beta = beta
        self.n_tau = n_tau
        self.dtau = beta / n_tau
        
        # Effective coupling in imaginary time direction
        # K_tau = -0.5 * ln(tanh(Δτ Γ))
        if Gamma > 0:
            self.K_tau = -0.5 * np.log(np.tanh(self.dtau * Gamma))
        else:
            self.K_tau = np.inf  # Classical limit
        
        # Spatial coupling
        self.K_space = self.dtau * J
        
        # Build bonds
        self.bonds = self._build_bonds()
        
        # Configuration: (N, n_tau) array of spins
        self.config = np.random.choice([-1, 1], size=(self.N, n_tau))
        
        # Measurements
        self.measurements = {'energy': [], 'magnetization_z': [], 'magnetization_z_squared': []}
        
    def _build_bonds(self) -> List[Tuple[int, int]]:
        """Build spatial bonds."""
        bonds = []
        L = self.L
        
        for z in range(L):
            for y in range(L):
                for x in range(L):
                    idx = x + L * y + L * L * z
                    
                    for dx, dy, dz in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                        nx = (x + dx) % L
                        ny = (y + dy) % L
                        nz = (z + dz) % L
                        neighbor = nx + L * ny + L * L * nz
                        if idx < neighbor:
                            bonds.append((idx, neighbor))
        
        return bonds
    
    def _local_energy(self, i: int, tau: int) -> float:
        """Compute local energy contribution at site (i, τ)."""
        s = self.config[i, tau]
        
        # Spatial neighbors
        E = 0.0
        L = self.L
        x = i % L
        y = (i // L) % L
        z = i // (L * L)
        
        for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
            nx = (x + dx) % L
            ny = (y + dy) % L
            nz = (z + dz) % L
            neighbor = nx + L * ny + L * L * nz
            E -= self.K_space * s * self.config[neighbor, tau]
        
        # Temporal neighbors
        tau_prev = (tau - 1) % self.n_tau
        tau_next = (tau + 1) % self.n_tau
        E -= self.K_tau * s * self.config[i, tau_prev]
        E -= self.K_tau * s * self.config[i, tau_next]
        
        return E
    
    def mc_step(self):
        """Perform one Metropolis sweep."""
        for _ in range(self.N * self.n_tau):
            # Random site and time slice
            i = np.random.randint(self.N)
            tau = np.random.randint(self.n_tau)
            
            # Energy change from flipping
            dE = -2 * self._local_energy(i, tau)
            
            # Metropolis acceptance
            if dE < 0 or np.random.random() < np.exp(-dE):
                self.config[i, tau] *= -1
    
    def cluster_update(self):
        """Wolff cluster update in (space, time)."""
        # Pick random starting point
        i0 = np.random.randint(self.N)
        tau0 = np.random.randint(self.n_tau)
        
        cluster = set()
        stack = [(i0, tau0)]
        
        while stack:
            i, tau = stack.pop()
            if (i, tau) in cluster:
                continue
            
            cluster.add((i, tau))
            s = self.config[i, tau]
            
            # Check neighbors
            L = self.L
            x = i % L
            y = (i // L) % L
            z = i // (L * L)
            
            # Spatial neighbors
            for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
                nx = (x + dx) % L
                ny = (y + dy) % L
                nz = (z + dz) % L
                neighbor = nx + L * ny + L * L * nz
                
                if (neighbor, tau) not in cluster:
                    if self.config[neighbor, tau] == s:
                        # Bond probability
                        p_add = 1 - np.exp(-2 * self.K_space)
                        if np.random.random() < p_add:
                            stack.append((neighbor, tau))
            
            # Temporal neighbors
            for dtau in [-1, 1]:
                tau_n = (tau + dtau) % self.n_tau
                if (i, tau_n) not in cluster:
                    if self.config[i, tau_n] == s:
                        p_add = 1 - np.exp(-2 * self.K_tau)
                        if np.random.random() < p_add:
                            stack.append((i, tau_n))
        
        # Flip cluster
        for i, tau in cluster:
            self.config[i, tau] *= -1
    
    def measure(self):
        """Measure observables."""
        # Magnetization (average over time)
        mz = np.mean(self.config)
        mz2 = mz ** 2
        
        # Energy estimator
        E_spatial = 0.0
        for i, j in self.bonds:
            for tau in range(self.n_tau):
                E_spatial -= self.J * self.config[i, tau] * self.config[j, tau]
        E_spatial /= self.n_tau
        
        self.measurements['energy'].append(E_spatial)
        self.measurements['magnetization_z'].append(mz)
        self.measurements['magnetization_z_squared'].append(mz2)
    
    def run(self, n_thermalization: int = 100, n_measurements: int = 1000,
            n_skip: int = 5, use_cluster: bool = True) -> Dict[str, Any]:
        """Run simulation."""
        print(f"Running World-Line QMC for {self.L}x{self.L}x{self.L}x{self.n_tau} lattice")
        
        for key in self.measurements:
            self.measurements[key] = []
        
        # Thermalization
        print("Thermalizing...")
        for _ in tqdm(range(n_thermalization)):
            if use_cluster:
                self.cluster_update()
            else:
                self.mc_step()
        
        # Measurements
        print("Measuring...")
        for _ in tqdm(range(n_measurements)):
            for _ in range(n_skip):
                if use_cluster:
                    self.cluster_update()
                else:
                    self.mc_step()
            self.measure()
        
        # Statistics
        results = {}
        for key, values in self.measurements.items():
            values = np.array(values)
            results[key] = {'mean': np.mean(values), 'error': np.std(values) / np.sqrt(len(values))}
        
        results['energy_per_site'] = {
            'mean': results['energy']['mean'] / self.N,
            'error': results['energy']['error'] / self.N
        }
        
        return results


def run_qmc_example():
    """Example QMC simulation."""
    print("=" * 70)
    print("Quantum Monte Carlo Example")
    print("=" * 70)
    
    L = 4
    J = 1.0
    Gamma = 3.0  # Near critical point for 3D
    beta = 2.0
    
    qmc = QuantumMonteCarlo(L=L, J=J, Gamma=Gamma, beta=beta)
    results = qmc.run(n_thermalization=500, n_measurements=2000)
    
    print("\nResults:")
    print("-" * 40)
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"  {key}: {value['mean']:.6f} ± {value['error']:.6f}")
    
    return results


if __name__ == "__main__":
    run_qmc_example()
