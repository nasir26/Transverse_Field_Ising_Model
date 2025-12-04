"""
Quantum Annealing Simulation for 3D Transverse Field Ising Model

Implements time-dependent quantum evolution:
    H(t) = -J Σ_{<i,j>} σ_i^z σ_j^z - Γ(t) Σ_i σ_i^x

Starting with large Γ(0) (easy quantum ground state) and slowly
decreasing Γ(t) → 0, the system can find the ground state of the
classical Ising problem.

Methods implemented:
1. Exact time evolution (for small systems)
2. Simulated quantum annealing (Path Integral Monte Carlo)
3. Adiabatic theorem analysis
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply, eigsh
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from typing import Tuple, List, Optional, Dict, Any, Callable
import time
from tqdm import tqdm

from .hamiltonian import TransverseFieldIsing3D, pauli_z, pauli_x


class QuantumAnnealing:
    """
    Quantum Annealing simulation for finding ground states.
    
    Parameters
    ----------
    model : TransverseFieldIsing3D
        The Ising model (J defines the problem Hamiltonian)
    """
    
    def __init__(self, model: TransverseFieldIsing3D):
        self.model = model
        self.N = model.N
        self.dim = model.dim
        self.J = model.J
        
        # Build problem Hamiltonian (Ising part only)
        self._build_problem_hamiltonian()
        
        # Build transverse field operator
        self._build_field_operator()
        
    def _build_problem_hamiltonian(self):
        """Build H_problem = -J Σ σ_i^z σ_j^z (diagonal in computational basis)."""
        print("Building problem Hamiltonian...")
        
        bonds = self.model.get_bonds()
        diag = np.zeros(self.dim)
        
        for s in range(self.dim):
            for i, j in bonds:
                si = 1 - 2 * ((s >> i) & 1)
                sj = 1 - 2 * ((s >> j) & 1)
                diag[s] -= self.J * si * sj
        
        self.H_problem = sparse.diags(diag, format='csr')
        
        # Find classical ground state
        self.classical_gs_idx = np.argmin(diag)
        self.classical_gs_energy = diag[self.classical_gs_idx]
        
        print(f"  Classical ground state energy: {self.classical_gs_energy:.4f}")
        
    def _build_field_operator(self):
        """Build H_field = -Σ σ_i^x (off-diagonal, uniform field)."""
        print("Building field operator...")
        
        from scipy.sparse import lil_matrix
        H = lil_matrix((self.dim, self.dim), dtype=float)
        
        for s in range(self.dim):
            for i in range(self.N):
                s_flip = s ^ (1 << i)
                H[s, s_flip] = -1.0
        
        self.H_field = H.tocsr()
    
    def get_hamiltonian(self, Gamma: float) -> sparse.csr_matrix:
        """Get full Hamiltonian for given transverse field strength."""
        return self.H_problem + Gamma * self.H_field
    
    def annealing_schedule_linear(self, t: float, t_anneal: float,
                                   Gamma_0: float = 10.0) -> float:
        """
        Linear annealing schedule.
        
        Γ(t) = Γ_0 * (1 - t/t_anneal)
        """
        s = t / t_anneal
        return Gamma_0 * (1 - s)
    
    def annealing_schedule_exponential(self, t: float, t_anneal: float,
                                        Gamma_0: float = 10.0,
                                        Gamma_min: float = 0.01) -> float:
        """
        Exponential annealing schedule.
        
        Γ(t) = Γ_0 * exp(-t * ln(Γ_0/Γ_min) / t_anneal)
        """
        rate = np.log(Gamma_0 / Gamma_min) / t_anneal
        return Gamma_0 * np.exp(-rate * t)
    
    def annealing_schedule_polynomial(self, t: float, t_anneal: float,
                                       Gamma_0: float = 10.0, power: float = 2.0) -> float:
        """
        Polynomial annealing schedule.
        
        Γ(t) = Γ_0 * (1 - t/t_anneal)^power
        """
        s = t / t_anneal
        return Gamma_0 * (1 - s) ** power
    
    def run_exact_annealing(self, t_anneal: float, n_steps: int = 100,
                             Gamma_0: float = 10.0,
                             schedule: str = 'linear') -> Dict[str, Any]:
        """
        Run exact quantum annealing using time-dependent Schrödinger evolution.
        
        Uses Trotter decomposition for time evolution:
            |ψ(t+dt)⟩ = exp(-i H(t) dt) |ψ(t)⟩
        
        Parameters
        ----------
        t_anneal : float
            Total annealing time
        n_steps : int
            Number of time steps
        Gamma_0 : float
            Initial transverse field strength
        schedule : str
            'linear', 'exponential', or 'polynomial'
            
        Returns
        -------
        Dict with annealing results
        """
        print(f"Running exact quantum annealing (t_anneal = {t_anneal})")
        print(f"  System size: {self.dim} states")
        
        dt = t_anneal / n_steps
        
        # Select schedule
        if schedule == 'exponential':
            sched_func = self.annealing_schedule_exponential
        elif schedule == 'polynomial':
            sched_func = self.annealing_schedule_polynomial
        else:
            sched_func = self.annealing_schedule_linear
        
        # Initial state: ground state of large transverse field
        # This is |+⟩^⊗N = (1/√2^N) Σ_s |s⟩
        psi = np.ones(self.dim, dtype=complex) / np.sqrt(self.dim)
        
        # History
        times = []
        Gamma_history = []
        energy_history = []
        overlap_history = []
        magnetization_history = []
        
        start_time = time.time()
        
        # Time evolution
        for step in tqdm(range(n_steps + 1)):
            t = step * dt
            Gamma = sched_func(t, t_anneal, Gamma_0)
            
            # Current Hamiltonian
            H = self.get_hamiltonian(Gamma)
            
            # Measure
            energy = np.real(np.vdot(psi, H @ psi))
            
            # Overlap with classical ground state
            overlap = np.abs(psi[self.classical_gs_idx]) ** 2
            
            # Magnetization
            mz = 0.0
            for s in range(self.dim):
                prob = np.abs(psi[s]) ** 2
                # Sum of σ^z values
                m = sum(1 - 2 * ((s >> i) & 1) for i in range(self.N))
                mz += prob * m / self.N
            
            times.append(t)
            Gamma_history.append(Gamma)
            energy_history.append(energy)
            overlap_history.append(overlap)
            magnetization_history.append(mz)
            
            # Evolve (except last step)
            if step < n_steps:
                # Time evolution operator (using sparse expm_multiply)
                psi = expm_multiply(-1j * H * dt, psi)
                psi /= np.linalg.norm(psi)  # Renormalize for numerical stability
        
        elapsed = time.time() - start_time
        
        results = {
            'times': np.array(times),
            'Gamma': np.array(Gamma_history),
            'energy': np.array(energy_history),
            'overlap_gs': np.array(overlap_history),
            'magnetization_z': np.array(magnetization_history),
            'final_state': psi,
            'success_probability': overlap_history[-1],
            'final_energy': energy_history[-1],
            'classical_gs_energy': self.classical_gs_energy,
            'elapsed_time': elapsed
        }
        
        print(f"\nCompleted in {elapsed:.2f} seconds")
        print(f"Success probability: {results['success_probability']:.4f}")
        print(f"Final energy: {results['final_energy']:.4f}")
        print(f"Classical ground state energy: {self.classical_gs_energy:.4f}")
        
        return results
    
    def compute_minimum_gap(self, Gamma_values: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Compute energy gap as function of Γ to find minimum gap.
        
        The adiabatic theorem requires t_anneal >> 1/Δ² where Δ is the minimum gap.
        
        Returns
        -------
        gaps : np.ndarray
            Energy gaps for each Γ value
        min_gap : float
            Minimum gap value
        Gamma_critical : float
            Γ value at minimum gap
        """
        print("Computing energy gap spectrum...")
        
        gaps = []
        
        for Gamma in tqdm(Gamma_values):
            H = self.get_hamiltonian(Gamma)
            
            # Get two lowest eigenvalues
            if self.dim <= 512:
                eigenvalues = np.linalg.eigvalsh(H.toarray())
                gap = eigenvalues[1] - eigenvalues[0]
            else:
                eigenvalues, _ = eigsh(H, k=2, which='SA')
                eigenvalues = np.sort(eigenvalues)
                gap = eigenvalues[1] - eigenvalues[0]
            
            gaps.append(gap)
        
        gaps = np.array(gaps)
        min_idx = np.argmin(gaps)
        min_gap = gaps[min_idx]
        Gamma_critical = Gamma_values[min_idx]
        
        print(f"Minimum gap: {min_gap:.6f} at Γ = {Gamma_critical:.4f}")
        
        return gaps, min_gap, Gamma_critical
    
    def adiabatic_condition(self, t_anneal: float, min_gap: float) -> float:
        """
        Evaluate adiabatic condition.
        
        For successful annealing, need:
            t_anneal >> max |dH/dt| / Δ²
        
        Returns ratio of t_anneal to minimum required time.
        """
        # For linear schedule, |dΓ/dt| = Γ_0 / t_anneal
        # The transition element is of order N (all spins can flip)
        Gamma_0 = 10.0
        dGamma_dt = Gamma_0 / t_anneal
        dH_dt_max = self.N * dGamma_dt  # Order of magnitude
        
        t_min = dH_dt_max / (min_gap ** 2)
        
        return t_anneal / t_min
    
    def study_annealing_time_dependence(self, t_anneal_values: np.ndarray,
                                         n_steps: int = 100,
                                         schedule: str = 'linear') -> Dict[str, np.ndarray]:
        """
        Study how success probability depends on annealing time.
        
        Returns
        -------
        Dict with success probabilities and final energies for each t_anneal
        """
        print(f"Studying annealing time dependence ({len(t_anneal_values)} values)")
        
        results = {
            't_anneal': t_anneal_values,
            'success_probability': [],
            'final_energy': [],
            'energy_error': []
        }
        
        for t_anneal in t_anneal_values:
            print(f"\n  t_anneal = {t_anneal:.2f}")
            data = self.run_exact_annealing(t_anneal, n_steps, schedule=schedule)
            
            results['success_probability'].append(data['success_probability'])
            results['final_energy'].append(data['final_energy'])
            results['energy_error'].append(data['final_energy'] - self.classical_gs_energy)
        
        for key in ['success_probability', 'final_energy', 'energy_error']:
            results[key] = np.array(results[key])
        
        return results


class SimulatedQuantumAnnealing:
    """
    Simulated Quantum Annealing using classical Monte Carlo.
    
    Uses Suzuki-Trotter mapping to simulate quantum annealing
    with a classical (d+1)-dimensional system.
    
    This is practical for larger systems where exact evolution is infeasible.
    
    Parameters
    ----------
    L : int
        Linear lattice size (L x L x L for 3D)
    J : float
        Ising coupling
    n_replicas : int
        Number of Trotter replicas
    """
    
    def __init__(self, L: int, J: float = 1.0, n_replicas: int = 20):
        self.L = L
        self.N = L ** 3
        self.J = J
        self.n_replicas = n_replicas
        
        # Build bonds
        self.bonds = self._build_bonds()
        
        # Configuration: (N, n_replicas) array of ±1 spins
        self.config = np.random.choice([-1, 1], size=(self.N, n_replicas))
        
    def _build_bonds(self) -> List[Tuple[int, int]]:
        """Build nearest neighbor bonds."""
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
    
    def _local_energy(self, i: int, r: int, Gamma: float, beta: float) -> float:
        """
        Compute local energy for spin (i, r).
        
        E = -J Σ_j s_i s_j (spatial) - J_tau s_i(r) s_i(r±1) (temporal)
        
        where J_tau = -(1/2β) ln(tanh(Γβ/n_replicas))
        """
        s = self.config[i, r]
        
        # Spatial neighbors
        E_spatial = 0.0
        L = self.L
        x = i % L
        y = (i // L) % L
        z = i // (L * L)
        
        for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
            nx = (x + dx) % L
            ny = (y + dy) % L
            nz = (z + dz) % L
            neighbor = nx + L * ny + L * L * nz
            E_spatial -= (self.J / self.n_replicas) * s * self.config[neighbor, r]
        
        # Temporal neighbors (Trotter coupling)
        if Gamma > 0:
            J_tau = -0.5 / beta * np.log(np.tanh(Gamma * beta / self.n_replicas))
            r_prev = (r - 1) % self.n_replicas
            r_next = (r + 1) % self.n_replicas
            E_temporal = -J_tau * s * (self.config[i, r_prev] + self.config[i, r_next])
        else:
            E_temporal = 0.0
        
        return E_spatial + E_temporal
    
    def mc_sweep(self, Gamma: float, beta: float):
        """Perform one Metropolis sweep."""
        for _ in range(self.N * self.n_replicas):
            i = np.random.randint(self.N)
            r = np.random.randint(self.n_replicas)
            
            dE = -2 * self._local_energy(i, r, Gamma, beta)
            
            if dE < 0 or np.random.random() < np.exp(-beta * dE):
                self.config[i, r] *= -1
    
    def compute_ising_energy(self) -> float:
        """Compute classical Ising energy (averaged over replicas)."""
        E = 0.0
        for i, j in self.bonds:
            for r in range(self.n_replicas):
                E -= self.J * self.config[i, r] * self.config[j, r]
        return E / self.n_replicas
    
    def run_annealing(self, beta: float = 10.0, Gamma_0: float = 10.0,
                       n_steps: int = 1000, mc_sweeps_per_step: int = 10,
                       schedule: str = 'linear') -> Dict[str, Any]:
        """
        Run simulated quantum annealing.
        
        Parameters
        ----------
        beta : float
            Inverse temperature
        Gamma_0 : float
            Initial transverse field
        n_steps : int
            Number of annealing steps
        mc_sweeps_per_step : int
            MC sweeps at each step
        schedule : str
            Annealing schedule
            
        Returns
        -------
        Dict with annealing results
        """
        print(f"Running simulated quantum annealing (L={self.L}, n_replicas={self.n_replicas})")
        
        # Initialize random configuration
        self.config = np.random.choice([-1, 1], size=(self.N, self.n_replicas))
        
        Gamma_history = []
        energy_history = []
        
        for step in tqdm(range(n_steps + 1)):
            s = step / n_steps
            
            if schedule == 'linear':
                Gamma = Gamma_0 * (1 - s)
            elif schedule == 'exponential':
                Gamma = Gamma_0 * np.exp(-5 * s)
            else:
                Gamma = Gamma_0 * (1 - s) ** 2
            
            # MC sweeps at this Gamma
            for _ in range(mc_sweeps_per_step):
                self.mc_sweep(Gamma, beta)
            
            # Measure
            energy = self.compute_ising_energy()
            Gamma_history.append(Gamma)
            energy_history.append(energy)
        
        # Final configuration (majority vote across replicas)
        final_config = np.sign(np.sum(self.config, axis=1))
        final_config[final_config == 0] = 1
        
        # Compute final classical energy
        E_final = 0.0
        for i, j in self.bonds:
            E_final -= self.J * final_config[i] * final_config[j]
        
        results = {
            'Gamma': np.array(Gamma_history),
            'energy': np.array(energy_history),
            'final_config': final_config,
            'final_energy': E_final,
            'magnetization': np.mean(final_config)
        }
        
        print(f"Final energy: {E_final:.4f}")
        print(f"Final magnetization: {results['magnetization']:.4f}")
        
        return results


def run_annealing_example():
    """Example quantum annealing simulation."""
    print("=" * 70)
    print("Quantum Annealing Example")
    print("=" * 70)
    
    # Small system for exact annealing
    L = 2
    model = TransverseFieldIsing3D(L=L, J=1.0, Gamma=0.0)
    
    qa = QuantumAnnealing(model)
    
    # Compute minimum gap
    Gamma_values = np.linspace(0.1, 10.0, 50)
    gaps, min_gap, Gamma_c = qa.compute_minimum_gap(Gamma_values)
    
    print(f"\nMinimum gap: {min_gap:.6f} at Γ = {Gamma_c:.4f}")
    
    # Run annealing for different times
    print("\nRunning annealing for different times...")
    results = qa.run_exact_annealing(t_anneal=10.0, n_steps=200, schedule='linear')
    
    print("\nAnnealing Results:")
    print(f"  Success probability: {results['success_probability']:.4f}")
    print(f"  Final energy: {results['final_energy']:.6f}")
    print(f"  Classical GS energy: {results['classical_gs_energy']:.6f}")
    
    return results


def run_sqa_example():
    """Example simulated quantum annealing."""
    print("\n" + "=" * 70)
    print("Simulated Quantum Annealing Example")
    print("=" * 70)
    
    L = 4
    sqa = SimulatedQuantumAnnealing(L=L, J=1.0, n_replicas=20)
    
    results = sqa.run_annealing(beta=10.0, Gamma_0=10.0, n_steps=500)
    
    return results


if __name__ == "__main__":
    run_annealing_example()
    run_sqa_example()
