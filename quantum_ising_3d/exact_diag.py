"""
Exact Diagonalization Solver for 3D Transverse Field Ising Model

Uses sparse matrix diagonalization (Lanczos/Arnoldi) to find
ground state and low-lying excitations.

Practical limits:
- L=2 (8 sites, 256 states): instant
- L=3 (27 sites, ~134M states): requires specialized techniques
- For larger systems, use Quantum Monte Carlo
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, LinearOperator
from typing import Tuple, List, Optional, Dict, Any
import time

from .hamiltonian import TransverseFieldIsing3D, TransverseFieldIsing1D


class ExactDiagonalization:
    """
    Exact diagonalization solver for quantum Ising models.
    
    Uses sparse Lanczos algorithm to find ground state and
    low-lying excited states efficiently.
    
    Parameters
    ----------
    model : TransverseFieldIsing3D or TransverseFieldIsing1D
        The quantum Ising model to solve
    """
    
    def __init__(self, model):
        self.model = model
        self.H = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.ground_state_energy = None
        self.ground_state = None
        
    def build_hamiltonian(self, use_fast: bool = True):
        """
        Build the Hamiltonian matrix.
        
        Parameters
        ----------
        use_fast : bool
            Use fast bit-manipulation method (recommended for larger systems)
        """
        if hasattr(self.model, 'build_hamiltonian_fast') and use_fast:
            self.H = self.model.build_hamiltonian_fast()
        else:
            self.H = self.model.build_hamiltonian()
        return self.H
    
    def diagonalize(self, n_states: int = 1, which: str = 'SA',
                    tol: float = 1e-10, maxiter: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find lowest eigenvalues and eigenvectors using Lanczos algorithm.
        
        Parameters
        ----------
        n_states : int
            Number of eigenstates to compute
        which : str
            Which eigenvalues to find: 'SA' (smallest algebraic), 'LA' (largest)
        tol : float
            Convergence tolerance
        maxiter : int
            Maximum iterations
            
        Returns
        -------
        eigenvalues : np.ndarray
            Array of eigenvalues
        eigenvectors : np.ndarray
            Matrix of eigenvectors (columns)
        """
        if self.H is None:
            self.build_hamiltonian()
        
        print(f"Diagonalizing {self.H.shape[0]} x {self.H.shape[0]} matrix...")
        start_time = time.time()
        
        # For very small systems, use full diagonalization
        if self.model.dim <= 512:
            print("  Using full diagonalization...")
            H_dense = self.H.toarray()
            eigenvalues, eigenvectors = np.linalg.eigh(H_dense.real)
            idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx[:n_states]]
            eigenvectors = eigenvectors[:, idx[:n_states]]
        else:
            print("  Using sparse Lanczos algorithm...")
            eigenvalues, eigenvectors = eigsh(self.H, k=n_states, which=which,
                                              tol=tol, maxiter=maxiter)
            # Sort by eigenvalue
            idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
        
        elapsed = time.time() - start_time
        print(f"  Completed in {elapsed:.2f} seconds")
        
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.ground_state_energy = eigenvalues[0]
        self.ground_state = eigenvectors[:, 0]
        
        return eigenvalues, eigenvectors
    
    def get_ground_state(self) -> Tuple[float, np.ndarray]:
        """
        Get ground state energy and wavefunction.
        
        Returns
        -------
        E0 : float
            Ground state energy
        psi0 : np.ndarray
            Ground state wavefunction
        """
        if self.ground_state is None:
            self.diagonalize(n_states=1)
        return self.ground_state_energy, self.ground_state
    
    def compute_expectation_value(self, operator: sparse.csr_matrix,
                                   state: Optional[np.ndarray] = None) -> complex:
        """
        Compute expectation value <ψ|O|ψ>.
        
        Parameters
        ----------
        operator : sparse.csr_matrix
            Operator O
        state : np.ndarray, optional
            State vector (defaults to ground state)
            
        Returns
        -------
        complex
            Expectation value
        """
        if state is None:
            if self.ground_state is None:
                self.diagonalize(n_states=1)
            state = self.ground_state
        
        return np.vdot(state, operator @ state)
    
    def compute_magnetization(self, direction: str = 'z',
                              state: Optional[np.ndarray] = None) -> float:
        """
        Compute magnetization per site ⟨M⟩ = (1/N)⟨Σ_i σ_i^α⟩.
        
        Parameters
        ----------
        direction : str
            'x', 'y', or 'z'
        state : np.ndarray, optional
            State vector (defaults to ground state)
            
        Returns
        -------
        float
            Magnetization per site
        """
        M_op = self.model.get_magnetization_operator(direction)
        return np.real(self.compute_expectation_value(M_op, state))
    
    def compute_magnetization_squared(self, direction: str = 'z',
                                       state: Optional[np.ndarray] = None) -> float:
        """
        Compute ⟨M²⟩ = (1/N²)⟨(Σ_i σ_i^α)²⟩.
        """
        M_op = self.model.get_magnetization_operator(direction)
        M2_op = M_op @ M_op
        return np.real(self.compute_expectation_value(M2_op, state))
    
    def compute_energy_per_site(self) -> float:
        """Compute ground state energy per site."""
        if self.ground_state_energy is None:
            self.diagonalize(n_states=1)
        return self.ground_state_energy / self.model.N
    
    def compute_correlation_function(self, site1: int, site2: int,
                                       direction: str = 'z',
                                       state: Optional[np.ndarray] = None) -> float:
        """
        Compute two-point correlation ⟨σ_i^α σ_j^α⟩.
        """
        C_op = self.model.get_correlation_operator(site1, site2, direction)
        return np.real(self.compute_expectation_value(C_op, state))
    
    def compute_connected_correlation(self, site1: int, site2: int,
                                        direction: str = 'z',
                                        state: Optional[np.ndarray] = None) -> float:
        """
        Compute connected correlation ⟨σ_i^α σ_j^α⟩ - ⟨σ_i^α⟩⟨σ_j^α⟩.
        """
        from .hamiltonian import pauli_z, pauli_x, pauli_y
        
        if direction == 'x':
            sigma = pauli_x()
        elif direction == 'y':
            sigma = pauli_y()
        else:
            sigma = pauli_z()
        
        sigma_i = self.model._single_site_operator(sigma, site1)
        sigma_j = self.model._single_site_operator(sigma, site2)
        
        if state is None:
            if self.ground_state is None:
                self.diagonalize(n_states=1)
            state = self.ground_state
        
        sigma_ij = np.real(self.compute_expectation_value(sigma_i @ sigma_j, state))
        sigma_i_exp = np.real(self.compute_expectation_value(sigma_i, state))
        sigma_j_exp = np.real(self.compute_expectation_value(sigma_j, state))
        
        return sigma_ij - sigma_i_exp * sigma_j_exp
    
    def compute_energy_gap(self) -> float:
        """Compute gap between ground and first excited state."""
        if self.eigenvalues is None or len(self.eigenvalues) < 2:
            self.diagonalize(n_states=2)
        return self.eigenvalues[1] - self.eigenvalues[0]
    
    def compute_susceptibility(self, beta: float = 1.0,
                                direction: str = 'z') -> float:
        """
        Compute magnetic susceptibility χ = β N (⟨M²⟩ - ⟨M⟩²).
        
        Parameters
        ----------
        beta : float
            Inverse temperature (for finite T, would need thermal average)
        direction : str
            Magnetization direction
            
        Returns
        -------
        float
            Magnetic susceptibility
        """
        M = self.compute_magnetization(direction)
        M2 = self.compute_magnetization_squared(direction)
        return beta * self.model.N * (M2 - M**2)
    
    def analyze_ground_state(self) -> Dict[str, Any]:
        """
        Comprehensive analysis of the ground state.
        
        Returns dictionary with:
        - energy: ground state energy
        - energy_per_site: energy per site
        - gap: energy gap to first excited state
        - magnetization_z: ⟨M_z⟩
        - magnetization_x: ⟨M_x⟩
        - susceptibility_z: χ_z
        """
        print("Analyzing ground state...")
        
        # Ensure we have at least 2 eigenstates
        self.diagonalize(n_states=min(4, self.model.dim))
        
        results = {
            'energy': self.ground_state_energy,
            'energy_per_site': self.ground_state_energy / self.model.N,
            'gap': self.eigenvalues[1] - self.eigenvalues[0] if len(self.eigenvalues) > 1 else 0,
            'magnetization_z': self.compute_magnetization('z'),
            'magnetization_x': self.compute_magnetization('x'),
            'magnetization_z_squared': self.compute_magnetization_squared('z'),
            'magnetization_x_squared': self.compute_magnetization_squared('x'),
        }
        
        # Susceptibilities
        results['susceptibility_z'] = self.model.N * (
            results['magnetization_z_squared'] - results['magnetization_z']**2)
        results['susceptibility_x'] = self.model.N * (
            results['magnetization_x_squared'] - results['magnetization_x']**2)
        
        return results
    
    def phase_diagram_scan(self, Gamma_values: np.ndarray,
                            observables: List[str] = ['energy', 'magnetization_z', 'gap']
                            ) -> Dict[str, np.ndarray]:
        """
        Scan over Γ values to map out phase diagram.
        
        Parameters
        ----------
        Gamma_values : np.ndarray
            Array of Γ values to scan
        observables : List[str]
            Which observables to compute
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary mapping observable names to arrays of values
        """
        results = {obs: [] for obs in observables}
        results['Gamma'] = Gamma_values
        
        J_original = self.model.J
        
        print(f"Scanning {len(Gamma_values)} Γ values...")
        
        for i, Gamma in enumerate(Gamma_values):
            print(f"  Γ = {Gamma:.4f} ({i+1}/{len(Gamma_values)})")
            
            # Update model parameters
            self.model.Gamma = Gamma
            self.H = None  # Force rebuild
            self.eigenvalues = None
            self.ground_state = None
            
            # Build and diagonalize
            self.build_hamiltonian()
            self.diagonalize(n_states=2)
            
            # Compute requested observables
            for obs in observables:
                if obs == 'energy':
                    results[obs].append(self.ground_state_energy)
                elif obs == 'energy_per_site':
                    results[obs].append(self.ground_state_energy / self.model.N)
                elif obs == 'magnetization_z':
                    results[obs].append(self.compute_magnetization('z'))
                elif obs == 'magnetization_x':
                    results[obs].append(self.compute_magnetization('x'))
                elif obs == 'magnetization_z_squared':
                    results[obs].append(self.compute_magnetization_squared('z'))
                elif obs == 'magnetization_x_squared':
                    results[obs].append(self.compute_magnetization_squared('x'))
                elif obs == 'gap':
                    results[obs].append(self.eigenvalues[1] - self.eigenvalues[0])
        
        # Convert to arrays
        for key in results:
            results[key] = np.array(results[key])
        
        return results
    
    def compute_entanglement_entropy(self, subsystem_sites: List[int],
                                      state: Optional[np.ndarray] = None) -> float:
        """
        Compute von Neumann entanglement entropy for a subsystem.
        
        S = -Tr(ρ_A log ρ_A)
        
        Parameters
        ----------
        subsystem_sites : List[int]
            Sites in subsystem A
        state : np.ndarray, optional
            State vector (defaults to ground state)
            
        Returns
        -------
        float
            Entanglement entropy
        """
        if state is None:
            if self.ground_state is None:
                self.diagonalize(n_states=1)
            state = self.ground_state
        
        N = self.model.N
        N_A = len(subsystem_sites)
        N_B = N - N_A
        
        # Create subsystem B sites
        subsystem_B = [i for i in range(N) if i not in subsystem_sites]
        
        # Reshape state to bipartite form
        # Need to reorder indices: A sites first, then B sites
        all_sites = subsystem_sites + subsystem_B
        
        # Create permutation matrix for reordering
        dim_A = 2 ** N_A
        dim_B = 2 ** N_B
        
        # Reshape and compute reduced density matrix
        # This is a simplified version - full implementation would need
        # proper index permutation
        psi = state.reshape([2] * N)
        
        # Move A sites to front
        for new_pos, old_pos in enumerate(subsystem_sites):
            if new_pos != old_pos:
                psi = np.moveaxis(psi, old_pos, new_pos)
        
        # Reshape to (dim_A, dim_B)
        psi = psi.reshape(dim_A, dim_B)
        
        # Reduced density matrix for A
        rho_A = psi @ psi.conj().T
        
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(rho_A)
        eigenvalues = eigenvalues[eigenvalues > 1e-15]  # Remove numerical zeros
        
        # Von Neumann entropy
        entropy = -np.sum(eigenvalues * np.log(eigenvalues))
        
        return entropy


def run_exact_diag_example():
    """Example usage of exact diagonalization."""
    print("=" * 70)
    print("Exact Diagonalization Example")
    print("=" * 70)
    
    # Create 2x2x2 lattice
    L = 2
    J = 1.0
    Gamma = 1.0
    
    model = TransverseFieldIsing3D(L=L, J=J, Gamma=Gamma)
    model.analyze_system()
    
    solver = ExactDiagonalization(model)
    results = solver.analyze_ground_state()
    
    print("\nGround State Properties:")
    print("-" * 40)
    for key, value in results.items():
        print(f"  {key}: {value:.6f}")
    
    # Phase diagram
    print("\n" + "=" * 70)
    print("Phase Diagram Scan")
    print("=" * 70)
    
    Gamma_values = np.linspace(0.1, 5.0, 20)
    phase_data = solver.phase_diagram_scan(
        Gamma_values, 
        observables=['energy_per_site', 'magnetization_z', 'magnetization_x', 'gap']
    )
    
    print("\nΓ/J\t\tE/site\t\t⟨M_z²⟩^½\t⟨M_x⟩\t\tΔ")
    print("-" * 70)
    for i in range(len(Gamma_values)):
        print(f"{phase_data['Gamma'][i]:.3f}\t\t"
              f"{phase_data['energy_per_site'][i]:.4f}\t\t"
              f"{np.sqrt(abs(phase_data['magnetization_z'][i])):.4f}\t\t"
              f"{phase_data['magnetization_x'][i]:.4f}\t\t"
              f"{phase_data['gap'][i]:.4f}")
    
    return phase_data


if __name__ == "__main__":
    run_exact_diag_example()
