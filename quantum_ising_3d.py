"""
Complete Quantum Simulation of 3D Transverse Field Ising Model

Implements:
- Exact diagonalization for small systems
- Quantum phase transition analysis
- Quantum annealing protocol
- Ground state properties
- Various observables (magnetization, energy, correlations)
"""

import numpy as np
from scipy.sparse import csr_matrix, kron, eye, diags
from scipy.sparse.linalg import eigsh
from scipy.linalg import expm
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import time


class QuantumIsing3D:
    """
    3D Transverse Field Ising Model Quantum Simulator
    
    Hamiltonian: H = -J * Σ_<i,j> σ_i^z σ_j^z - Γ * Σ_i σ_i^x
    
    Parameters:
    -----------
    Lx, Ly, Lz : int
        Lattice dimensions (total N = Lx * Ly * Lz spins)
    J : float
        Coupling strength (positive for ferromagnetic)
    Gamma : float
        Transverse field strength
    periodic : bool
        Whether to use periodic boundary conditions
    """
    
    def __init__(self, Lx: int, Ly: int, Lz: int, J: float = 1.0, 
                 Gamma: float = 0.5, periodic: bool = False):
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.N = Lx * Ly * Lz  # Total number of spins
        self.J = J
        self.Gamma = Gamma
        self.periodic = periodic
        
        # Check feasibility
        if self.N > 20:
            print(f"Warning: System size N={self.N} is large. Hilbert space dimension = 2^{self.N}")
            print(f"This may require significant memory (~{2**(self.N-27):.2f} GB for the Hamiltonian)")
            
        self.dim = 2**self.N  # Hilbert space dimension
        
        # Pauli matrices (sparse)
        self.sigma_x = csr_matrix([[0, 1], [1, 0]], dtype=np.float64)
        self.sigma_z = csr_matrix([[1, 0], [0, -1]], dtype=np.float64)
        self.I = csr_matrix([[1, 0], [0, 1]], dtype=np.float64)
        
        # Cached Hamiltonian
        self._H = None
        self._eigenvalues = None
        self._eigenvectors = None
        
    def _index_to_coords(self, idx: int) -> Tuple[int, int, int]:
        """Convert linear index to 3D coordinates"""
        x = idx % self.Lx
        y = (idx // self.Lx) % self.Ly
        z = idx // (self.Lx * self.Ly)
        return x, y, z
    
    def _coords_to_index(self, x: int, y: int, z: int) -> int:
        """Convert 3D coordinates to linear index"""
        return x + y * self.Lx + z * self.Lx * self.Ly
    
    def _get_neighbors(self, idx: int) -> List[int]:
        """Get nearest neighbors in 3D lattice (all directions)"""
        x, y, z = self._index_to_coords(idx)
        neighbors = []
        
        # X direction (both + and -)
        if x < self.Lx - 1:
            neighbors.append(self._coords_to_index(x + 1, y, z))
        elif self.periodic:
            neighbors.append(self._coords_to_index(0, y, z))
        
        if x > 0:
            neighbors.append(self._coords_to_index(x - 1, y, z))
        elif self.periodic:
            neighbors.append(self._coords_to_index(self.Lx - 1, y, z))
            
        # Y direction (both + and -)
        if y < self.Ly - 1:
            neighbors.append(self._coords_to_index(x, y + 1, z))
        elif self.periodic:
            neighbors.append(self._coords_to_index(x, 0, z))
        
        if y > 0:
            neighbors.append(self._coords_to_index(x, y - 1, z))
        elif self.periodic:
            neighbors.append(self._coords_to_index(x, self.Ly - 1, z))
            
        # Z direction (both + and -)
        if z < self.Lz - 1:
            neighbors.append(self._coords_to_index(x, y, z + 1))
        elif self.periodic:
            neighbors.append(self._coords_to_index(x, y, 0))
        
        if z > 0:
            neighbors.append(self._coords_to_index(x, y, z - 1))
        elif self.periodic:
            neighbors.append(self._coords_to_index(x, y, self.Lz - 1))
            
        return neighbors
    
    def _single_site_operator(self, site: int, op: csr_matrix) -> csr_matrix:
        """
        Construct operator acting on single site in full Hilbert space
        
        O_total = I ⊗ I ⊗ ... ⊗ O_site ⊗ ... ⊗ I
        """
        result = self.I if site > 0 else op
        
        for i in range(1, self.N):
            if i == site:
                result = kron(result, op, format='csr')
            else:
                result = kron(result, self.I, format='csr')
                
        return result
    
    def _two_site_operator(self, site1: int, site2: int, 
                           op1: csr_matrix, op2: csr_matrix) -> csr_matrix:
        """
        Construct operator acting on two sites
        
        O_total = I ⊗ ... ⊗ O_site1 ⊗ ... ⊗ O_site2 ⊗ ... ⊗ I
        """
        result = None
        
        for i in range(self.N):
            if i == site1:
                op = op1
            elif i == site2:
                op = op2
            else:
                op = self.I
                
            if result is None:
                result = op
            else:
                result = kron(result, op, format='csr')
                
        return result
    
    def construct_hamiltonian(self, J: Optional[float] = None, 
                             Gamma: Optional[float] = None) -> csr_matrix:
        """
        Construct the full Hamiltonian matrix
        
        H = -J * Σ_<i,j> σ_i^z σ_j^z - Γ * Σ_i σ_i^x
        
        Returns sparse CSR matrix of dimension (2^N, 2^N)
        """
        if J is None:
            J = self.J
        if Gamma is None:
            Gamma = self.Gamma
            
        print(f"Constructing Hamiltonian for {self.N} spins ({self.Lx}x{self.Ly}x{self.Lz})...")
        print(f"Hilbert space dimension: {self.dim}")
        start_time = time.time()
        
        H = csr_matrix((self.dim, self.dim), dtype=np.float64)
        
        # Interaction term: -J * Σ_<i,j> σ_i^z σ_j^z
        print("Adding interaction terms...")
        interaction_count = 0
        for i in range(self.N):
            neighbors = self._get_neighbors(i)
            for j in neighbors:
                if j > i:  # Count each pair once
                    H = H - J * self._two_site_operator(i, j, self.sigma_z, self.sigma_z)
                    interaction_count += 1
                    
        print(f"  Added {interaction_count} interaction terms")
        
        # Transverse field term: -Γ * Σ_i σ_i^x
        print("Adding transverse field terms...")
        for i in range(self.N):
            H = H - Gamma * self._single_site_operator(i, self.sigma_x)
            
        print(f"  Added {self.N} transverse field terms")
        
        elapsed = time.time() - start_time
        print(f"Hamiltonian constructed in {elapsed:.2f} seconds")
        print(f"Matrix size: {H.shape}, Non-zero elements: {H.nnz}")
        
        self._H = H
        return H
    
    def get_hamiltonian(self) -> csr_matrix:
        """Get cached Hamiltonian or construct if not available"""
        if self._H is None:
            self.construct_hamiltonian()
        return self._H
    
    def find_ground_state(self, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k lowest energy eigenstates using sparse eigenvalue solver
        
        Returns:
        --------
        eigenvalues : array of shape (k,)
            Lowest k eigenvalues (energies)
        eigenvectors : array of shape (dim, k)
            Corresponding eigenvectors (quantum states)
        """
        H = self.get_hamiltonian()
        
        print(f"\nFinding {k} lowest eigenstates...")
        start_time = time.time()
        
        # For small systems or when k is close to dimension, use dense solver
        if k >= self.dim - 1 or self.dim <= 10:
            from scipy.linalg import eigh
            H_dense = H.toarray()
            eigenvalues_all, eigenvectors_all = eigh(H_dense)
            eigenvalues = eigenvalues_all[:k]
            eigenvectors = eigenvectors_all[:, :k]
        else:
            eigenvalues, eigenvectors = eigsh(H, k=k, which='SA')
        
        elapsed = time.time() - start_time
        print(f"Diagonalization completed in {elapsed:.2f} seconds")
        print(f"Ground state energy: E_0 = {eigenvalues[0]:.6f}")
        
        if k > 1:
            print(f"First excited state: E_1 = {eigenvalues[1]:.6f}")
            print(f"Energy gap: ΔE = {eigenvalues[1] - eigenvalues[0]:.6f}")
        
        self._eigenvalues = eigenvalues
        self._eigenvectors = eigenvectors
        
        return eigenvalues, eigenvectors
    
    def expectation_value(self, operator: csr_matrix, state: np.ndarray) -> float:
        """
        Compute expectation value <ψ|O|ψ>
        
        Parameters:
        -----------
        operator : sparse matrix
            Observable operator
        state : array
            Quantum state (normalized)
        """
        return np.real(state.conj() @ operator @ state)
    
    def magnetization_z(self, state: Optional[np.ndarray] = None) -> float:
        """
        Compute magnetization in z direction: <M_z> = (1/N) Σ_i <σ_i^z>
        """
        if state is None:
            if self._eigenvectors is None:
                self.find_ground_state()
            state = self._eigenvectors[:, 0]
            
        mag = 0.0
        for i in range(self.N):
            op = self._single_site_operator(i, self.sigma_z)
            mag += self.expectation_value(op, state)
            
        return mag / self.N
    
    def magnetization_x(self, state: Optional[np.ndarray] = None) -> float:
        """
        Compute magnetization in x direction: <M_x> = (1/N) Σ_i <σ_i^x>
        """
        if state is None:
            if self._eigenvectors is None:
                self.find_ground_state()
            state = self._eigenvectors[:, 0]
            
        mag = 0.0
        for i in range(self.N):
            op = self._single_site_operator(i, self.sigma_x)
            mag += self.expectation_value(op, state)
            
        return mag / self.N
    
    def energy(self, state: Optional[np.ndarray] = None) -> float:
        """Compute energy <E> = <ψ|H|ψ>"""
        if state is None:
            if self._eigenvectors is None:
                self.find_ground_state()
            state = self._eigenvectors[:, 0]
            
        H = self.get_hamiltonian()
        return self.expectation_value(H, state)
    
    def correlation_zz(self, site1: int, site2: int, 
                       state: Optional[np.ndarray] = None) -> float:
        """
        Compute spin-spin correlation: <σ_i^z σ_j^z>
        """
        if state is None:
            if self._eigenvectors is None:
                self.find_ground_state()
            state = self._eigenvectors[:, 0]
            
        op = self._two_site_operator(site1, site2, self.sigma_z, self.sigma_z)
        return self.expectation_value(op, state)


class QuantumAnnealing:
    """
    Quantum Annealing Protocol for 3D Ising Model
    
    Time-dependent Hamiltonian:
    H(t) = -J * Σ_<i,j> σ_i^z σ_j^z - Γ(t) * Σ_i σ_i^x
    
    Starting with large Γ(0) and slowly decreasing to Γ(T) ≈ 0
    """
    
    def __init__(self, ising_model: QuantumIsing3D):
        self.model = ising_model
        
    def transverse_field_schedule(self, t: float, T: float, 
                                  Gamma_init: float, Gamma_final: float,
                                  schedule: str = 'linear') -> float:
        """
        Time-dependent transverse field Γ(t)
        
        Parameters:
        -----------
        t : float
            Current time
        T : float
            Total annealing time
        Gamma_init : float
            Initial transverse field
        Gamma_final : float
            Final transverse field
        schedule : str
            'linear', 'exponential', or 'polynomial'
        """
        s = t / T  # Normalized time [0, 1]
        
        if schedule == 'linear':
            return Gamma_init + (Gamma_final - Gamma_init) * s
        elif schedule == 'exponential':
            return Gamma_init * np.exp(np.log(Gamma_final / Gamma_init) * s)
        elif schedule == 'polynomial':
            # Polynomial schedule (slower at beginning and end)
            s_poly = 3 * s**2 - 2 * s**3
            return Gamma_init + (Gamma_final - Gamma_init) * s_poly
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
    
    def time_evolution_operator(self, H: csr_matrix, dt: float) -> np.ndarray:
        """
        Compute time evolution operator U(dt) = exp(-i H dt / ℏ)
        
        For imaginary time evolution, use dt -> -i β
        """
        # Convert to dense for matrix exponential (expensive!)
        H_dense = H.toarray()
        U = expm(-1j * H_dense * dt)
        return U
    
    def anneal(self, T: float, num_steps: int, 
               Gamma_init: float = 5.0, Gamma_final: float = 0.1,
               schedule: str = 'linear',
               initial_state: Optional[np.ndarray] = None) -> dict:
        """
        Perform quantum annealing
        
        Parameters:
        -----------
        T : float
            Total annealing time
        num_steps : int
            Number of time steps
        Gamma_init : float
            Initial transverse field (should be large)
        Gamma_final : float
            Final transverse field (should be small)
        schedule : str
            Annealing schedule type
        initial_state : array, optional
            Initial quantum state (if None, uses eigenstate of initial Hamiltonian)
            
        Returns:
        --------
        dict with keys:
            'times' : array of time points
            'gammas' : array of Γ(t) values
            'states' : list of quantum states
            'energies' : array of energies
            'mag_z' : array of z-magnetizations
            'mag_x' : array of x-magnetizations
        """
        print(f"\n{'='*60}")
        print("QUANTUM ANNEALING SIMULATION")
        print(f"{'='*60}")
        print(f"Total time: T = {T}")
        print(f"Time steps: {num_steps}")
        print(f"Γ: {Gamma_init} → {Gamma_final} ({schedule} schedule)")
        print(f"{'='*60}\n")
        
        dt = T / num_steps
        times = np.linspace(0, T, num_steps + 1)
        gammas = np.array([self.transverse_field_schedule(t, T, Gamma_init, 
                                                           Gamma_final, schedule) 
                          for t in times])
        
        # Initialize state
        if initial_state is None:
            print("Finding initial ground state...")
            self.model.Gamma = Gamma_init
            self.model._H = None  # Reset Hamiltonian
            _, initial_eigvecs = self.model.find_ground_state(k=1)
            state = initial_eigvecs[:, 0]
        else:
            state = initial_state / np.linalg.norm(initial_state)
            
        # Storage for observables
        states = [state.copy()]
        energies = [self.model.energy(state)]
        mag_z = [self.model.magnetization_z(state)]
        mag_x = [self.model.magnetization_x(state)]
        
        print(f"\nInitial state:")
        print(f"  Energy: {energies[0]:.6f}")
        print(f"  M_z: {mag_z[0]:.6f}")
        print(f"  M_x: {mag_x[0]:.6f}\n")
        
        # Time evolution
        print("Performing time evolution...")
        for i in range(num_steps):
            # Update Hamiltonian for current Γ(t)
            self.model.Gamma = gammas[i]
            self.model._H = None
            H = self.model.construct_hamiltonian()
            
            # Time evolution
            U = self.time_evolution_operator(H, dt)
            state = U @ state
            state = state / np.linalg.norm(state)  # Renormalize
            
            # Compute observables
            states.append(state.copy())
            energies.append(self.model.energy(state))
            mag_z.append(self.model.magnetization_z(state))
            mag_x.append(self.model.magnetization_x(state))
            
            if (i + 1) % max(1, num_steps // 10) == 0:
                print(f"  Step {i+1}/{num_steps}: Γ = {gammas[i]:.4f}, "
                      f"E = {energies[-1]:.6f}, M_z = {mag_z[-1]:.6f}")
        
        print("\nAnnealing complete!")
        print(f"Final state:")
        print(f"  Energy: {energies[-1]:.6f}")
        print(f"  M_z: {mag_z[-1]:.6f}")
        print(f"  M_x: {mag_x[-1]:.6f}")
        
        return {
            'times': times,
            'gammas': gammas,
            'states': states,
            'energies': np.array(energies),
            'mag_z': np.array(mag_z),
            'mag_x': np.array(mag_x)
        }


class PhaseTransitionAnalyzer:
    """
    Analyze quantum phase transition in transverse field Ising model
    
    Studies ground state properties as function of Γ/J
    """
    
    def __init__(self, Lx: int, Ly: int, Lz: int, J: float = 1.0,
                 periodic: bool = False):
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.J = J
        self.periodic = periodic
        
    def scan_transverse_field(self, gamma_values: np.ndarray,
                             compute_gap: bool = True) -> dict:
        """
        Scan ground state properties across range of Γ values
        
        Returns:
        --------
        dict with arrays of observables vs Γ
        """
        print(f"\n{'='*60}")
        print("PHASE TRANSITION ANALYSIS")
        print(f"{'='*60}")
        print(f"Lattice: {self.Lx}x{self.Ly}x{self.Lz}")
        print(f"Scanning {len(gamma_values)} values of Γ/J")
        print(f"{'='*60}\n")
        
        results = {
            'gamma': gamma_values,
            'energy': np.zeros(len(gamma_values)),
            'mag_z': np.zeros(len(gamma_values)),
            'mag_x': np.zeros(len(gamma_values)),
            'gap': np.zeros(len(gamma_values)) if compute_gap else None
        }
        
        for i, gamma in enumerate(gamma_values):
            print(f"\n[{i+1}/{len(gamma_values)}] Γ/J = {gamma:.4f}")
            print("-" * 40)
            
            model = QuantumIsing3D(self.Lx, self.Ly, self.Lz, 
                                  J=self.J, Gamma=gamma * self.J,
                                  periodic=self.periodic)
            
            k = 2 if compute_gap else 1
            eigenvalues, eigenvectors = model.find_ground_state(k=k)
            
            ground_state = eigenvectors[:, 0]
            
            results['energy'][i] = eigenvalues[0]
            results['mag_z'][i] = model.magnetization_z(ground_state)
            results['mag_x'][i] = model.magnetization_x(ground_state)
            
            if compute_gap:
                results['gap'][i] = eigenvalues[1] - eigenvalues[0]
                
            print(f"Results:")
            print(f"  E_0 = {results['energy'][i]:.6f}")
            print(f"  M_z = {results['mag_z'][i]:.6f}")
            print(f"  M_x = {results['mag_x'][i]:.6f}")
            if compute_gap:
                print(f"  Gap = {results['gap'][i]:.6f}")
        
        return results


def visualize_results(results: dict, title: str = "Quantum Ising Simulation"):
    """
    Visualize simulation results
    
    Parameters:
    -----------
    results : dict
        Dictionary containing simulation data
    title : str
        Plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Energy
    ax = axes[0, 0]
    if 'energies' in results:
        ax.plot(results['times'], results['energies'], 'b-', linewidth=2)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Energy', fontsize=12)
        ax.set_title('Energy vs Time')
    elif 'gamma' in results and 'energy' in results:
        ax.plot(results['gamma'], results['energy'], 'b-o', linewidth=2)
        ax.set_xlabel('Γ/J', fontsize=12)
        ax.set_ylabel('Ground State Energy', fontsize=12)
        ax.set_title('Energy vs Transverse Field')
        ax.axvline(x=1.0, color='r', linestyle='--', alpha=0.5, label='Γ_c = J (1D)')
        ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Magnetization Z
    ax = axes[0, 1]
    if 'mag_z' in results:
        x_var = results.get('times', results.get('gamma'))
        x_label = 'Time' if 'times' in results else 'Γ/J'
        ax.plot(x_var, results['mag_z'], 'r-o', linewidth=2, markersize=4)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel('M_z', fontsize=12)
        ax.set_title('Z-Magnetization')
        if 'gamma' in results:
            ax.axvline(x=1.0, color='k', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    # Magnetization X
    ax = axes[1, 0]
    if 'mag_x' in results:
        x_var = results.get('times', results.get('gamma'))
        x_label = 'Time' if 'times' in results else 'Γ/J'
        ax.plot(x_var, results['mag_x'], 'g-o', linewidth=2, markersize=4)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel('M_x', fontsize=12)
        ax.set_title('X-Magnetization')
        if 'gamma' in results:
            ax.axvline(x=1.0, color='k', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    # Transverse field or Gap
    ax = axes[1, 1]
    if 'gammas' in results:
        ax.plot(results['times'], results['gammas'], 'm-', linewidth=2)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Γ(t)', fontsize=12)
        ax.set_title('Transverse Field Schedule')
    elif 'gap' in results and results['gap'] is not None:
        ax.plot(results['gamma'], results['gap'], 'c-o', linewidth=2)
        ax.set_xlabel('Γ/J', fontsize=12)
        ax.set_ylabel('Energy Gap ΔE', fontsize=12)
        ax.set_title('Gap vs Transverse Field')
        ax.axvline(x=1.0, color='r', linestyle='--', alpha=0.5, label='Γ_c = J (1D)')
        ax.legend()
        ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║  3D TRANSVERSE FIELD ISING MODEL - QUANTUM SIMULATOR          ║
    ║                                                                ║
    ║  H = -J Σ_<i,j> σ_i^z σ_j^z - Γ Σ_i σ_i^x                    ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Note: For demonstration, using small system
    # For 3D, even 2x2x2 = 8 spins requires 2^8 = 256 dimensional Hilbert space
    print("This is a demonstration script. See examples/ for detailed usage.\n")
