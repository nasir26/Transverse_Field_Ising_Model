"""
3D Transverse Field Ising Model Quantum Simulation

This module implements a complete quantum simulation of the 3D transverse field
Ising model with Hamiltonian:
    H = -J * sum_{<i,j>} σᵢᵢ σⱼᵢ - Γ * sum_i σᵢˣ

where σᵢᵢ and σᵢˣ are Pauli matrices.
"""

import numpy as np
from scipy.sparse import csr_matrix, kron, identity
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import product
import warnings
warnings.filterwarnings('ignore')


class TransverseFieldIsing3D:
    """
    3D Transverse Field Ising Model Quantum Simulator
    
    Parameters:
    -----------
    Lx, Ly, Lz : int
        Lattice dimensions
    J : float
        Ising coupling strength (default: 1.0)
    Gamma : float
        Transverse field strength
    """
    
    def __init__(self, Lx, Ly, Lz, J=1.0, Gamma=1.0):
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.N = Lx * Ly * Lz  # Total number of spins
        self.J = J
        self.Gamma = Gamma
        
        # Pauli matrices
        self.sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        self.sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        self.identity = np.array([[1, 0], [0, 1]], dtype=complex)
        
        # Build lattice and neighbors
        self._build_lattice()
        self._build_neighbors()
        
    def _build_lattice(self):
        """Build 3D lattice coordinates"""
        self.sites = list(product(range(self.Lx), range(self.Ly), range(self.Lz)))
        self.site_to_index = {site: i for i, site in enumerate(self.sites)}
        
    def _build_neighbors(self):
        """Find all nearest neighbor pairs in 3D"""
        self.bonds = []
        for i, (x, y, z) in enumerate(self.sites):
            # Check 6 neighbors (x±1, y±1, z±1)
            neighbors = [
                ((x+1) % self.Lx, y, z),
                ((x-1) % self.Lx, y, z),
                (x, (y+1) % self.Ly, z),
                (x, (y-1) % self.Ly, z),
                (x, y, (z+1) % self.Lz),
                (x, y, (z-1) % self.Lz)
            ]
            for neighbor in neighbors:
                if neighbor in self.site_to_index:
                    j = self.site_to_index[neighbor]
                    if i < j:  # Avoid double counting
                        self.bonds.append((i, j))
        
        print(f"Lattice: {self.Lx}×{self.Ly}×{self.Lz} = {self.N} spins")
        print(f"Number of bonds: {len(self.bonds)}")
        
    def _pauli_operator(self, site_index, pauli_matrix):
        """
        Construct Pauli operator acting on a single site
        
        Parameters:
        -----------
        site_index : int
            Site index (0 to N-1)
        pauli_matrix : np.array
            Pauli matrix (sigma_x or sigma_z)
        
        Returns:
        --------
        scipy.sparse matrix of shape (2^N, 2^N)
        """
        operators = []
        for i in range(self.N):
            if i == site_index:
                operators.append(csr_matrix(pauli_matrix))
            else:
                operators.append(csr_matrix(self.identity))
        
        result = operators[0]
        for op in operators[1:]:
            result = kron(result, op, format='csr')
        return result
    
    def _build_hamiltonian(self):
        """
        Build the full Hamiltonian matrix
        
        H = -J * sum_{<i,j>} σᵢᵢ σⱼᵢ - Γ * sum_i σᵢˣ
        """
        print("Building Hamiltonian matrix...")
        H = csr_matrix((2**self.N, 2**self.N), dtype=complex)
        
        # Ising interaction term: -J * sum_{<i,j>} σᵢᵢ σⱼᵢ
        print("  Adding Ising interaction terms...")
        for i, j in self.bonds:
            sigma_z_i = self._pauli_operator(i, self.sigma_z)
            sigma_z_j = self._pauli_operator(j, self.sigma_z)
            H -= self.J * (sigma_z_i @ sigma_z_j)
        
        # Transverse field term: -Γ * sum_i σᵢˣ
        print("  Adding transverse field terms...")
        for i in range(self.N):
            sigma_x_i = self._pauli_operator(i, self.sigma_x)
            H -= self.Gamma * sigma_x_i
        
        print(f"  Hamiltonian matrix size: {H.shape}")
        print(f"  Non-zero elements: {H.nnz}")
        return H
    
    def compute_ground_state(self, k=1):
        """
        Compute ground state energy and wavefunction
        
        Parameters:
        -----------
        k : int
            Number of eigenvalues to compute (default: 1 for ground state)
        
        Returns:
        --------
        E0 : float
            Ground state energy
        psi0 : np.array
            Ground state wavefunction
        """
        print(f"\nComputing ground state for Γ/J = {self.Gamma/self.J:.3f}...")
        H = self._build_hamiltonian()
        
        print("Diagonalizing Hamiltonian...")
        # Use sparse eigensolver for efficiency
        eigenvalues, eigenvectors = eigs(H, k=k, which='SR')  # Smallest Real
        eigenvalues = np.real(eigenvalues)
        idx = np.argsort(eigenvalues)
        E0 = eigenvalues[idx[0]]
        psi0 = np.real(eigenvectors[:, idx[0]])
        
        print(f"Ground state energy: E0 = {E0:.6f}")
        print(f"Energy per spin: E0/N = {E0/self.N:.6f}")
        
        return E0, psi0
    
    def compute_magnetization(self, psi):
        """
        Compute magnetization along z-axis
        
        <M_z> = (1/N) * sum_i <σᵢᵢ>
        """
        M_z = 0.0
        for i in range(self.N):
            sigma_z_i = self._pauli_operator(i, self.sigma_z)
            M_z += np.real(psi.conj() @ sigma_z_i @ psi)
        return M_z / self.N
    
    def compute_correlation(self, psi, i, j):
        """
        Compute spin-spin correlation <σᵢᵢ σⱼᵢ>
        """
        sigma_z_i = self._pauli_operator(i, self.sigma_z)
        sigma_z_j = self._pauli_operator(j, self.sigma_z)
        corr = np.real(psi.conj() @ (sigma_z_i @ sigma_z_j) @ psi)
        return corr
    
    def compute_transverse_magnetization(self, psi):
        """
        Compute magnetization along x-axis (transverse)
        
        <M_x> = (1/N) * sum_i <σᵢˣ>
        """
        M_x = 0.0
        for i in range(self.N):
            sigma_x_i = self._pauli_operator(i, self.sigma_x)
            M_x += np.real(psi.conj() @ sigma_x_i @ psi)
        return M_x / self.N
    
    def phase_transition_scan(self, gamma_values, k=1):
        """
        Scan across different transverse field values to study phase transition
        
        Parameters:
        -----------
        gamma_values : array-like
            Values of Γ to scan
        k : int
            Number of eigenvalues to compute
        
        Returns:
        --------
        results : dict
            Dictionary with energies, magnetizations, etc.
        """
        print("\n" + "="*60)
        print("Phase Transition Scan")
        print("="*60)
        
        energies = []
        mag_z = []
        mag_x = []
        
        original_gamma = self.Gamma
        
        for gamma in gamma_values:
            self.Gamma = gamma
            print(f"\nΓ/J = {gamma/self.J:.3f}")
            
            E0, psi0 = self.compute_ground_state(k=k)
            M_z = self.compute_magnetization(psi0)
            M_x = self.compute_transverse_magnetization(psi0)
            
            energies.append(E0)
            mag_z.append(M_z)
            mag_x.append(M_x)
            
            print(f"  <M_z> = {M_z:.6f}")
            print(f"  <M_x> = {M_x:.6f}")
        
        self.Gamma = original_gamma
        
        return {
            'gamma': np.array(gamma_values),
            'energy': np.array(energies),
            'magnetization_z': np.array(mag_z),
            'magnetization_x': np.array(mag_x)
        }
    
    def visualize_phase_transition(self, results, save_path=None):
        """
        Visualize phase transition results
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        gamma_ratio = results['gamma'] / self.J
        
        # Energy per spin
        axes[0, 0].plot(gamma_ratio, results['energy'] / self.N, 'b-o', markersize=4)
        axes[0, 0].set_xlabel('Γ/J')
        axes[0, 0].set_ylabel('E₀/N')
        axes[0, 0].set_title('Ground State Energy per Spin')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Magnetization along z
        axes[0, 1].plot(gamma_ratio, results['magnetization_z'], 'r-o', markersize=4)
        axes[0, 1].set_xlabel('Γ/J')
        axes[0, 1].set_ylabel('<M_z>')
        axes[0, 1].set_title('Longitudinal Magnetization')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axvline(x=1.0, color='k', linestyle='--', alpha=0.5, label='Γ_c/J = 1')
        axes[0, 1].legend()
        
        # Magnetization along x
        axes[1, 0].plot(gamma_ratio, results['magnetization_x'], 'g-o', markersize=4)
        axes[1, 0].set_xlabel('Γ/J')
        axes[1, 0].set_ylabel('<M_x>')
        axes[1, 0].set_title('Transverse Magnetization')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axvline(x=1.0, color='k', linestyle='--', alpha=0.5, label='Γ_c/J = 1')
        axes[1, 0].legend()
        
        # Energy derivative (susceptibility)
        dE_dgamma = np.gradient(results['energy'] / self.N, results['gamma'] / self.J)
        axes[1, 1].plot(gamma_ratio, -dE_dgamma, 'm-o', markersize=4)
        axes[1, 1].set_xlabel('Γ/J')
        axes[1, 1].set_ylabel('-d(E₀/N)/d(Γ/J)')
        axes[1, 1].set_title('Energy Derivative (Susceptibility)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axvline(x=1.0, color='k', linestyle='--', alpha=0.5, label='Γ_c/J = 1')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nFigure saved to {save_path}")
        
        plt.show()


def main():
    """
    Main simulation function
    """
    print("="*60)
    print("3D Transverse Field Ising Model Quantum Simulation")
    print("="*60)
    
    # Small lattice for demonstration (larger systems require more memory)
    # For 2×2×2 = 8 spins: 2^8 = 256 states (manageable)
    # For 3×3×3 = 27 spins: 2^27 = 134M states (requires significant memory)
    
    Lx, Ly, Lz = 2, 2, 2
    J = 1.0
    
    print(f"\nSimulation parameters:")
    print(f"  Lattice size: {Lx}×{Ly}×{Lz} = {Lx*Ly*Lz} spins")
    print(f"  Hilbert space dimension: 2^{Lx*Ly*Lz} = {2**(Lx*Ly*Lz)}")
    print(f"  Ising coupling: J = {J}")
    
    # Example 1: Single point calculation
    print("\n" + "-"*60)
    print("Example 1: Ground state at Γ/J = 0.5 (ferromagnetic phase)")
    print("-"*60)
    model1 = TransverseFieldIsing3D(Lx, Ly, Lz, J=J, Gamma=0.5*J)
    E0_1, psi0_1 = model1.compute_ground_state()
    M_z_1 = model1.compute_magnetization(psi0_1)
    M_x_1 = model1.compute_transverse_magnetization(psi0_1)
    print(f"Longitudinal magnetization: <M_z> = {M_z_1:.6f}")
    print(f"Transverse magnetization: <M_x> = {M_x_1:.6f}")
    
    # Example 2: Phase transition scan
    print("\n" + "-"*60)
    print("Example 2: Phase transition scan")
    print("-"*60)
    gamma_values = np.linspace(0.1, 2.0, 10) * J
    model2 = TransverseFieldIsing3D(Lx, Ly, Lz, J=J, Gamma=1.0*J)
    results = model2.phase_transition_scan(gamma_values)
    
    # Visualize results
    print("\n" + "-"*60)
    print("Generating phase transition plots...")
    print("-"*60)
    model2.visualize_phase_transition(results, save_path='phase_transition_3d.png')
    
    # Example 3: Critical point
    print("\n" + "-"*60)
    print("Example 3: Ground state at critical point Γ/J = 1.0")
    print("-"*60)
    model3 = TransverseFieldIsing3D(Lx, Ly, Lz, J=J, Gamma=1.0*J)
    E0_3, psi0_3 = model3.compute_ground_state()
    M_z_3 = model3.compute_magnetization(psi0_3)
    M_x_3 = model3.compute_transverse_magnetization(psi0_3)
    print(f"Longitudinal magnetization: <M_z> = {M_z_3:.6f}")
    print(f"Transverse magnetization: <M_x> = {M_x_3:.6f}")
    
    print("\n" + "="*60)
    print("Simulation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
