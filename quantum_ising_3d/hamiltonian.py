"""
Hamiltonian Construction for 3D Transverse Field Ising Model

The Hamiltonian is:
    H = -J Σ_{<i,j>} σ_i^z σ_j^z - Γ Σ_i σ_i^x

where <i,j> denotes nearest neighbor pairs on a 3D cubic lattice.

For a cubic lattice of size L x L x L:
- Total sites: N = L³
- Each site has 6 nearest neighbors (with periodic boundary conditions)
- Hilbert space dimension: 2^N
"""

import numpy as np
from scipy import sparse
from scipy.sparse import kron, eye, csr_matrix, lil_matrix
from typing import Tuple, List, Optional
from functools import lru_cache

# Pauli matrices
def pauli_x() -> np.ndarray:
    """Pauli X matrix (σ^x)"""
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)

def pauli_y() -> np.ndarray:
    """Pauli Y matrix (σ^y)"""
    return np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)

def pauli_z() -> np.ndarray:
    """Pauli Z matrix (σ^z)"""
    return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

def identity() -> np.ndarray:
    """2x2 Identity matrix"""
    return np.eye(2, dtype=complex)


class TransverseFieldIsing3D:
    """
    3D Transverse Field Ising Model on a cubic lattice.
    
    H = -J Σ_{<i,j>} σ_i^z σ_j^z - Γ Σ_i σ_i^x
    
    Parameters
    ----------
    L : int
        Linear size of the cubic lattice (L x L x L)
    J : float
        Ising coupling strength (default: 1.0)
    Gamma : float
        Transverse field strength (default: 1.0)
    periodic : bool
        Use periodic boundary conditions (default: True)
    """
    
    def __init__(self, L: int, J: float = 1.0, Gamma: float = 1.0, periodic: bool = True):
        self.L = L
        self.N = L ** 3  # Total number of sites
        self.J = J
        self.Gamma = Gamma
        self.periodic = periodic
        self.dim = 2 ** self.N  # Hilbert space dimension
        
        # Check if system is small enough for exact diagonalization
        if self.N > 20:
            print(f"Warning: System has {self.N} sites (dim = 2^{self.N}). "
                  "Exact diagonalization may not be feasible. Consider using QMC.")
        
        # Build neighbor list
        self._neighbors = self._build_neighbor_list()
        
    def _site_index(self, x: int, y: int, z: int) -> int:
        """Convert 3D coordinates to 1D site index."""
        return x + self.L * y + self.L * self.L * z
    
    def _site_coords(self, idx: int) -> Tuple[int, int, int]:
        """Convert 1D site index to 3D coordinates."""
        x = idx % self.L
        y = (idx // self.L) % self.L
        z = idx // (self.L * self.L)
        return x, y, z
    
    def _build_neighbor_list(self) -> List[List[int]]:
        """
        Build list of nearest neighbors for each site.
        
        Returns list where neighbors[i] contains indices of neighbors of site i.
        Each bond (i,j) with i < j is counted once.
        """
        neighbors = [[] for _ in range(self.N)]
        
        for z in range(self.L):
            for y in range(self.L):
                for x in range(self.L):
                    idx = self._site_index(x, y, z)
                    
                    # Neighbors in +x, +y, +z directions (to avoid double counting)
                    for dx, dy, dz in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                        nx, ny, nz = x + dx, y + dy, z + dz
                        
                        # Handle boundary conditions
                        if self.periodic:
                            nx = nx % self.L
                            ny = ny % self.L
                            nz = nz % self.L
                        else:
                            if nx >= self.L or ny >= self.L or nz >= self.L:
                                continue
                        
                        neighbor_idx = self._site_index(nx, ny, nz)
                        neighbors[idx].append(neighbor_idx)
        
        return neighbors
    
    def get_bonds(self) -> List[Tuple[int, int]]:
        """Return list of all bonds (i, j) where i < j."""
        bonds = []
        for i, neighs in enumerate(self._neighbors):
            for j in neighs:
                if i < j:
                    bonds.append((i, j))
        return bonds
    
    def _single_site_operator(self, op: np.ndarray, site: int) -> sparse.csr_matrix:
        """
        Create operator acting on single site.
        
        O_site = I ⊗ I ⊗ ... ⊗ op ⊗ ... ⊗ I
        
        Parameters
        ----------
        op : np.ndarray
            2x2 single-site operator
        site : int
            Site index where operator acts
            
        Returns
        -------
        sparse.csr_matrix
            Sparse matrix representation in full Hilbert space
        """
        if site == 0:
            result = csr_matrix(op)
        else:
            result = sparse.eye(2, format='csr')
            
        for i in range(1, self.N):
            if i == site:
                result = kron(result, csr_matrix(op), format='csr')
            else:
                result = kron(result, sparse.eye(2, format='csr'), format='csr')
        
        return result
    
    def _two_site_operator(self, op1: np.ndarray, site1: int, 
                           op2: np.ndarray, site2: int) -> sparse.csr_matrix:
        """
        Create product of operators acting on two sites.
        
        O = O1_site1 * O2_site2
        """
        # Build the full operator by tensor product
        ops = [sparse.eye(2, format='csr') for _ in range(self.N)]
        ops[site1] = csr_matrix(op1)
        ops[site2] = csr_matrix(op2)
        
        result = ops[0]
        for i in range(1, self.N):
            result = kron(result, ops[i], format='csr')
        
        return result
    
    def build_hamiltonian(self) -> sparse.csr_matrix:
        """
        Build the full Hamiltonian matrix in sparse format.
        
        H = -J Σ_{<i,j>} σ_i^z σ_j^z - Γ Σ_i σ_i^x
        
        Returns
        -------
        sparse.csr_matrix
            Sparse Hamiltonian matrix
        """
        print(f"Building Hamiltonian for {self.L}x{self.L}x{self.L} lattice...")
        print(f"  Hilbert space dimension: {self.dim}")
        
        # Initialize as LIL matrix for efficient construction
        H = lil_matrix((self.dim, self.dim), dtype=complex)
        
        sigma_z = pauli_z()
        sigma_x = pauli_x()
        
        # Ising interaction: -J Σ_{<i,j>} σ_i^z σ_j^z
        print("  Adding Ising interactions...")
        bonds = self.get_bonds()
        for idx, (i, j) in enumerate(bonds):
            if idx % 100 == 0 and idx > 0:
                print(f"    Bond {idx}/{len(bonds)}")
            H -= self.J * self._two_site_operator(sigma_z, i, sigma_z, j)
        
        # Transverse field: -Γ Σ_i σ_i^x
        print("  Adding transverse field terms...")
        for i in range(self.N):
            H -= self.Gamma * self._single_site_operator(sigma_x, i)
        
        print("  Converting to CSR format...")
        return H.tocsr()
    
    def build_hamiltonian_fast(self) -> sparse.csr_matrix:
        """
        Build Hamiltonian using efficient bit manipulation.
        
        This method is faster for larger systems as it directly
        computes matrix elements without building intermediate operators.
        """
        print(f"Building Hamiltonian (fast) for {self.L}x{self.L}x{self.L} lattice...")
        print(f"  Hilbert space dimension: {self.dim}")
        
        # Use lil_matrix for efficient construction
        H = lil_matrix((self.dim, self.dim), dtype=float)
        
        bonds = self.get_bonds()
        
        # For each basis state |s⟩ (represented as integer)
        for s in range(self.dim):
            if s % 10000 == 0 and s > 0:
                print(f"  Processing state {s}/{self.dim}")
            
            # Diagonal part: -J Σ_{<i,j>} σ_i^z σ_j^z
            diag_element = 0.0
            for i, j in bonds:
                # Get spin values: +1 for |0⟩, -1 for |1⟩
                si = 1 - 2 * ((s >> i) & 1)
                sj = 1 - 2 * ((s >> j) & 1)
                diag_element -= self.J * si * sj
            
            H[s, s] = diag_element
            
            # Off-diagonal part: -Γ Σ_i σ_i^x (flips each spin)
            for i in range(self.N):
                s_flipped = s ^ (1 << i)  # Flip spin at site i
                H[s, s_flipped] -= self.Gamma
        
        return H.tocsr()
    
    def get_magnetization_operator(self, direction: str = 'z') -> sparse.csr_matrix:
        """
        Build magnetization operator M = (1/N) Σ_i σ_i^{direction}
        
        Parameters
        ----------
        direction : str
            'x', 'y', or 'z'
        """
        if direction == 'x':
            sigma = pauli_x()
        elif direction == 'y':
            sigma = pauli_y()
        else:
            sigma = pauli_z()
        
        M = sparse.csr_matrix((self.dim, self.dim), dtype=complex)
        for i in range(self.N):
            M += self._single_site_operator(sigma, i)
        
        return M / self.N
    
    def get_correlation_operator(self, site1: int, site2: int, 
                                  direction: str = 'z') -> sparse.csr_matrix:
        """
        Build two-point correlation operator σ_i^z σ_j^z
        """
        if direction == 'x':
            sigma = pauli_x()
        elif direction == 'y':
            sigma = pauli_y()
        else:
            sigma = pauli_z()
        
        return self._two_site_operator(sigma, site1, sigma, site2)
    
    def analyze_system(self):
        """Print system information."""
        print("=" * 60)
        print("3D Transverse Field Ising Model")
        print("=" * 60)
        print(f"Lattice size: {self.L} x {self.L} x {self.L}")
        print(f"Number of sites: {self.N}")
        print(f"Hilbert space dimension: {self.dim}")
        print(f"Coupling J: {self.J}")
        print(f"Transverse field Γ: {self.Gamma}")
        print(f"Γ/J ratio: {self.Gamma/self.J:.4f}")
        print(f"Periodic boundaries: {self.periodic}")
        print(f"Number of bonds: {len(self.get_bonds())}")
        print("=" * 60)


class TransverseFieldIsing1D:
    """
    1D Transverse Field Ising Model for comparison and validation.
    
    This model has an exact solution via Jordan-Wigner transformation.
    Critical point at Γ_c = J.
    """
    
    def __init__(self, L: int, J: float = 1.0, Gamma: float = 1.0, periodic: bool = True):
        self.L = L
        self.N = L
        self.J = J
        self.Gamma = Gamma
        self.periodic = periodic
        self.dim = 2 ** self.N
        
    def build_hamiltonian(self) -> sparse.csr_matrix:
        """Build 1D TFIM Hamiltonian."""
        H = lil_matrix((self.dim, self.dim), dtype=float)
        
        for s in range(self.dim):
            # Diagonal: -J Σ_i σ_i^z σ_{i+1}^z
            diag = 0.0
            for i in range(self.L):
                j = (i + 1) % self.L if self.periodic else i + 1
                if j < self.L:
                    si = 1 - 2 * ((s >> i) & 1)
                    sj = 1 - 2 * ((s >> j) & 1)
                    diag -= self.J * si * sj
            H[s, s] = diag
            
            # Off-diagonal: -Γ Σ_i σ_i^x
            for i in range(self.L):
                s_flip = s ^ (1 << i)
                H[s, s_flip] -= self.Gamma
        
        return H.tocsr()
    
    def exact_ground_state_energy(self) -> float:
        """
        Exact ground state energy per site using Jordan-Wigner transformation.
        
        For 1D TFIM with PBC, the exact result involves a sum over fermion modes.
        """
        # Dispersion relation: ε_k = 2√[(J cos(k) - Γ)² + J² sin²(k)]
        # For anti-periodic BC in fermion sector (ground state sector for even L)
        
        E0 = 0.0
        for n in range(self.L):
            k = 2 * np.pi * (n + 0.5) / self.L  # Anti-periodic BC
            eps_k = np.sqrt((self.J * np.cos(k) - self.Gamma)**2 + 
                           (self.J * np.sin(k))**2)
            E0 -= eps_k
        
        return E0 / self.L  # Energy per site
