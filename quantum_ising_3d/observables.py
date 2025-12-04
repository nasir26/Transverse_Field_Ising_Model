"""
Physical Observables for the 3D Transverse Field Ising Model

This module provides functions to compute various physical quantities:
- Magnetization and susceptibility
- Correlation functions
- Energy and specific heat
- Binder cumulant
- Entanglement measures

These are used to characterize quantum phases and phase transitions.
"""

import numpy as np
from scipy import sparse
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class PhaseProperties:
    """Container for phase characterization data."""
    Gamma_over_J: float
    magnetization_z: float
    magnetization_x: float
    susceptibility_z: float
    susceptibility_x: float
    energy_per_site: float
    gap: float
    phase: str  # 'ferromagnetic', 'paramagnetic', or 'critical'


class Observables:
    """
    Compute physical observables for the TFIM.
    
    Parameters
    ----------
    model : TransverseFieldIsing3D or similar
        The quantum Ising model
    solver : ExactDiagonalization or QuantumMonteCarlo
        Solver providing state information
    """
    
    def __init__(self, model, solver=None):
        self.model = model
        self.solver = solver
        self.N = model.N
        self.L = model.L
    
    @staticmethod
    def compute_magnetization(spins: np.ndarray) -> float:
        """
        Compute magnetization for a classical spin configuration.
        
        Parameters
        ----------
        spins : np.ndarray
            Array of ±1 spins
            
        Returns
        -------
        float
            Magnetization per site
        """
        return np.mean(spins)
    
    @staticmethod
    def compute_staggered_magnetization(spins: np.ndarray, L: int) -> float:
        """
        Compute staggered (antiferromagnetic) magnetization.
        
        For 3D cubic lattice, alternating sign based on parity of x+y+z.
        """
        N = L ** 3
        M_stag = 0.0
        
        for z in range(L):
            for y in range(L):
                for x in range(L):
                    idx = x + L * y + L * L * z
                    sign = (-1) ** (x + y + z)
                    M_stag += sign * spins[idx]
        
        return M_stag / N
    
    def compute_correlation_function(self, state: np.ndarray,
                                       r_max: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute z-z correlation function C(r) = ⟨σ_0^z σ_r^z⟩ - ⟨σ_0^z⟩⟨σ_r^z⟩
        averaged over sites.
        
        Parameters
        ----------
        state : np.ndarray
            Quantum state vector
        r_max : int, optional
            Maximum distance to compute
            
        Returns
        -------
        distances : np.ndarray
            Array of distances
        correlations : np.ndarray
            Correlation values
        """
        if r_max is None:
            r_max = self.L // 2
        
        # For each distance r, compute average correlation
        from collections import defaultdict
        corr_by_distance = defaultdict(list)
        
        L = self.L
        
        # Reference site (center of lattice)
        ref_x, ref_y, ref_z = L // 2, L // 2, L // 2
        ref_idx = ref_x + L * ref_y + L * L * ref_z
        
        for z in range(L):
            for y in range(L):
                for x in range(L):
                    idx = x + L * y + L * L * z
                    
                    # Manhattan distance (with PBC)
                    dx = min(abs(x - ref_x), L - abs(x - ref_x))
                    dy = min(abs(y - ref_y), L - abs(y - ref_y))
                    dz = min(abs(z - ref_z), L - abs(z - ref_z))
                    r = dx + dy + dz
                    
                    if r <= r_max:
                        # Compute correlation
                        C = self._correlation_from_state(state, ref_idx, idx)
                        corr_by_distance[r].append(C)
        
        # Average over sites at same distance
        distances = sorted(corr_by_distance.keys())
        correlations = [np.mean(corr_by_distance[r]) for r in distances]
        
        return np.array(distances), np.array(correlations)
    
    def _correlation_from_state(self, state: np.ndarray, i: int, j: int) -> float:
        """Compute ⟨σ_i^z σ_j^z⟩ from quantum state."""
        # For each basis state, compute σ_i^z σ_j^z and weight by probability
        correlation = 0.0
        
        for s in range(len(state)):
            prob = np.abs(state[s]) ** 2
            si = 1 - 2 * ((s >> i) & 1)
            sj = 1 - 2 * ((s >> j) & 1)
            correlation += prob * si * sj
        
        return correlation
    
    @staticmethod
    def compute_binder_cumulant(magnetization_samples: np.ndarray) -> float:
        """
        Compute Binder cumulant U = 1 - ⟨m⁴⟩ / (3⟨m²⟩²)
        
        At a phase transition, Binder cumulants for different system sizes cross.
        In the ordered phase: U → 2/3
        In the disordered phase: U → 0
        At criticality: U has universal value
        
        Parameters
        ----------
        magnetization_samples : np.ndarray
            Array of magnetization measurements
            
        Returns
        -------
        float
            Binder cumulant value
        """
        m2 = np.mean(magnetization_samples ** 2)
        m4 = np.mean(magnetization_samples ** 4)
        
        if m2 < 1e-10:
            return 0.0
        
        return 1 - m4 / (3 * m2 ** 2)
    
    @staticmethod
    def compute_susceptibility(magnetization_samples: np.ndarray, 
                                beta: float, N: int) -> float:
        """
        Compute magnetic susceptibility χ = β N (⟨m²⟩ - ⟨m⟩²)
        
        Parameters
        ----------
        magnetization_samples : np.ndarray
            Array of magnetization measurements
        beta : float
            Inverse temperature
        N : int
            Number of sites
            
        Returns
        -------
        float
            Magnetic susceptibility
        """
        m_mean = np.mean(magnetization_samples)
        m2_mean = np.mean(magnetization_samples ** 2)
        
        return beta * N * (m2_mean - m_mean ** 2)
    
    @staticmethod
    def compute_specific_heat(energy_samples: np.ndarray,
                               beta: float, N: int) -> float:
        """
        Compute specific heat C = (β²/N) (⟨E²⟩ - ⟨E⟩²)
        
        Parameters
        ----------
        energy_samples : np.ndarray
            Array of energy measurements
        beta : float
            Inverse temperature
        N : int
            Number of sites
            
        Returns
        -------
        float
            Specific heat per site
        """
        E_mean = np.mean(energy_samples)
        E2_mean = np.mean(energy_samples ** 2)
        
        return (beta ** 2 / N) * (E2_mean - E_mean ** 2)
    
    def characterize_phase(self, Gamma: float) -> PhaseProperties:
        """
        Characterize the quantum phase at given Γ.
        
        Uses order parameters to determine if system is in:
        - Ferromagnetic phase (Γ << J): |⟨M_z⟩| ≈ 1, ⟨M_x⟩ ≈ 0
        - Paramagnetic phase (Γ >> J): ⟨M_z⟩ ≈ 0, |⟨M_x⟩| ≈ 1
        - Critical region: enhanced fluctuations
        
        Parameters
        ----------
        Gamma : float
            Transverse field strength
            
        Returns
        -------
        PhaseProperties
            Phase characterization data
        """
        if self.solver is None:
            raise ValueError("Solver required for phase characterization")
        
        # Update model and solver
        self.model.Gamma = Gamma
        self.solver.H = None
        self.solver.eigenvalues = None
        self.solver.ground_state = None
        
        # Compute ground state properties
        self.solver.build_hamiltonian()
        self.solver.diagonalize(n_states=2)
        
        mz = self.solver.compute_magnetization('z')
        mx = self.solver.compute_magnetization('x')
        mz2 = self.solver.compute_magnetization_squared('z')
        mx2 = self.solver.compute_magnetization_squared('x')
        
        chi_z = self.N * (mz2 - mz ** 2)
        chi_x = self.N * (mx2 - mx ** 2)
        
        gap = self.solver.eigenvalues[1] - self.solver.eigenvalues[0]
        E_per_site = self.solver.ground_state_energy / self.N
        
        # Determine phase
        ratio = Gamma / self.model.J
        
        if mz2 > 0.5 and chi_z < chi_x:
            phase = 'ferromagnetic'
        elif mx2 > 0.5 and chi_x < chi_z:
            phase = 'paramagnetic'
        else:
            phase = 'critical'
        
        return PhaseProperties(
            Gamma_over_J=ratio,
            magnetization_z=mz,
            magnetization_x=mx,
            susceptibility_z=chi_z,
            susceptibility_x=chi_x,
            energy_per_site=E_per_site,
            gap=gap,
            phase=phase
        )
    
    def find_critical_point(self, Gamma_range: Tuple[float, float] = (0.1, 10.0),
                             n_points: int = 50) -> Tuple[float, Dict[str, np.ndarray]]:
        """
        Find critical point by locating peak in susceptibility.
        
        For 3D TFIM, the critical point is approximately Γ_c/J ≈ 5.2
        (depending on lattice type and boundary conditions).
        
        Returns
        -------
        Gamma_c : float
            Estimated critical field
        data : Dict
            Full scan data
        """
        if self.solver is None:
            raise ValueError("Solver required for critical point search")
        
        Gamma_values = np.linspace(Gamma_range[0], Gamma_range[1], n_points)
        
        data = {
            'Gamma': Gamma_values,
            'susceptibility_z': [],
            'gap': [],
            'magnetization_z_squared': []
        }
        
        print(f"Scanning for critical point in Γ ∈ [{Gamma_range[0]}, {Gamma_range[1]}]")
        
        for Gamma in Gamma_values:
            props = self.characterize_phase(Gamma)
            data['susceptibility_z'].append(props.susceptibility_z)
            data['gap'].append(props.gap)
            data['magnetization_z_squared'].append(props.magnetization_z ** 2)
        
        for key in data:
            data[key] = np.array(data[key])
        
        # Critical point: maximum susceptibility or minimum gap
        chi_max_idx = np.argmax(data['susceptibility_z'])
        gap_min_idx = np.argmin(data['gap'])
        
        Gamma_c_chi = Gamma_values[chi_max_idx]
        Gamma_c_gap = Gamma_values[gap_min_idx]
        
        print(f"Critical point estimates:")
        print(f"  From susceptibility peak: Γ_c = {Gamma_c_chi:.4f}")
        print(f"  From gap minimum: Γ_c = {Gamma_c_gap:.4f}")
        
        # Average estimate
        Gamma_c = (Gamma_c_chi + Gamma_c_gap) / 2
        
        return Gamma_c, data
    
    @staticmethod
    def finite_size_scaling(observable_L: Dict[int, np.ndarray],
                            Gamma_values: np.ndarray,
                            critical_exponents: Dict[str, float]) -> Dict[str, Any]:
        """
        Perform finite-size scaling analysis.
        
        Near a quantum phase transition:
            χ ~ L^{γ/ν} f((Γ - Γ_c) L^{1/ν})
            
        Parameters
        ----------
        observable_L : Dict[int, np.ndarray]
            Observable values for different system sizes L
        Gamma_values : np.ndarray
            Γ values used
        critical_exponents : Dict[str, float]
            Dictionary with 'nu', 'gamma', etc.
            
        Returns
        -------
        Dict with scaling collapse data
        """
        # Placeholder for finite-size scaling analysis
        # Would require multiple system sizes
        return {
            'Gamma_c': None,
            'nu': critical_exponents.get('nu', 0.63),
            'gamma': critical_exponents.get('gamma', 1.24),
            'collapsed_data': None
        }


class QuantumOrderParameters:
    """
    Quantum order parameters for the TFIM.
    
    In the TFIM:
    - Order parameter: ⟨σ^z⟩ (ferromagnetic order)
    - Transverse magnetization: ⟨σ^x⟩ (paramagnetic tendency)
    
    At the quantum phase transition:
    - Order parameter vanishes: ⟨σ^z⟩ → 0
    - Gap closes: Δ → 0
    - Correlation length diverges: ξ → ∞
    """
    
    @staticmethod
    def ferromagnetic_order(magnetization_z: float) -> float:
        """
        Ferromagnetic order parameter.
        
        Non-zero in the ordered phase (Γ < Γ_c).
        """
        return abs(magnetization_z)
    
    @staticmethod
    def paramagnetic_order(magnetization_x: float) -> float:
        """
        Paramagnetic order parameter.
        
        Approaches 1 in the paramagnetic phase (Γ >> Γ_c).
        """
        return abs(magnetization_x)
    
    @staticmethod
    def order_parameter_ratio(mz: float, mx: float) -> float:
        """
        Ratio of order parameters: |⟨σ^z⟩| / |⟨σ^x⟩|
        
        > 1 in ferromagnetic phase
        < 1 in paramagnetic phase
        ≈ 1 near critical point
        """
        if abs(mx) < 1e-10:
            return np.inf
        return abs(mz) / abs(mx)


def compute_critical_exponents_3d() -> Dict[str, float]:
    """
    Return known critical exponents for 3D Ising universality class.
    
    The 3D TFIM belongs to the (3+1)D classical Ising universality class.
    """
    return {
        'nu': 0.6301,      # Correlation length exponent
        'gamma': 1.2372,   # Susceptibility exponent
        'beta': 0.3265,    # Order parameter exponent
        'alpha': 0.110,    # Specific heat exponent
        'eta': 0.0364,     # Anomalous dimension
        'delta': 4.789,    # Critical isotherm exponent
        'z': 1.0           # Dynamic critical exponent (for quantum case)
    }
