"""
Quantum 3D Transverse Field Ising Model Simulation Package

This package provides tools for simulating the 3D Transverse Field Ising Model (TFIM):
    H = -J Σ_{<i,j>} σ_i^z σ_j^z - Γ Σ_i σ_i^x

Features:
- Exact diagonalization for small systems
- Quantum Monte Carlo (Stochastic Series Expansion) for larger systems
- Quantum annealing simulation
- Phase transition analysis
- Comprehensive visualization tools

Authors: Quantum Simulation Package
"""

__version__ = "1.0.0"

from .hamiltonian import (
    TransverseFieldIsing3D,
    pauli_x,
    pauli_z,
    pauli_y,
    identity
)
from .exact_diag import ExactDiagonalization
from .qmc import QuantumMonteCarlo
from .annealing import QuantumAnnealing
from .observables import Observables
from .visualization import Visualization

__all__ = [
    'TransverseFieldIsing3D',
    'ExactDiagonalization',
    'QuantumMonteCarlo',
    'QuantumAnnealing',
    'Observables',
    'Visualization',
    'pauli_x',
    'pauli_z',
    'pauli_y',
    'identity'
]
