# Quantum Simulation of 3D Transverse Field Ising Model

A complete implementation of quantum simulation for the 3D transverse field Ising model, featuring exact diagonalization, quantum phase transition analysis, and quantum annealing protocols.

## Overview

This project provides a comprehensive toolkit for studying quantum many-body physics through the transverse field Ising model:

```
H = -J ∑_<i,j> σ_i^z σ_j^z - Γ ∑_i σ_i^x
```

where:
- `σ^z, σ^x` are Pauli matrices
- `J` is the coupling strength (ferromagnetic when positive)
- `Γ` is the transverse field strength
- `<i,j>` denotes nearest-neighbor pairs on a 3D lattice

## Features

### Core Functionality
- ✅ **3D Lattice Construction**: Arbitrary `Lx × Ly × Lz` lattices with periodic/open boundary conditions
- ✅ **Exact Diagonalization**: Sparse matrix implementation for efficient ground state finding
- ✅ **Quantum Observables**: Energy, magnetization, spin correlations
- ✅ **Phase Transition Analysis**: Scan order parameters across transverse field
- ✅ **Quantum Annealing**: Time-dependent Hamiltonian with multiple schedules
- ✅ **Visualization Tools**: Comprehensive plotting utilities

### Physical Phenomena

#### Quantum Phase Transition
At zero temperature, the system undergoes a quantum phase transition at critical field `Γ_c`:

- **Γ << J**: Ferromagnetic phase (spins aligned along z)
  - `|M_z| ≈ 1`, `|M_x| ≈ 0`
  - Classical Ising ground state
  
- **Γ >> J**: Paramagnetic phase (spins aligned along x)
  - `|M_z| ≈ 0`, `|M_x| ≈ 1`
  - Quantum ground state

- **Critical Point**: `Γ_c/J` depends on dimensionality
  - 1D chain: `Γ_c/J = 1.0` (exact)
  - 2D lattice: `Γ_c/J ≈ 3.0-3.5`
  - 3D lattice: `Γ_c/J ≈ 4.5-5.5`

#### Quantum Annealing
Time-dependent protocol for optimization:

```
H(t) = -J ∑_<i,j> σ_i^z σ_j^z - Γ(t) ∑_i σ_i^x
```

Starting with large `Γ(0)` (easy quantum state) and slowly decreasing to `Γ(T) → 0`, the system adiabatically evolves to the classical ground state.

## Installation

### Requirements
- Python 3.8+
- NumPy
- SciPy
- Matplotlib

### Setup

```bash
# Clone repository
git clone <repository-url>
cd <repository-directory>

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from quantum_ising_3d import QuantumIsing3D

# Create 2×2×2 lattice
model = QuantumIsing3D(Lx=2, Ly=2, Lz=2, J=1.0, Gamma=0.5)

# Find ground state
eigenvalues, eigenvectors = model.find_ground_state(k=2)
ground_state = eigenvectors[:, 0]

# Compute observables
E0 = eigenvalues[0]
mag_z = model.magnetization_z(ground_state)
mag_x = model.magnetization_x(ground_state)

print(f"Ground state energy: {E0:.6f}")
print(f"Z-magnetization: {mag_z:.6f}")
print(f"X-magnetization: {mag_x:.6f}")
```

### Quantum Annealing

```python
from quantum_ising_3d import QuantumIsing3D, QuantumAnnealing

model = QuantumIsing3D(Lx=3, Ly=1, Lz=1, J=1.0)
annealer = QuantumAnnealing(model)

results = annealer.anneal(
    T=10.0,              # Total time
    num_steps=50,        # Time steps
    Gamma_init=5.0,      # Initial Γ
    Gamma_final=0.1,     # Final Γ
    schedule='linear'    # Annealing schedule
)

# Access results
energies = results['energies']
magnetizations = results['mag_z']
```

### Phase Transition Analysis

```python
from quantum_ising_3d import PhaseTransitionAnalyzer
import numpy as np

analyzer = PhaseTransitionAnalyzer(Lx=4, Ly=1, Lz=1, J=1.0)

# Scan transverse field
gamma_values = np.linspace(0.1, 3.0, 15)
results = analyzer.scan_transverse_field(gamma_values, compute_gap=True)

# Plot results
from quantum_ising_3d import visualize_results
import matplotlib.pyplot as plt

fig = visualize_results(results, title="Phase Transition")
plt.show()
```

## Examples

The `examples/` directory contains detailed demonstrations:

### Example 1: Ground State Properties
```bash
python examples/example_1_ground_state.py
```
- Finding ground states
- Computing observables
- Spin correlations
- Wavefunction analysis

### Example 2: Phase Transition
```bash
python examples/example_2_phase_transition.py
```
- Scanning order parameters
- Identifying critical point
- Energy gap analysis
- Susceptibility calculations

### Example 3: Quantum Annealing
```bash
python examples/example_3_quantum_annealing.py
```
- Annealing protocols
- Different schedules (linear, polynomial, exponential)
- Convergence analysis
- Ground state optimization

### Example 4: 3D Systems
```bash
python examples/example_4_3d_system.py
```
- True 3D lattices
- Geometry comparison
- Coordination effects
- 3D correlation structures

## System Size Limitations

Due to exponential growth of Hilbert space (`dim = 2^N`), exact diagonalization is limited to small systems:

| Lattice | Spins (N) | Hilbert Dim | Memory (est.) |
|---------|-----------|-------------|---------------|
| 3×1×1   | 3         | 8           | < 1 KB        |
| 2×2×1   | 4         | 16          | < 1 KB        |
| 2×2×2   | 8         | 256         | ~1 KB         |
| 3×3×1   | 9         | 512         | ~4 KB         |
| 4×2×2   | 16        | 65,536      | ~64 MB        |
| 3×3×3   | 27        | 134,217,728 | ~2 TB         |

**Practical limit**: N ≤ 12-15 spins for dense matrices, N ≤ 18-20 with sparse methods.

For larger systems, consider:
- Quantum Monte Carlo methods
- Tensor network algorithms (MPS, PEPS)
- Mean-field approximations
- Quantum computing hardware

## Theory Background

### Hamiltonian Structure

The model has two competing terms:

1. **Ising Interaction**: `-J σ_i^z σ_j^z`
   - Classical energy term
   - Favors aligned spins
   - Diagonal in computational basis

2. **Transverse Field**: `-Γ σ_i^x`
   - Quantum fluctuation term
   - Causes spin flips
   - Off-diagonal in computational basis

### Quantum vs Classical

- **Classical Ising** (Γ=0): Ground state is all spins aligned
- **Transverse Ising** (Γ>0): Quantum superposition of spin configurations
- **Phase transition** at Γ_c: Competition between order and quantum fluctuations

### Jordan-Wigner Transformation

For 1D chains, the model can be mapped to free fermions:

```
σ_j^+ = exp(iπ ∑_{l<j} c_l† c_l) c_j
σ_j^z = 2 c_j† c_j - 1
```

This allows exact solution of 1D case. Higher dimensions remain interacting.

### Mapping to 2D Classical Ising

Via Trotter decomposition, quantum 1D Ising maps to classical 2D Ising:
- Imaginary time becomes additional spatial dimension
- Quantum phase transition ↔ Classical phase transition in higher dimension

## API Reference

### QuantumIsing3D

Main class for quantum Ising simulation.

#### Constructor
```python
QuantumIsing3D(Lx, Ly, Lz, J=1.0, Gamma=0.5, periodic=False)
```

#### Methods
- `construct_hamiltonian()`: Build sparse Hamiltonian matrix
- `find_ground_state(k=1)`: Find k lowest eigenstates
- `magnetization_z(state)`: Compute z-magnetization
- `magnetization_x(state)`: Compute x-magnetization
- `energy(state)`: Compute energy expectation value
- `correlation_zz(i, j, state)`: Spin-spin correlation

### QuantumAnnealing

Quantum annealing protocol implementation.

#### Methods
```python
anneal(T, num_steps, Gamma_init, Gamma_final, schedule='linear')
```

Schedules: `'linear'`, `'exponential'`, `'polynomial'`

Returns dictionary with:
- `times`: Time points
- `gammas`: Γ(t) values
- `energies`: Energy evolution
- `mag_z`, `mag_x`: Magnetization evolution
- `states`: Quantum states at each time

### PhaseTransitionAnalyzer

Tools for studying phase transitions.

#### Methods
```python
scan_transverse_field(gamma_values, compute_gap=True)
```

Returns observables as function of Γ.

## Visualization

Built-in visualization tools:

```python
from quantum_ising_3d import visualize_results

fig = visualize_results(results, title="My Simulation")
fig.savefig("output.png", dpi=150)
```

Automatically detects data type (annealing vs phase transition) and creates appropriate plots.

## Performance Tips

1. **Use small systems**: Start with N ≤ 10 spins
2. **Sparse matrices**: Implementation uses scipy.sparse
3. **Eigenvalue count**: Request only needed eigenstates (k parameter)
4. **Periodic boundaries**: Increases bonds but maintains translational symmetry
5. **Caching**: Hamiltonian is cached after first construction

## Physical Insights

### What to Look For

1. **Order Parameters**:
   - M_z ≈ ±1: Ferromagnetic order
   - M_x ≈ ±1: Paramagnetic order
   - Both ≈ 0: Near critical point

2. **Energy Gap**:
   - Large gap: Gapped phase
   - Small/closing gap: Near transition
   - Gap minimum locates Γ_c

3. **Correlations**:
   - Positive <σ_i σ_j>: Ferromagnetic
   - Decay with distance: Correlation length
   - Long-range: Ordered phase

### Critical Behavior

Near quantum critical point:
- Energy gap: `Δ ∼ |Γ - Γ_c|^zν` (dynamical exponent z, correlation length exponent ν)
- Magnetization: `M ∼ |Γ - Γ_c|^β` (order parameter exponent β)
- Susceptibility: `χ ∼ |Γ - Γ_c|^{-γ}` (susceptibility exponent γ)

## Extensions

Possible extensions (not implemented):

1. **Finite Temperature**: Thermal density matrix ρ = exp(-βH)
2. **Dynamics**: Time-dependent Schrödinger equation
3. **Disorder**: Random couplings J_ij or fields Γ_i
4. **Long-range**: Power-law interactions ∼ 1/r^α
5. **Multiple Species**: Different spin types
6. **Measurements**: Quantum measurement protocols

## References

### Key Papers

1. **Transverse Ising Model**: Sachdev, S. "Quantum Phase Transitions" (2011)
2. **Quantum Annealing**: Kadowaki & Nishimori, PRE 58, 5355 (1998)
3. **1D Exact Solution**: Pfeuty, Ann. Phys. 57, 79 (1970)
4. **Critical Behavior**: Fisher, Rep. Prog. Phys. 30, 615 (1967)

### Textbooks

- "Quantum Ising Phases and Transitions" - Suzuki et al.
- "Quantum Phase Transitions" - Sachdev
- "Statistical Mechanics" - Pathria & Beale

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! Areas for improvement:
- Tensor network methods for larger systems
- GPU acceleration
- More observables (entanglement, etc.)
- Additional annealing schedules
- Benchmarking suite

## Citation

If you use this code in research, please cite:

```bibtex
@software{quantum_ising_3d,
  title={Quantum Simulation of 3D Transverse Field Ising Model},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/quantum-ising-3d}
}
```

## Contact

For questions or issues, please open a GitHub issue or contact [your-email].

---

**Note**: This implementation is for educational and research purposes. For production quantum simulations, consider specialized libraries like QuSpin, NetKet, or ITensor.
