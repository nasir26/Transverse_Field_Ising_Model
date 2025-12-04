# 3D Transverse Field Ising Model Quantum Simulation

A complete quantum simulation of the 3D transverse field Ising model, implementing the Hamiltonian:

```
H = -J * Σ_{<i,j>} σᵢᵢ σⱼᵢ - Γ * Σ_i σᵢˣ
```

where `σᵢᵢ` and `σᵢˣ` are Pauli matrices, `J` is the Ising coupling strength, and `Γ` is the transverse field strength.

## Features

- **3D Lattice Construction**: Builds periodic 3D lattices of arbitrary size
- **Quantum Hamiltonian**: Constructs the full quantum Hamiltonian using sparse matrices
- **Ground State Computation**: Finds the ground state energy and wavefunction using sparse eigensolvers
- **Observables**: Computes magnetization (longitudinal and transverse) and spin-spin correlations
- **Phase Transition Analysis**: Scans across transverse field values to study quantum phase transitions
- **Visualization**: Generates plots of energy, magnetization, and phase transitions

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from ising_3d_quantum import TransverseFieldIsing3D
import numpy as np

# Create a 2×2×2 lattice
model = TransverseFieldIsing3D(Lx=2, Ly=2, Lz=2, J=1.0, Gamma=0.5)

# Compute ground state
E0, psi0 = model.compute_ground_state()

# Compute observables
M_z = model.compute_magnetization(psi0)  # Longitudinal magnetization
M_x = model.compute_transverse_magnetization(psi0)  # Transverse magnetization
```

### Phase Transition Scan

```python
# Scan across different transverse field values
gamma_values = np.linspace(0.1, 2.0, 20) * J
model = TransverseFieldIsing3D(Lx=2, Ly=2, Lz=2, J=1.0, Gamma=1.0)
results = model.phase_transition_scan(gamma_values)

# Visualize results
model.visualize_phase_transition(results, save_path='phase_transition.png')
```

### Run Complete Simulation

```bash
python ising_3d_quantum.py
```

This will:
1. Compute ground state at different transverse field values
2. Perform a phase transition scan
3. Generate visualization plots

## Physics Background

### Quantum Phase Transition

The 3D transverse field Ising model exhibits a quantum phase transition at zero temperature:

- **Ferromagnetic phase** (Γ ≪ J): Spins align along the z-axis
- **Paramagnetic phase** (Γ ≫ J): Spins align along the x-axis (transverse field)
- **Critical point**: The transition occurs at a critical value Γ_c

For the 1D case, the critical point is exactly at Γ_c = J. In 3D, the critical point is shifted and requires numerical determination.

### Hamiltonian Structure

The Hamiltonian consists of two terms:

1. **Ising interaction**: `-J * Σ_{<i,j>} σᵢᵢ σⱼᵢ`
   - Favors alignment of neighboring spins along z-axis
   - Creates ferromagnetic order

2. **Transverse field**: `-Γ * Σ_i σᵢˣ`
   - Flips spins in the x-direction
   - Destroys ferromagnetic order
   - Creates quantum fluctuations

### Computational Considerations

The Hilbert space dimension grows exponentially with system size:
- 2×2×2 = 8 spins → 2^8 = 256 states
- 3×3×3 = 27 spins → 2^27 ≈ 134 million states
- 4×4×4 = 64 spins → 2^64 states (computationally intractable)

The code uses sparse matrix representations for efficiency, allowing simulation of systems up to ~20-30 spins on typical hardware.

## Output

The simulation generates:
- Ground state energy and wavefunction
- Longitudinal magnetization `<M_z>` = (1/N) Σ_i <σᵢᵢ>
- Transverse magnetization `<M_x>` = (1/N) Σ_i <σᵢˣ>
- Phase transition plots showing energy, magnetization, and susceptibility

## References

- Quantum phase transitions in transverse field Ising models
- Mapping to classical 2D Ising model via Trotter decomposition
- Jordan-Wigner transformation for 1D chains (exact solution)
- Quantum annealing applications

## License

This code is provided for educational and research purposes.
