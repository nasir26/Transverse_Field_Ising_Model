# 3D Transverse Field Ising Model Quantum Simulation

A comprehensive quantum simulation package for the 3D Transverse Field Ising Model (TFIM), implementing exact diagonalization, quantum Monte Carlo, and quantum annealing methods.

## 📚 Physics Background

### The Transverse Field Ising Model

The quantum Hamiltonian is:

$$\hat{H} = -J\sum_{\langle i,j\rangle} \hat{\sigma}_i^z \hat{\sigma}_j^z - \Gamma \sum_i \hat{\sigma}_i^x$$

where:
- $\hat{\sigma}^z$ and $\hat{\sigma}^x$ are Pauli matrices
- $J$ is the Ising coupling (ferromagnetic for $J > 0$)
- $\Gamma$ is the transverse field strength
- $\langle i,j \rangle$ denotes nearest neighbor pairs on a 3D cubic lattice

### Quantum Phase Transition

The system exhibits a **quantum phase transition** at zero temperature:

| Regime | Phase | Order Parameter |
|--------|-------|-----------------|
| $\Gamma \ll J$ | Ferromagnetic | $\langle \sigma^z \rangle \neq 0$ |
| $\Gamma \gg J$ | Paramagnetic | $\langle \sigma^x \rangle \approx 1$ |
| $\Gamma \approx \Gamma_c$ | Critical | Gap closes, $\xi \to \infty$ |

For the 3D cubic lattice, the critical point is approximately $\Gamma_c/J \approx 5.2$.

### Key Concepts Implemented

1. **Jordan-Wigner Transformation** (1D verification)
   - Maps spins to free fermions for exact solution
   - Used to validate numerical methods

2. **Trotter Decomposition** (QMC)
   - Maps quantum (d)D system to classical (d+1)D system
   - Enables Monte Carlo sampling

3. **Quantum Annealing**
   - Time-dependent Hamiltonian: $H(t) = H_{\text{problem}} + \Gamma(t) H_{\text{field}}$
   - Adiabatic evolution finds ground state

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd quantum_ising_3d

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from quantum_ising_3d import TransverseFieldIsing3D, ExactDiagonalization

# Create a 2x2x2 lattice
model = TransverseFieldIsing3D(L=2, J=1.0, Gamma=1.0)
model.analyze_system()

# Solve for ground state
solver = ExactDiagonalization(model)
solver.build_hamiltonian()
solver.diagonalize(n_states=4)

# Compute observables
print(f"Ground state energy: {solver.ground_state_energy:.6f}")
print(f"Magnetization <M_z>: {solver.compute_magnetization('z'):.6f}")
print(f"Energy gap: {solver.compute_energy_gap():.6f}")
```

### Run Full Simulation

```bash
# Run complete demonstration
python run_simulation.py all

# Run specific modes
python run_simulation.py exact --L 2 --Gamma 1.0
python run_simulation.py phase --L 2 --n-points 30
python run_simulation.py qmc --L 4 --beta 2.0
python run_simulation.py anneal --L 2 --t-anneal 20.0
```

## 📁 Project Structure

```
quantum_ising_3d/
├── __init__.py           # Package initialization
├── hamiltonian.py        # Hamiltonian construction (3D & 1D)
├── exact_diag.py         # Exact diagonalization solver
├── qmc.py                # Quantum Monte Carlo (SSE & World-Line)
├── annealing.py          # Quantum annealing simulation
├── observables.py        # Physical observables
├── visualization.py      # Plotting tools
└── utils.py              # Utility functions

examples/
├── basic_simulation.py          # Basic usage example
├── quantum_annealing_example.py # Annealing demonstration
└── qmc_example.py               # QMC demonstration

tests/
└── test_simulation.py    # Unit tests

run_simulation.py         # Main entry point
requirements.txt          # Dependencies
```

## 🔬 Simulation Methods

### 1. Exact Diagonalization

For small systems ($N \leq 12$ sites, $L \leq 2$):

```python
from quantum_ising_3d import TransverseFieldIsing3D, ExactDiagonalization

model = TransverseFieldIsing3D(L=2, J=1.0, Gamma=1.0)
solver = ExactDiagonalization(model)

# Phase diagram scan
Gamma_values = np.linspace(0.1, 5.0, 30)
results = solver.phase_diagram_scan(Gamma_values, 
    observables=['energy_per_site', 'magnetization_z', 'gap'])
```

**Capabilities:**
- Full spectrum computation
- Ground state wavefunction
- Correlation functions
- Entanglement entropy
- Energy gap analysis

### 2. Quantum Monte Carlo (SSE)

For larger systems using Stochastic Series Expansion:

```python
from quantum_ising_3d import QuantumMonteCarlo

qmc = QuantumMonteCarlo(L=4, J=1.0, Gamma=3.0, beta=2.0)
results = qmc.run(n_thermalization=500, n_measurements=5000)

# Phase diagram
phase_data = qmc.phase_diagram_scan(Gamma_values)
```

**Capabilities:**
- Finite temperature simulations
- Large system sizes (N ~ 100+)
- Automatic expansion cutoff adjustment
- Error estimation via binning

### 3. Quantum Annealing

Time-dependent simulation for optimization:

```python
from quantum_ising_3d import TransverseFieldIsing3D, QuantumAnnealing

model = TransverseFieldIsing3D(L=2, J=1.0, Gamma=0.0)
qa = QuantumAnnealing(model)

# Compute minimum gap
gaps, min_gap, Gamma_c = qa.compute_minimum_gap(Gamma_values)

# Run annealing
results = qa.run_exact_annealing(t_anneal=20.0, schedule='linear')
print(f"Success probability: {results['success_probability']:.4f}")
```

**Annealing Schedules:**
- `linear`: $\Gamma(t) = \Gamma_0 (1 - t/t_{\text{anneal}})$
- `exponential`: $\Gamma(t) = \Gamma_0 e^{-\alpha t}$
- `polynomial`: $\Gamma(t) = \Gamma_0 (1 - t/t_{\text{anneal}})^p$

## 📊 Physical Observables

| Observable | Description | Method |
|-----------|-------------|--------|
| `magnetization_z` | $\langle M_z \rangle = \frac{1}{N}\sum_i \langle \sigma_i^z \rangle$ | Order parameter |
| `magnetization_x` | $\langle M_x \rangle = \frac{1}{N}\sum_i \langle \sigma_i^x \rangle$ | Field alignment |
| `energy_per_site` | $E_0 / N$ | Ground state energy density |
| `gap` | $E_1 - E_0$ | Excitation gap |
| `susceptibility` | $\chi = \beta N (\langle M^2 \rangle - \langle M \rangle^2)$ | Fluctuations |
| `binder_cumulant` | $U = 1 - \langle m^4 \rangle / (3\langle m^2 \rangle^2)$ | Phase transition indicator |
| `correlation_function` | $C(r) = \langle \sigma_0^z \sigma_r^z \rangle$ | Spatial correlations |

## 📈 Visualization

```python
from quantum_ising_3d import Visualization

viz = Visualization(save_dir='./figures')

# Phase diagram
viz.plot_phase_diagram(results, title="3D TFIM Phase Diagram")

# Energy gap spectrum
viz.plot_gap_vs_Gamma(Gamma_values, gaps, Gamma_c=5.2)

# Annealing dynamics
viz.plot_annealing_dynamics(anneal_results)

# 3D spin configuration
viz.plot_spin_configuration_3d(spins, L=4)
```

## 🧪 Running Tests

```bash
cd tests
python test_simulation.py
```

Tests include:
- Pauli matrix algebra
- Hamiltonian Hermiticity
- Zero-field (classical) limit
- Infinite-field limit
- 1D exact solution comparison
- Energy bounds
- Gap positivity

## ⚡ Performance Notes

| System Size | Sites | Hilbert Dim | Method | Memory |
|------------|-------|-------------|--------|--------|
| 2×2×2 | 8 | 256 | Exact | ~1 MB |
| 3×3×3 | 27 | 134M | QMC only | ~100 MB |
| 4×4×4 | 64 | $10^{19}$ | QMC only | ~500 MB |

**Recommendations:**
- $L \leq 2$: Use exact diagonalization
- $L \geq 3$: Use Quantum Monte Carlo
- For optimization: Use quantum annealing (exact for small L, simulated for large L)

## 📖 References

1. Sachdev, S. "Quantum Phase Transitions" (Cambridge, 2011)
2. Sandvik, A.W. "Stochastic Series Expansion", PRB 59, R14157 (1999)
3. Suzuki, M. "Trotter decomposition", Prog. Theor. Phys. 56, 1454 (1976)
4. Farhi et al. "Quantum Annealing", Science 292, 472 (2001)

## 🔧 Dependencies

- `numpy>=1.21.0`: Numerical operations
- `scipy>=1.7.0`: Sparse matrices, diagonalization
- `matplotlib>=3.5.0`: Visualization
- `numba>=0.55.0`: JIT compilation for QMC
- `tqdm>=4.62.0`: Progress bars

## 📝 License

MIT License

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

---

*Quantum simulation of the 3D Transverse Field Ising Model - studying quantum phase transitions at the intersection of condensed matter physics and quantum computing.*
