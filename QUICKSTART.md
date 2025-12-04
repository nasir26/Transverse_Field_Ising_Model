# Quick Start Guide

Get up and running with the Quantum Ising 3D simulator in 5 minutes!

## Installation

```bash
# Install dependencies
pip install numpy scipy matplotlib

# Clone repository (if not already done)
git clone <repo-url>
cd quantum-ising-3d
```

## 30-Second Example

```python
from quantum_ising_3d import QuantumIsing3D

# Create 2×2×1 lattice
model = QuantumIsing3D(2, 2, 1, J=1.0, Gamma=0.5)

# Find ground state
eigenvalues, eigenvectors = model.find_ground_state()
ground_state = eigenvectors[:, 0]

# Compute observables
print(f"Energy: {eigenvalues[0]:.4f}")
print(f"M_z: {model.magnetization_z(ground_state):.4f}")
print(f"M_x: {model.magnetization_x(ground_state):.4f}")
```

## Run Tests

```bash
python3 test_quantum_ising.py
```

Expected output: `Total: 8/8 tests passed`

## Run Demo

```bash
python3 demo.py
```

This runs 4 quick simulations and generates plots.

## Interactive Tutorial

```bash
jupyter notebook tutorial.ipynb
```

Or use any Jupyter-compatible environment (JupyterLab, VS Code, Google Colab).

## Examples

### 1. Ground State Analysis
```bash
python3 examples/example_1_ground_state.py
```

### 2. Phase Transition
```bash
python3 examples/example_2_phase_transition.py
```

### 3. Quantum Annealing
```bash
python3 examples/example_3_quantum_annealing.py
```

### 4. 3D Systems
```bash
python3 examples/example_4_3d_system.py
```

## What Can You Do?

### Study Ground States
```python
model = QuantumIsing3D(3, 1, 1, J=1.0, Gamma=0.5)
E, psi = model.find_ground_state(k=1)
```

### Scan Phase Transitions
```python
from quantum_ising_3d import PhaseTransitionAnalyzer
import numpy as np

analyzer = PhaseTransitionAnalyzer(4, 1, 1, J=1.0)
gammas = np.linspace(0.1, 3.0, 15)
results = analyzer.scan_transverse_field(gammas)
```

### Quantum Annealing
```python
from quantum_ising_3d import QuantumAnnealing

model = QuantumIsing3D(3, 1, 1, J=1.0)
annealer = QuantumAnnealing(model)

results = annealer.anneal(
    T=10.0,
    num_steps=50,
    Gamma_init=5.0,
    Gamma_final=0.1
)
```

### Visualize Results
```python
from quantum_ising_3d import visualize_results
import matplotlib.pyplot as plt

fig = visualize_results(results)
plt.show()
```

## System Size Guide

| Spins | Hilbert Dim | Memory | Time | Recommended |
|-------|-------------|--------|------|-------------|
| 3 | 8 | < 1 KB | < 1s | ✓ Demo |
| 4 | 16 | < 1 KB | < 1s | ✓ Learning |
| 8 | 256 | ~1 KB | ~1s | ✓ Small 3D |
| 10 | 1,024 | ~10 KB | ~5s | ✓ Max for tutorial |
| 12 | 4,096 | ~100 KB | ~30s | ⚠ Slow |
| 15 | 32,768 | ~8 MB | ~5min | ⚠ Very slow |
| 20 | 1,048,576 | ~8 GB | hours | ✗ Impractical |

**Recommendation**: Start with N ≤ 10 spins.

## Common Issues

### "ModuleNotFoundError: No module named 'numpy'"
```bash
pip install numpy scipy matplotlib
```

### "Memory error" or system hangs
- System too large!
- Use smaller lattice (N ≤ 12)

### Slow computation
- Reduce `num_steps` in annealing
- Reduce number of Γ values in phase scan
- Use smaller system

## Understanding the Physics

### Parameters

- **J**: Coupling strength (positive = ferromagnetic)
- **Γ** (Gamma): Transverse field strength
- **Γ/J ratio**: Determines phase
  - Γ/J << 1: Ferromagnetic (classical)
  - Γ/J >> 1: Paramagnetic (quantum)
  - Γ/J ≈ 1: Critical point (1D)

### Observables

- **M_z**: Order parameter
  - |M_z| ≈ 1 → ordered phase
  - |M_z| ≈ 0 → disordered phase
  
- **M_x**: Quantum magnetization
  - Large in paramagnetic phase
  
- **Energy gap**: ΔE = E₁ - E₀
  - Closes at quantum critical point

### Lattice Types

```python
# 1D chain
model = QuantumIsing3D(N, 1, 1)

# 2D square
model = QuantumIsing3D(Lx, Ly, 1)

# 3D cubic
model = QuantumIsing3D(Lx, Ly, Lz)

# Periodic boundaries
model = QuantumIsing3D(N, 1, 1, periodic=True)
```

## Next Steps

1. ✓ Run tests and demo
2. ✓ Try examples
3. ✓ Work through tutorial notebook
4. Read [README.md](README.md) for full API
5. Read [THEORY.md](THEORY.md) for physics details
6. Experiment with your own systems!

## Getting Help

- **Documentation**: See README.md
- **Theory**: See THEORY.md
- **Examples**: Check examples/ directory
- **Tests**: Look at test_quantum_ising.py for usage patterns

## Citation

If you use this code in research:

```bibtex
@software{quantum_ising_3d,
  title={Quantum Simulation of 3D Transverse Field Ising Model},
  year={2025},
  url={https://github.com/yourusername/quantum-ising-3d}
}
```

---

**Happy simulating!** 🎉

For questions or issues, please open a GitHub issue.
