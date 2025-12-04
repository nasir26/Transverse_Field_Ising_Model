# Theory: Transverse Field Ising Model

Detailed theoretical background for the quantum Ising model simulation.

## Table of Contents
1. [Classical Ising Model](#classical-ising-model)
2. [Transverse Field Ising Model](#transverse-field-ising-model)
3. [Quantum Phase Transition](#quantum-phase-transition)
4. [Exact Solutions](#exact-solutions)
5. [Computational Methods](#computational-methods)
6. [Physical Observables](#physical-observables)

---

## Classical Ising Model

### Hamiltonian

The classical Ising model describes interacting spins on a lattice:

```
H_classical = -J ∑_<i,j> s_i s_j - h ∑_i s_i
```

where:
- `s_i = ±1` are classical spin variables
- `J > 0` is ferromagnetic coupling
- `h` is external magnetic field
- `<i,j>` denotes nearest-neighbor pairs

### Ground State (T=0)

- **J > 0, h = 0**: Ferromagnetic ground state
  - Two degenerate states: all spins ↑ or all spins ↓
  - Energy: `E_0 = -J × (number of bonds)`

- **Excited states**: Domain walls separate aligned regions
  - Cost: `ΔE ∝ J × (surface area)`

### Phase Transition (T > 0)

At finite temperature, thermal fluctuations disorder the system:

- **T < T_c**: Ordered phase (spontaneous magnetization)
- **T > T_c**: Disordered phase (zero magnetization)
- **T = T_c**: Critical point

Critical temperature (mean-field):
```
k_B T_c ≈ z J
```
where z = coordination number (neighbors per site)

### Dimensionality

Critical behavior depends on dimension:

| Dimension | T_c | Notes |
|-----------|-----|-------|
| 1D | 0 | No order at T > 0 |
| 2D | >0 | Onsager solution |
| 3D | >0 | Mean-field valid |

---

## Transverse Field Ising Model

### Quantum Hamiltonian

Adding a transverse field makes the model quantum:

```
H = -J ∑_<i,j> σ_i^z σ_j^z - Γ ∑_i σ_i^x
```

where `σ^{x,z}` are Pauli matrices:

```
σ^z = [1   0 ]     σ^x = [0  1]
      [0  -1]            [1  0]
```

### Physical Interpretation

1. **Ising term** (`-J σ^z σ^z`):
   - Diagonal in computational basis |↑⟩, |↓⟩
   - Favors aligned spins (ferromagnetic)
   - Classical energy term

2. **Transverse field** (`-Γ σ^x`):
   - Off-diagonal in computational basis
   - Causes quantum spin flips: |↑⟩ ↔ |↓⟩
   - Quantum fluctuation term

3. **Competition**:
   - J tries to align spins
   - Γ tries to create superpositions
   - Balance determines ground state

### Computational Basis

States written as product states:

```
|s_1, s_2, ..., s_N⟩ where s_i ∈ {↑, ↓}
```

Example for N=3:
```
|↑↑↑⟩, |↑↑↓⟩, |↑↓↑⟩, ..., |↓↓↓⟩
```

Total: `2^N` basis states

### Matrix Representation

For N spins, Hamiltonian is `2^N × 2^N` matrix:

- **Ising term**: Diagonal elements
  - Energy of each configuration
  
- **Transverse field**: Off-diagonal elements
  - Connects states differing by one spin flip

---

## Quantum Phase Transition

### Zero Temperature Transition

At T=0, system exhibits quantum phase transition as function of Γ/J:

```
    Γ << J          Γ ≈ Γ_c          Γ >> J
   ────────────────────────────────────────
   Ferro phase      Critical      Para phase
   Classical        Quantum       Quantum
   |M_z| ≈ 1       |M_z| ≈ 0.5    |M_z| ≈ 0
   |M_x| ≈ 0       |M_x| varies   |M_x| ≈ 1
```

### Critical Point

Location of critical point depends on dimension:

| System | Γ_c/J | Method |
|--------|-------|--------|
| 1D chain (PBC) | 1.000 | Exact |
| 2D square | ~3.0 | Numerical |
| 3D cubic | ~5.0 | Numerical |

### Order Parameters

1. **Ferromagnetic order**: `M_z = ⟨σ^z⟩`
   - M_z ≠ 0 for Γ < Γ_c
   - M_z = 0 for Γ > Γ_c

2. **Paramagnetic order**: `M_x = ⟨σ^x⟩`
   - M_x ≈ 0 for Γ < Γ_c
   - M_x ≈ 1 for Γ > Γ_c

### Critical Scaling

Near critical point, observables scale as:

```
|M_z| ∼ |Γ - Γ_c|^β           (order parameter)
χ ∼ |Γ - Γ_c|^(-γ)            (susceptibility)
ξ ∼ |Γ - Γ_c|^(-ν)            (correlation length)
Δ ∼ |Γ - Γ_c|^(zν)            (energy gap)
```

Critical exponents (1D transverse Ising):
- β = 1/8
- γ = 7/4
- ν = 1
- z = 1

### Correlation Length

At T=0, quantum correlations:

```
⟨σ_i^z σ_j^z⟩ - ⟨σ_i^z⟩⟨σ_j^z⟩ ∼ e^(-|r_i - r_j|/ξ)
```

Correlation length ξ:
- ξ → ∞ at Γ = Γ_c (critical)
- ξ ~ J/|Γ - Γ_c| for Γ ≈ Γ_c

---

## Exact Solutions

### 1D Chain: Jordan-Wigner Transformation

For 1D systems, Jordan-Wigner maps spins → fermions:

```
c_j = σ_j^- exp(iπ ∑_{l<j} σ_l^+ σ_l^-)
σ_j^z = 2c_j† c_j - 1
```

This transforms the Hamiltonian into free fermions:

```
H = ∑_k ε(k) c_k† c_k
```

with dispersion:

```
ε(k) = 2[J cos(k) - Γ]
```

#### Ground State Energy

```
E_0 = -∫_{-π}^π (dk/2π) |ε(k)|
```

#### Critical Point

Gap closes when min(ε(k)) = 0:

```
Γ_c = J    (1D with periodic boundaries)
```

### Single Spin

For N=1, only transverse field matters:

```
H = -Γ σ^x
```

Eigenvalues: `E = ±Γ`

Ground state: `|+⟩ = (|↑⟩ + |↓⟩)/√2`

### Two Spins (No Interaction)

If spins don't interact (J=0):

```
H = -Γ(σ_1^x + σ_2^x)
```

Product state ground state:
```
|ψ_0⟩ = |+⟩ ⊗ |+⟩
E_0 = -2Γ
```

---

## Computational Methods

### Exact Diagonalization

For small systems (N ≤ 20):

1. **Construct Hamiltonian**:
   - Build sparse matrix using tensor products
   - Size: `2^N × 2^N`
   - Sparsity: ~`2N × 2^N` non-zero elements

2. **Diagonalize**:
   - Use sparse eigenvalue solvers
   - Find k lowest eigenpairs
   - Algorithms: Lanczos, Arnoldi

3. **Compute Observables**:
   - Expectation values: `⟨O⟩ = ⟨ψ|O|ψ⟩`
   - Correlations, structure factor, etc.

**Limitations**:
- Memory: O(2^N) for state vectors
- Time: O(2^N × iterations) for eigensolving

### Quantum Monte Carlo

For larger systems at finite T:

1. **Path Integral Monte Carlo**:
   - Map to (d+1)-dimensional classical system
   - Imaginary time = extra dimension

2. **Variational Monte Carlo**:
   - Ansatz wavefunction with parameters
   - Optimize to minimize energy

### Tensor Networks

For 1D/2D at T=0:

1. **Matrix Product States (MPS)**:
   - Efficient for 1D systems
   - Captures area-law entanglement

2. **Projected Entangled Pair States (PEPS)**:
   - Extension to 2D
   - More challenging numerically

---

## Physical Observables

### Energy

Ground state energy density:

```
e_0 = E_0/N
```

In ferromagnetic phase (Γ → 0):
```
e_0 → -J × z/2
```
where z = coordination number

In paramagnetic phase (Γ → ∞):
```
e_0 → -Γ
```

### Magnetization

**Z-magnetization** (order parameter):
```
M_z = (1/N) ∑_i ⟨σ_i^z⟩
```

**X-magnetization**:
```
M_x = (1/N) ∑_i ⟨σ_i^x⟩
```

### Susceptibility

Response to field perturbation:

```
χ_z = ∂M_z/∂h_z
    = (1/N) ∑_i ⟨σ_i^z⟩^2 - |M_z|^2
```

Diverges at critical point: `χ ∼ |Γ - Γ_c|^(-γ)`

### Correlation Functions

**Equal-time correlations**:
```
C(r) = ⟨σ_i^z σ_{i+r}^z⟩ - ⟨σ_i^z⟩⟨σ_{i+r}^z⟩
```

**Structure factor** (Fourier transform):
```
S(q) = ∑_r e^(iq·r) C(r)
```

Peaks at q=0 in ferromagnetic phase.

### Entanglement Entropy

For bipartition A∪B:

```
S_A = -Tr(ρ_A ln ρ_A)
```

where `ρ_A = Tr_B(|ψ⟩⟨ψ|)`

**Area law**: S_A ∼ |∂A| (boundary area)
- Holds in gapped phases
- Violated at critical points: S_A ∼ |∂A| ln L

### Energy Gap

Excitation gap:

```
Δ = E_1 - E_0
```

- Gapped phases: Δ > 0
- Critical point: Δ → 0
- Scaling: Δ ∼ L^(-z) at criticality

---

## Quantum Annealing

### Adiabatic Theorem

If Hamiltonian changes slowly enough:

```
d/dt H(t),    t ∈ [0,T]
```

System remains in instantaneous ground state provided:

```
T >> ℏ/Δ_min^2
```

where `Δ_min` is minimum gap along path.

### Annealing Schedule

Time-dependent transverse field:

```
Γ(t) = Γ_0 + (Γ_f - Γ_0) × s(t/T)
```

**Schedules**:
1. Linear: `s(x) = x`
2. Polynomial: `s(x) = 3x^2 - 2x^3`
3. Exponential: `s(x) = [exp(αx) - 1]/[exp(α) - 1]`

### Performance

Success probability depends on:

1. **Annealing time T**:
   - Longer T → higher success
   - Adiabatic condition: `T ∝ 1/Δ_min^2`

2. **Minimum gap**:
   - Often at critical point
   - Small gap → slow down needed

3. **Schedule**:
   - Optimize for problem
   - Pause near critical region

### Applications

- Optimization problems
- Sampling from Boltzmann distribution
- Quantum algorithm design
- Machine learning

---

## Relation to Other Models

### Quantum Rotor Model

In path integral formulation, equivalent to:

```
H = (1/2I) L_z^2 - K cos(θ)
```

### XY Model

Generalization with different couplings:

```
H = -J_x ∑ σ_i^x σ_j^x - J_z ∑ σ_i^z σ_j^z - Γ ∑ σ_i^x
```

### Heisenberg Model

Isotropic spin exchange:

```
H = -J ∑ (σ_i^x σ_j^x + σ_i^y σ_j^y + σ_i^z σ_j^z)
```

---

## References

1. **Sachdev, S.** "Quantum Phase Transitions", Cambridge (2011)
2. **Pfeuty, P.** "The one-dimensional Ising model with a transverse field", Ann. Phys. 57, 79 (1970)
3. **Suzuki, S. et al.** "Quantum Ising Phases and Transitions in Transverse Ising Models", Springer (2012)
4. **Kadowaki, T. & Nishimori, H.** "Quantum annealing in the transverse Ising model", PRE 58, 5355 (1998)

---

*For implementation details, see README.md and code documentation.*
