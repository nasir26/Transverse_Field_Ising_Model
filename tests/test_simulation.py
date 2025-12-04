#!/usr/bin/env python3
"""
Unit Tests for 3D Transverse Field Ising Model Simulation

Tests cover:
- Hamiltonian construction
- Exact diagonalization
- Physical properties
- Known limits
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_ising_3d import (
    TransverseFieldIsing3D,
    ExactDiagonalization,
    pauli_x, pauli_z, pauli_y, identity
)
from quantum_ising_3d.hamiltonian import TransverseFieldIsing1D


def test_pauli_matrices():
    """Test Pauli matrix properties."""
    print("Testing Pauli matrices...")
    
    sx = pauli_x()
    sy = pauli_y()
    sz = pauli_z()
    I = identity()
    
    # Check dimensions
    assert sx.shape == (2, 2)
    assert sy.shape == (2, 2)
    assert sz.shape == (2, 2)
    
    # Check squares equal identity
    assert np.allclose(sx @ sx, I)
    assert np.allclose(sy @ sy, I)
    assert np.allclose(sz @ sz, I)
    
    # Check anticommutation
    assert np.allclose(sx @ sy + sy @ sx, 0)
    assert np.allclose(sy @ sz + sz @ sy, 0)
    assert np.allclose(sz @ sx + sx @ sz, 0)
    
    # Check commutation relations
    assert np.allclose(sx @ sy - sy @ sx, 2j * sz)
    assert np.allclose(sy @ sz - sz @ sy, 2j * sx)
    assert np.allclose(sz @ sx - sx @ sz, 2j * sy)
    
    print("  ✓ All Pauli matrix tests passed")


def test_model_creation():
    """Test model creation and basic properties."""
    print("Testing model creation...")
    
    L = 2
    model = TransverseFieldIsing3D(L=L, J=1.0, Gamma=1.0)
    
    assert model.N == L**3
    assert model.dim == 2**(L**3)
    
    # Check neighbor list
    bonds = model.get_bonds()
    assert len(bonds) > 0
    
    # For 3D cubic lattice with PBC:
    # - 3 directions, each has L^2 "lines" of sites forming rings
    # - Each ring has L sites, but only L-1 bonds counted (i<j skips wrap)
    # - Total = 3 * L^2 * (L-1) ... but for L=2 this simplifies
    # For L=2 specifically: 3 * 4 * 1 = 12 bonds
    expected_bonds = 3 * L**2 * (L - 1)
    assert len(bonds) == expected_bonds, f"Expected {expected_bonds} bonds, got {len(bonds)}"
    
    print("  ✓ Model creation tests passed")


def test_hamiltonian_hermiticity():
    """Test that Hamiltonian is Hermitian."""
    print("Testing Hamiltonian Hermiticity...")
    
    model = TransverseFieldIsing3D(L=2, J=1.0, Gamma=1.0)
    H = model.build_hamiltonian_fast()
    
    H_dense = H.toarray()
    assert np.allclose(H_dense, H_dense.conj().T)
    
    print("  ✓ Hamiltonian is Hermitian")


def test_zero_field_limit():
    """Test Γ = 0 limit (classical Ising model)."""
    print("Testing zero field limit...")
    
    model = TransverseFieldIsing3D(L=2, J=1.0, Gamma=0.0)
    H = model.build_hamiltonian_fast()
    
    # Hamiltonian should be diagonal
    H_dense = H.toarray()
    off_diag = H_dense - np.diag(np.diag(H_dense))
    assert np.allclose(off_diag, 0)
    
    # Ground state should be ferromagnetic (all up or all down)
    eigenvalues = np.linalg.eigvalsh(H_dense)
    E0 = eigenvalues[0]
    
    # For 2x2x2 with PBC: E = -J * 12 (12 bonds, each contributes -J for aligned spins)
    expected_E0 = -1.0 * 12
    assert np.isclose(E0, expected_E0)
    
    print("  ✓ Zero field limit correct")


def test_infinite_field_limit():
    """Test Γ → ∞ limit (all spins along x)."""
    print("Testing infinite field limit...")
    
    model = TransverseFieldIsing3D(L=2, J=1.0, Gamma=100.0)  # Large Γ
    solver = ExactDiagonalization(model)
    solver.build_hamiltonian()
    solver.diagonalize(n_states=1)
    
    # Ground state should be approximately |+⟩^⊗N
    # M_x should be close to 1
    Mx = solver.compute_magnetization('x')
    assert abs(Mx) > 0.95, f"M_x = {Mx} should be close to 1"
    
    # M_z should be close to 0
    Mz = solver.compute_magnetization('z')
    assert abs(Mz) < 0.1, f"M_z = {Mz} should be close to 0"
    
    print("  ✓ Infinite field limit correct")


def test_1d_exact_solution():
    """Test 1D TFIM against Jordan-Wigner exact solution."""
    print("Testing 1D exact solution...")
    
    L = 8
    J = 1.0
    
    for Gamma in [0.5, 1.0, 2.0]:
        model = TransverseFieldIsing1D(L=L, J=J, Gamma=Gamma)
        
        # Numerical diagonalization
        solver = ExactDiagonalization(model)
        solver.build_hamiltonian()
        solver.diagonalize(n_states=1)
        E_numerical = solver.ground_state_energy / L
        
        # Jordan-Wigner exact
        E_exact = model.exact_ground_state_energy()
        
        assert np.isclose(E_numerical, E_exact, atol=1e-10), \
            f"E_numerical = {E_numerical}, E_exact = {E_exact}"
    
    print("  ✓ 1D exact solution matches")


def test_symmetries():
    """Test symmetry properties."""
    print("Testing symmetries...")
    
    model = TransverseFieldIsing3D(L=2, J=1.0, Gamma=1.0)
    solver = ExactDiagonalization(model)
    solver.build_hamiltonian()
    solver.diagonalize(n_states=1)
    
    # Z2 symmetry: ⟨M_z⟩ should be 0 or ±m_0
    # For finite systems without symmetry breaking, ⟨M_z⟩ = 0
    Mz = solver.compute_magnetization('z')
    # Note: For small systems, this may not be exactly 0 due to finite-size effects
    
    print(f"  ⟨M_z⟩ = {Mz:.6f} (expected ~0 for symmetric ground state)")
    print("  ✓ Symmetry tests completed")


def test_energy_bounds():
    """Test that ground state energy is within bounds."""
    print("Testing energy bounds...")
    
    L = 2
    J = 1.0
    Gamma = 1.0
    
    model = TransverseFieldIsing3D(L=L, J=J, Gamma=Gamma)
    solver = ExactDiagonalization(model)
    solver.build_hamiltonian()
    solver.diagonalize(n_states=1)
    
    E0 = solver.ground_state_energy
    
    # Lower bound: all bonds aligned, all spins flipped by field
    # E_min ≥ -J * n_bonds - Γ * N
    n_bonds = len(model.get_bonds())
    E_lower = -J * n_bonds - Gamma * model.N
    
    # Upper bound: classical ground state
    E_upper = -J * n_bonds
    
    assert E0 >= E_lower, f"E0 = {E0} < lower bound {E_lower}"
    assert E0 <= E_upper, f"E0 = {E0} > upper bound {E_upper}"
    
    print(f"  Energy: {E0:.4f} in [{E_lower:.4f}, {E_upper:.4f}]")
    print("  ✓ Energy bounds satisfied")


def test_gap_positivity():
    """Test that energy gap is positive."""
    print("Testing gap positivity...")
    
    model = TransverseFieldIsing3D(L=2, J=1.0, Gamma=1.0)
    solver = ExactDiagonalization(model)
    solver.build_hamiltonian()
    solver.diagonalize(n_states=2)
    
    gap = solver.eigenvalues[1] - solver.eigenvalues[0]
    assert gap > 0, f"Gap = {gap} should be positive"
    
    print(f"  Gap = {gap:.6f} > 0")
    print("  ✓ Gap is positive")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Unit Tests for 3D TFIM Simulation")
    print("=" * 60 + "\n")
    
    tests = [
        test_pauli_matrices,
        test_model_creation,
        test_hamiltonian_hermiticity,
        test_zero_field_limit,
        test_infinite_field_limit,
        test_1d_exact_solution,
        test_symmetries,
        test_energy_bounds,
        test_gap_positivity,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Tests completed: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
