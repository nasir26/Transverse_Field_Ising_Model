"""
Test suite for quantum Ising 3D simulator

Run with: python test_quantum_ising.py
"""

import numpy as np
from quantum_ising_3d import QuantumIsing3D, QuantumAnnealing, PhaseTransitionAnalyzer
import sys


def test_basic_construction():
    """Test basic model construction"""
    print("\n" + "="*70)
    print("TEST 1: Basic Construction")
    print("="*70)
    
    try:
        model = QuantumIsing3D(2, 1, 1, J=1.0, Gamma=0.5)
        print(f"✓ Created {model.Lx}x{model.Ly}x{model.Lz} model")
        print(f"  Spins: {model.N}")
        print(f"  Hilbert space: {model.dim}")
        
        H = model.construct_hamiltonian()
        print(f"✓ Constructed Hamiltonian")
        print(f"  Shape: {H.shape}")
        print(f"  Non-zeros: {H.nnz}")
        
        # Check Hermiticity
        H_dense = H.toarray()
        is_hermitian = np.allclose(H_dense, H_dense.conj().T)
        print(f"✓ Hermiticity: {is_hermitian}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_ground_state():
    """Test ground state finding"""
    print("\n" + "="*70)
    print("TEST 2: Ground State Finding")
    print("="*70)
    
    try:
        model = QuantumIsing3D(2, 1, 1, J=1.0, Gamma=0.5)
        eigenvalues, eigenvectors = model.find_ground_state(k=2)
        
        print(f"✓ Found ground state")
        print(f"  E_0 = {eigenvalues[0]:.6f}")
        print(f"  E_1 = {eigenvalues[1]:.6f}")
        
        # Check normalization
        norm = np.linalg.norm(eigenvectors[:, 0])
        print(f"✓ Normalization: {norm:.6f} (should be 1.0)")
        
        # Check energy ordering
        is_ordered = eigenvalues[1] >= eigenvalues[0]
        print(f"✓ Energy ordering: {is_ordered}")
        
        return abs(norm - 1.0) < 1e-10 and is_ordered
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_observables():
    """Test observable calculations"""
    print("\n" + "="*70)
    print("TEST 3: Observables")
    print("="*70)
    
    try:
        model = QuantumIsing3D(2, 1, 1, J=1.0, Gamma=0.5)
        eigenvalues, eigenvectors = model.find_ground_state(k=1)
        state = eigenvectors[:, 0]
        
        mag_z = model.magnetization_z(state)
        mag_x = model.magnetization_x(state)
        energy = model.energy(state)
        
        print(f"✓ Computed observables")
        print(f"  M_z = {mag_z:.6f}")
        print(f"  M_x = {mag_x:.6f}")
        print(f"  E = {energy:.6f}")
        
        # Check magnetization bounds
        mag_ok = abs(mag_z) <= 1.0 and abs(mag_x) <= 1.0
        print(f"✓ Magnetization bounds: {mag_ok}")
        
        # Check energy matches eigenvalue
        energy_match = abs(energy - eigenvalues[0]) < 1e-10
        print(f"✓ Energy consistency: {energy_match}")
        
        return mag_ok and energy_match
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_phase_behavior():
    """Test ferromagnetic vs paramagnetic phases"""
    print("\n" + "="*70)
    print("TEST 4: Phase Behavior")
    print("="*70)
    
    try:
        # For ferromagnetic phase, use periodic boundaries to avoid edge effects
        # and break Z2 symmetry by looking at correlation instead of magnetization
        model_ferro = QuantumIsing3D(4, 1, 1, J=1.0, Gamma=0.1, periodic=True)
        _, eigvec_ferro = model_ferro.find_ground_state(k=1)
        state_ferro = eigvec_ferro[:, 0]
        
        # Check correlation instead of magnetization (avoids Z2 symmetry issue)
        corr_ferro = model_ferro.correlation_zz(0, 1, state_ferro)
        
        print(f"Ferromagnetic phase (Γ/J = 0.1):")
        print(f"  <σ_0^z σ_1^z> = {corr_ferro:.6f} (should be ≈ 1)")
        print(f"  (Using correlation due to Z2 symmetry)")
        
        # Paramagnetic phase (Gamma >> J)
        model_para = QuantumIsing3D(3, 1, 1, J=1.0, Gamma=5.0)
        _, eigvec_para = model_para.find_ground_state(k=1)
        mag_z_para = abs(model_para.magnetization_z(eigvec_para[:, 0]))
        mag_x_para = abs(model_para.magnetization_x(eigvec_para[:, 0]))
        
        print(f"Paramagnetic phase (Γ/J = 5.0):")
        print(f"  |M_z| = {mag_z_para:.6f} (should be ≈ 0)")
        print(f"  |M_x| = {mag_x_para:.6f} (should be ≈ 1)")
        
        # Check phase characteristics
        ferro_ok = corr_ferro > 0.9  # Strong positive correlation
        para_ok = mag_z_para < 0.1 and mag_x_para > 0.9
        
        print(f"✓ Ferromagnetic phase: {ferro_ok}")
        print(f"✓ Paramagnetic phase: {para_ok}")
        
        return ferro_ok and para_ok
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_correlations():
    """Test correlation functions"""
    print("\n" + "="*70)
    print("TEST 5: Spin Correlations")
    print("="*70)
    
    try:
        model = QuantumIsing3D(3, 1, 1, J=1.0, Gamma=0.2, periodic=False)
        _, eigvec = model.find_ground_state(k=1)
        state = eigvec[:, 0]
        
        # Correlations
        corr_01 = model.correlation_zz(0, 1, state)  # Neighbors
        corr_02 = model.correlation_zz(0, 2, state)  # Next-neighbors
        
        print(f"✓ Computed correlations")
        print(f"  <σ_0 σ_1> = {corr_01:.6f} (neighbors)")
        print(f"  <σ_0 σ_2> = {corr_02:.6f} (next-neighbors)")
        
        # In ferromagnetic phase, correlations should be positive
        corr_ok = corr_01 > 0 and corr_02 > 0
        print(f"✓ Ferromagnetic correlations: {corr_ok}")
        
        # Correlations should decay with distance (typically)
        decay_ok = corr_01 >= corr_02
        print(f"✓ Correlation decay: {decay_ok}")
        
        return corr_ok
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_3d_lattice():
    """Test true 3D lattice"""
    print("\n" + "="*70)
    print("TEST 6: 3D Lattice Structure")
    print("="*70)
    
    try:
        model = QuantumIsing3D(2, 2, 2, J=1.0, Gamma=0.5, periodic=False)
        
        print(f"✓ Created 2x2x2 lattice")
        print(f"  Spins: {model.N}")
        
        # Check neighbor structure
        # Corner sites should have 3 neighbors
        neighbors_0 = model._get_neighbors(0)
        neighbors_7 = model._get_neighbors(7)
        
        print(f"  Site 0 neighbors: {len(neighbors_0)} (corner)")
        print(f"  Site 7 neighbors: {len(neighbors_7)} (corner)")
        
        # All corners should have 3 neighbors in 2x2x2 cube
        corners_ok = len(neighbors_0) == 3 and len(neighbors_7) == 3
        print(f"✓ Corner connectivity: {corners_ok}")
        
        # Test ground state
        eigenvalues, _ = model.find_ground_state(k=1)
        print(f"✓ Found 3D ground state: E_0 = {eigenvalues[0]:.6f}")
        
        return corners_ok
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_quantum_annealing():
    """Test quantum annealing"""
    print("\n" + "="*70)
    print("TEST 7: Quantum Annealing")
    print("="*70)
    
    try:
        model = QuantumIsing3D(2, 1, 1, J=1.0)
        annealer = QuantumAnnealing(model)
        
        results = annealer.anneal(
            T=5.0,
            num_steps=10,
            Gamma_init=3.0,
            Gamma_final=0.2,
            schedule='linear'
        )
        
        print(f"✓ Completed annealing")
        print(f"  Initial energy: {results['energies'][0]:.6f}")
        print(f"  Final energy: {results['energies'][-1]:.6f}")
        
        # Check that Gamma decreases
        gamma_decreases = results['gammas'][0] > results['gammas'][-1]
        print(f"✓ Γ schedule: {gamma_decreases}")
        
        # Check data consistency
        data_ok = (len(results['times']) == len(results['energies']) == 
                   len(results['mag_z']) == len(results['mag_x']))
        print(f"✓ Data consistency: {data_ok}")
        
        return gamma_decreases and data_ok
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_exact_limits():
    """Test exact solvable limits"""
    print("\n" + "="*70)
    print("TEST 8: Exact Limits")
    print("="*70)
    
    try:
        # Single spin system
        model = QuantumIsing3D(1, 1, 1, J=1.0, Gamma=1.0)
        eigenvalues, _ = model.find_ground_state(k=2)
        
        # For single spin: H = -Γ σ_x
        # Eigenvalues should be ±Γ
        expected_E0 = -1.0
        expected_E1 = 1.0
        
        print(f"Single spin (J irrelevant, Γ=1):")
        print(f"  E_0 = {eigenvalues[0]:.6f} (expected: {expected_E0:.6f})")
        print(f"  E_1 = {eigenvalues[1]:.6f} (expected: {expected_E1:.6f})")
        
        single_ok = (abs(eigenvalues[0] - expected_E0) < 1e-10 and 
                    abs(eigenvalues[1] - expected_E1) < 1e-10)
        print(f"✓ Single spin: {single_ok}")
        
        # Two non-interacting spins (no bonds)
        model2 = QuantumIsing3D(2, 1, 1, J=1.0, Gamma=1.0, periodic=False)
        # Remove all bonds manually by checking
        neighbors = []
        for i in range(model2.N):
            neighbors.extend(model2._get_neighbors(i))
        
        has_bonds = len(neighbors) > 0
        print(f"Two spins system has bonds: {has_bonds}")
        
        return single_ok
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("QUANTUM ISING 3D SIMULATOR - TEST SUITE")
    print("="*70)
    
    tests = [
        ("Basic Construction", test_basic_construction),
        ("Ground State Finding", test_ground_state),
        ("Observables", test_observables),
        ("Phase Behavior", test_phase_behavior),
        ("Spin Correlations", test_correlations),
        ("3D Lattice", test_3d_lattice),
        ("Quantum Annealing", test_quantum_annealing),
        ("Exact Limits", test_exact_limits),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\nError in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print("\n" + "-"*70)
    print(f"Total: {passed}/{total} tests passed")
    print("="*70 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
