"""
Example usage of the 3D Transverse Field Ising Model simulator

This script demonstrates various use cases and analysis methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from ising_3d_quantum import TransverseFieldIsing3D


def example_single_point():
    """Example: Compute ground state at a single point"""
    print("\n" + "="*60)
    print("Example 1: Single Point Calculation")
    print("="*60)
    
    # Create model
    model = TransverseFieldIsing3D(Lx=2, Ly=2, Lz=2, J=1.0, Gamma=0.5)
    
    # Compute ground state
    E0, psi0 = model.compute_ground_state()
    
    # Compute observables
    M_z = model.compute_magnetization(psi0)
    M_x = model.compute_transverse_magnetization(psi0)
    
    print(f"\nResults:")
    print(f"  Ground state energy: E0 = {E0:.6f}")
    print(f"  Energy per spin: E0/N = {E0/model.N:.6f}")
    print(f"  Longitudinal magnetization: <M_z> = {M_z:.6f}")
    print(f"  Transverse magnetization: <M_x> = {M_x:.6f}")
    
    # Compute a correlation function
    if model.N >= 2:
        corr = model.compute_correlation(psi0, 0, 1)
        print(f"  Spin-spin correlation <σ₀ᵢ σ₁ᵢ>: {corr:.6f}")


def example_phase_transition():
    """Example: Study phase transition"""
    print("\n" + "="*60)
    print("Example 2: Phase Transition Analysis")
    print("="*60)
    
    # Create model
    model = TransverseFieldIsing3D(Lx=2, Ly=2, Lz=2, J=1.0, Gamma=1.0)
    
    # Scan across transverse field values
    gamma_values = np.linspace(0.1, 2.0, 15) * model.J
    
    print(f"\nScanning {len(gamma_values)} values of Γ/J from {gamma_values[0]/model.J:.2f} to {gamma_values[-1]/model.J:.2f}")
    
    # Perform scan
    results = model.phase_transition_scan(gamma_values)
    
    # Visualize
    model.visualize_phase_transition(results, save_path='phase_transition_3d.png')
    
    # Find approximate critical point (where M_z drops significantly)
    M_z = results['magnetization_z']
    gamma_ratio = results['gamma'] / model.J
    
    # Find where magnetization drops to half its maximum
    M_z_max = np.max(np.abs(M_z))
    critical_idx = np.where(np.abs(M_z) < 0.5 * M_z_max)[0]
    if len(critical_idx) > 0:
        gamma_c = gamma_ratio[critical_idx[0]]
        print(f"\nApproximate critical point: Γ_c/J ≈ {gamma_c:.3f}")


def example_energy_spectrum():
    """Example: Compute low-lying energy spectrum"""
    print("\n" + "="*60)
    print("Example 3: Low-lying Energy Spectrum")
    print("="*60)
    
    model = TransverseFieldIsing3D(Lx=2, Ly=2, Lz=2, J=1.0, Gamma=1.0)
    
    # Compute first 5 eigenvalues
    print("\nComputing first 5 eigenvalues...")
    H = model._build_hamiltonian()
    from scipy.sparse.linalg import eigs
    eigenvalues, _ = eigs(H, k=min(5, 2**model.N-1), which='SR')
    eigenvalues = np.sort(np.real(eigenvalues))
    
    print(f"\nLow-lying energy spectrum:")
    for i, E in enumerate(eigenvalues):
        print(f"  E_{i} = {E:.6f} (gap from ground state: {E - eigenvalues[0]:.6f})")


def example_different_sizes():
    """Example: Compare different lattice sizes"""
    print("\n" + "="*60)
    print("Example 4: Size Comparison")
    print("="*60)
    
    sizes = [(2, 2, 2), (2, 2, 3)]
    J = 1.0
    Gamma = 1.0
    
    results = {}
    
    for Lx, Ly, Lz in sizes:
        print(f"\nLattice: {Lx}×{Ly}×{Lz} = {Lx*Ly*Lz} spins")
        print(f"Hilbert space: 2^{Lx*Ly*Lz} = {2**(Lx*Ly*Lz)} states")
        
        try:
            model = TransverseFieldIsing3D(Lx, Ly, Lz, J=J, Gamma=Gamma)
            E0, psi0 = model.compute_ground_state()
            M_z = model.compute_magnetization(psi0)
            
            results[(Lx, Ly, Lz)] = {
                'E0': E0,
                'E0_per_spin': E0 / model.N,
                'M_z': M_z
            }
            
            print(f"  E0/N = {E0/model.N:.6f}")
            print(f"  <M_z> = {M_z:.6f}")
            
        except MemoryError:
            print(f"  ERROR: System too large for available memory")
            break
    
    # Compare results
    if len(results) > 1:
        print("\nComparison:")
        for size, data in results.items():
            print(f"  {size[0]}×{size[1]}×{size[2]}: E0/N = {data['E0_per_spin']:.6f}, <M_z> = {data['M_z']:.6f}")


def main():
    """Run all examples"""
    print("="*60)
    print("3D Transverse Field Ising Model - Example Usage")
    print("="*60)
    
    # Run examples
    example_single_point()
    example_phase_transition()
    example_energy_spectrum()
    example_different_sizes()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
