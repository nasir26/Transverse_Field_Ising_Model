#!/usr/bin/env python3
"""
Basic Simulation Example for 3D Transverse Field Ising Model

This example demonstrates the fundamental usage of the simulation package
for studying the quantum phase transition in the 3D TFIM.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_ising_3d import (
    TransverseFieldIsing3D,
    ExactDiagonalization,
    Visualization
)


def main():
    print("=" * 60)
    print("Basic 3D TFIM Simulation")
    print("=" * 60)
    
    # ==================================================================
    # 1. Create the model
    # ==================================================================
    print("\n1. Creating 2x2x2 lattice model...")
    
    L = 2      # Linear size (8 sites total)
    J = 1.0    # Ising coupling
    Gamma = 1.0  # Transverse field
    
    model = TransverseFieldIsing3D(L=L, J=J, Gamma=Gamma, periodic=True)
    model.analyze_system()
    
    # ==================================================================
    # 2. Build and diagonalize Hamiltonian
    # ==================================================================
    print("\n2. Exact diagonalization...")
    
    solver = ExactDiagonalization(model)
    solver.build_hamiltonian()
    eigenvalues, eigenvectors = solver.diagonalize(n_states=4)
    
    print(f"\nLowest 4 eigenvalues:")
    for i, E in enumerate(eigenvalues):
        print(f"  E_{i} = {E:.6f}")
    
    # ==================================================================
    # 3. Compute ground state properties
    # ==================================================================
    print("\n3. Ground state properties...")
    
    E0, psi0 = solver.get_ground_state()
    
    Mz = solver.compute_magnetization('z')
    Mx = solver.compute_magnetization('x')
    gap = solver.compute_energy_gap()
    
    print(f"\n  Ground state energy: {E0:.6f}")
    print(f"  Energy per site: {E0/model.N:.6f}")
    print(f"  Magnetization ⟨M_z⟩: {Mz:.6f}")
    print(f"  Magnetization ⟨M_x⟩: {Mx:.6f}")
    print(f"  Energy gap: {gap:.6f}")
    
    # ==================================================================
    # 4. Scan over transverse field Γ
    # ==================================================================
    print("\n4. Scanning phase diagram...")
    
    Gamma_values = np.linspace(0.1, 5.0, 25)
    
    energies = []
    mz_values = []
    mx_values = []
    gaps = []
    
    for Gamma in Gamma_values:
        # Update model
        model.Gamma = Gamma
        solver.H = None
        solver.eigenvalues = None
        solver.ground_state = None
        
        # Solve
        solver.build_hamiltonian()
        solver.diagonalize(n_states=2)
        
        # Measure
        energies.append(solver.ground_state_energy / model.N)
        mz_values.append(solver.compute_magnetization('z'))
        mx_values.append(solver.compute_magnetization('x'))
        gaps.append(solver.eigenvalues[1] - solver.eigenvalues[0])
    
    # ==================================================================
    # 5. Plot results
    # ==================================================================
    print("\n5. Plotting results...")
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Energy
    ax = axes[0, 0]
    ax.plot(Gamma_values, energies, 'bo-', markersize=4)
    ax.set_xlabel(r'$\Gamma/J$')
    ax.set_ylabel(r'$E_0/N$')
    ax.set_title('Ground State Energy per Site')
    ax.grid(True, alpha=0.3)
    
    # Order parameters
    ax = axes[0, 1]
    ax.plot(Gamma_values, np.abs(mz_values), 'ro-', markersize=4, label=r'$|\langle M_z \rangle|$')
    ax.plot(Gamma_values, mx_values, 'bs-', markersize=4, label=r'$\langle M_x \rangle$')
    ax.set_xlabel(r'$\Gamma/J$')
    ax.set_ylabel('Order Parameter')
    ax.set_title('Magnetization')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gap
    ax = axes[1, 0]
    ax.plot(Gamma_values, gaps, 'go-', markersize=4)
    ax.set_xlabel(r'$\Gamma/J$')
    ax.set_ylabel(r'$\Delta$')
    ax.set_title('Energy Gap')
    ax.grid(True, alpha=0.3)
    
    # Phase identification
    ax = axes[1, 1]
    mz2 = np.array(mz_values) ** 2
    ax.fill_between(Gamma_values, 0, 1, where=mz2 > 0.5, alpha=0.3, 
                   color='blue', label='Ferromagnetic')
    ax.fill_between(Gamma_values, 0, 1, where=mz2 <= 0.5, alpha=0.3,
                   color='red', label='Paramagnetic')
    ax.plot(Gamma_values, mz2, 'k-', linewidth=2)
    ax.set_xlabel(r'$\Gamma/J$')
    ax.set_ylabel(r'$\langle M_z \rangle^2$')
    ax.set_title('Phase Diagram')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('basic_simulation_results.png', dpi=150)
    print("\nSaved: basic_simulation_results.png")
    plt.show()
    
    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
