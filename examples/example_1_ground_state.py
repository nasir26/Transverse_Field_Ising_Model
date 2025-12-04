"""
Example 1: Ground State Properties of 3D Ising Model

This example demonstrates:
- Finding ground state for small 3D lattice
- Computing observables (energy, magnetization)
- Computing spin-spin correlations
"""

import sys
sys.path.append('..')
from quantum_ising_3d import QuantumIsing3D, visualize_results
import numpy as np
import matplotlib.pyplot as plt


def main():
    print("\n" + "="*70)
    print("EXAMPLE 1: Ground State Properties")
    print("="*70 + "\n")
    
    # Create 2x2x2 lattice (8 spins, 256-dimensional Hilbert space)
    Lx, Ly, Lz = 2, 2, 2
    J = 1.0
    Gamma = 0.5  # Γ < J: ferromagnetic phase expected
    
    print(f"System: {Lx}x{Ly}x{Lz} lattice with {Lx*Ly*Lz} spins")
    print(f"Parameters: J = {J}, Γ = {Gamma}")
    print(f"Γ/J = {Gamma/J:.2f} < 1 → Ferromagnetic phase\n")
    
    # Initialize model
    model = QuantumIsing3D(Lx, Ly, Lz, J=J, Gamma=Gamma, periodic=False)
    
    # Find ground state and first excited state
    eigenvalues, eigenvectors = model.find_ground_state(k=2)
    
    ground_state = eigenvectors[:, 0]
    E0 = eigenvalues[0]
    E1 = eigenvalues[1]
    gap = E1 - E0
    
    print(f"\nGround state energy: E_0 = {E0:.6f}")
    print(f"First excited state: E_1 = {E1:.6f}")
    print(f"Energy gap: ΔE = {gap:.6f}")
    
    # Compute observables
    print("\n" + "-"*70)
    print("Ground State Observables:")
    print("-"*70)
    
    mag_z = model.magnetization_z(ground_state)
    mag_x = model.magnetization_x(ground_state)
    energy = model.energy(ground_state)
    
    print(f"M_z (z-magnetization): {mag_z:.6f}")
    print(f"M_x (x-magnetization): {mag_x:.6f}")
    print(f"Energy per spin: {energy / model.N:.6f}")
    print(f"|M_z| = {abs(mag_z):.6f} (close to 1 → ordered)")
    print(f"|M_x| = {abs(mag_x):.6f} (close to 0 → no x-polarization)")
    
    # Compute correlation functions
    print("\n" + "-"*70)
    print("Spin-Spin Correlations <σ_i^z σ_j^z>:")
    print("-"*70)
    
    # Correlation between neighboring spins
    site1, site2 = 0, 1  # Neighboring sites
    corr = model.correlation_zz(site1, site2, ground_state)
    print(f"<σ_0^z σ_1^z> (neighbors): {corr:.6f}")
    
    # Correlation between distant spins
    site1, site2 = 0, 7  # Opposite corners
    corr_distant = model.correlation_zz(site1, site2, ground_state)
    print(f"<σ_0^z σ_7^z> (corners):   {corr_distant:.6f}")
    
    print("\nPositive correlations indicate ferromagnetic ordering.")
    
    # Analyze ground state structure
    print("\n" + "-"*70)
    print("Ground State Wavefunction Analysis:")
    print("-"*70)
    
    # Find basis states with largest amplitudes
    amplitudes = np.abs(ground_state)**2
    sorted_indices = np.argsort(amplitudes)[::-1]
    
    print("Top 5 basis states (by probability):")
    for i in range(min(5, len(sorted_indices))):
        idx = sorted_indices[i]
        prob = amplitudes[idx]
        # Convert index to binary string (spin configuration)
        spin_config = format(idx, f'0{model.N}b')
        spin_config = spin_config.replace('0', '↑').replace('1', '↓')
        print(f"  |{spin_config}>: probability = {prob:.4f}")
    
    print(f"\nTotal probability in top 5 states: {np.sum(amplitudes[sorted_indices[:5]]):.4f}")
    
    # Compare with high Γ regime
    print("\n" + "="*70)
    print("Comparison with Paramagnetic Phase (Γ >> J)")
    print("="*70 + "\n")
    
    Gamma_high = 5.0
    model_param = QuantumIsing3D(Lx, Ly, Lz, J=J, Gamma=Gamma_high, periodic=False)
    eigenvalues_param, eigenvectors_param = model_param.find_ground_state(k=2)
    
    ground_state_param = eigenvectors_param[:, 0]
    mag_z_param = model_param.magnetization_z(ground_state_param)
    mag_x_param = model_param.magnetization_x(ground_state_param)
    
    print(f"Γ = {Gamma_high} (Γ/J = {Gamma_high/J:.2f} >> 1)")
    print(f"M_z: {mag_z_param:.6f} (close to 0 → disordered)")
    print(f"M_x: {mag_x_param:.6f} (close to ±1 → x-polarized)")
    
    print("\n" + "="*70)
    print("Summary:")
    print("="*70)
    print(f"Low Γ (ferro):  M_z = {mag_z:.4f}, M_x = {mag_x:.4f}")
    print(f"High Γ (para):  M_z = {mag_z_param:.4f}, M_x = {mag_x_param:.4f}")
    print("\nThis demonstrates the quantum phase transition!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
