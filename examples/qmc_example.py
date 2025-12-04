#!/usr/bin/env python3
"""
Quantum Monte Carlo Example for 3D TFIM

Demonstrates the use of Stochastic Series Expansion (SSE) QMC
for simulating larger systems at finite temperature.

QMC allows simulation of systems too large for exact diagonalization
by stochastically sampling the partition function:
    Z = Tr[e^{-βH}]
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_ising_3d import QuantumMonteCarlo
from quantum_ising_3d.qmc import WorldLineQMC


def main():
    print("=" * 60)
    print("Quantum Monte Carlo Demonstration")
    print("=" * 60)
    
    # ==========================================
    # 1. Single point simulation
    # ==========================================
    print("\n1. Single point SSE-QMC simulation...")
    
    L = 4
    J = 1.0
    Gamma = 3.0
    beta = 2.0
    
    print(f"\n   Lattice: {L}x{L}x{L} = {L**3} sites")
    print(f"   Parameters: J = {J}, Γ = {Gamma}, β = {beta}")
    
    qmc = QuantumMonteCarlo(L=L, J=J, Gamma=Gamma, beta=beta)
    
    results = qmc.run(
        n_thermalization=500,
        n_measurements=3000,
        n_skip=10
    )
    
    print("\n   Results:")
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"   {key}: {value['mean']:.6f} ± {value['error']:.6f}")
    
    # ==========================================
    # 2. Phase diagram scan
    # ==========================================
    print("\n2. Scanning phase diagram...")
    
    Gamma_values = np.linspace(1.0, 8.0, 12)
    
    qmc = QuantumMonteCarlo(L=L, J=J, Gamma=1.0, beta=beta)
    
    phase_results = qmc.phase_diagram_scan(
        Gamma_values,
        n_thermalization=300,
        n_measurements=2000,
        n_skip=5
    )
    
    # ==========================================
    # 3. World-line QMC comparison
    # ==========================================
    print("\n3. Running World-Line (Path Integral) QMC...")
    
    wl_qmc = WorldLineQMC(L=L, J=J, Gamma=3.0, beta=beta, n_tau=40)
    
    wl_results = wl_qmc.run(
        n_thermalization=100,
        n_measurements=500,
        n_skip=5,
        use_cluster=True
    )
    
    print("\n   World-Line QMC Results:")
    for key, value in wl_results.items():
        if isinstance(value, dict):
            print(f"   {key}: {value['mean']:.6f} ± {value['error']:.6f}")
    
    # ==========================================
    # 4. Plot results
    # ==========================================
    print("\n4. Creating plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Order parameter
    ax = axes[0]
    ax.errorbar(phase_results['Gamma'], 
               phase_results['magnetization_z_squared'],
               yerr=phase_results.get('magnetization_z_error'),
               fmt='o-', capsize=3, color='blue')
    ax.set_xlabel(r'$\Gamma/J$')
    ax.set_ylabel(r'$\langle M_z^2 \rangle$')
    ax.set_title(f'Order Parameter (L={L}, β={beta})')
    ax.grid(True, alpha=0.3)
    
    # Energy
    ax = axes[1]
    ax.errorbar(phase_results['Gamma'],
               phase_results['energy'],
               yerr=phase_results['energy_error'],
               fmt='s-', capsize=3, color='red')
    ax.set_xlabel(r'$\Gamma/J$')
    ax.set_ylabel('Energy')
    ax.set_title('Total Energy')
    ax.grid(True, alpha=0.3)
    
    # Susceptibility
    ax = axes[2]
    if 'susceptibility' in phase_results:
        ax.plot(phase_results['Gamma'], phase_results['susceptibility'],
               'o-', color='green')
        ax.set_xlabel(r'$\Gamma/J$')
        ax.set_ylabel(r'$\chi$')
        ax.set_title('Magnetic Susceptibility')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('qmc_results.png', dpi=150)
    print("\nSaved: qmc_results.png")
    plt.show()
    
    # ==========================================
    # Summary
    # ==========================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    print(f"\nSystem: {L}x{L}x{L} lattice at β = {beta}")
    print(f"SSE-QMC measurements: 2000 per Γ value")
    print(f"\nPhase diagram scanned: Γ ∈ [{Gamma_values[0]}, {Gamma_values[-1]}]")
    
    # Estimate critical point from order parameter
    mz2 = phase_results['magnetization_z_squared']
    transition_idx = np.argmax(np.abs(np.diff(mz2)))
    Gamma_c_estimate = (Gamma_values[transition_idx] + Gamma_values[transition_idx + 1]) / 2
    print(f"Estimated critical point: Γ_c ≈ {Gamma_c_estimate:.2f}")
    
    print("\nNote: For more accurate critical point determination,")
    print("finite-size scaling analysis with multiple L values is needed.")


if __name__ == "__main__":
    main()
