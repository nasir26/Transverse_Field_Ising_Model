#!/usr/bin/env python3
"""
Quantum Annealing Example for 3D TFIM

Demonstrates quantum annealing for finding the ground state of the
classical Ising problem by slowly reducing the transverse field.

The time-dependent Hamiltonian is:
    H(t) = -J Σ σ_i^z σ_j^z - Γ(t) Σ σ_i^x

Starting from Γ(0) >> J (easy ground state: all spins along x)
and slowly decreasing Γ(t) → 0, the system adiabatically evolves
to the ground state of the classical Ising model.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_ising_3d import TransverseFieldIsing3D, QuantumAnnealing


def main():
    print("=" * 60)
    print("Quantum Annealing Demonstration")
    print("=" * 60)
    
    # Create model (L=2 for exact simulation)
    L = 2
    J = 1.0
    
    model = TransverseFieldIsing3D(L=L, J=J, Gamma=0.0)
    print(f"\nLattice: {L}x{L}x{L} = {model.N} sites")
    
    # Create annealing simulator
    qa = QuantumAnnealing(model)
    
    # ==========================================
    # 1. Study energy gap
    # ==========================================
    print("\n1. Computing energy gap spectrum...")
    
    Gamma_values = np.linspace(0.1, 15.0, 50)
    gaps, min_gap, Gamma_c = qa.compute_minimum_gap(Gamma_values)
    
    print(f"\n   Minimum gap: {min_gap:.6f}")
    print(f"   Critical transverse field: Γ_c = {Gamma_c:.4f}")
    print(f"   Adiabatic theorem suggests: t_anneal >> {1/min_gap**2:.1f}")
    
    # ==========================================
    # 2. Run annealing with different times
    # ==========================================
    print("\n2. Running quantum annealing...")
    
    t_anneal_values = [1.0, 5.0, 10.0, 20.0, 50.0]
    results_list = []
    
    for t_anneal in t_anneal_values:
        print(f"\n   t_anneal = {t_anneal}...")
        result = qa.run_exact_annealing(
            t_anneal=t_anneal,
            n_steps=int(20 * t_anneal),
            Gamma_0=15.0,
            schedule='linear'
        )
        results_list.append(result)
        print(f"   Success probability: {result['success_probability']:.4f}")
    
    # ==========================================
    # 3. Compare annealing schedules
    # ==========================================
    print("\n3. Comparing annealing schedules...")
    
    t_anneal = 15.0
    schedules = ['linear', 'exponential', 'polynomial']
    schedule_results = {}
    
    for schedule in schedules:
        print(f"\n   Schedule: {schedule}...")
        result = qa.run_exact_annealing(
            t_anneal=t_anneal,
            n_steps=300,
            Gamma_0=15.0,
            schedule=schedule
        )
        schedule_results[schedule] = result
        print(f"   Success probability: {result['success_probability']:.4f}")
    
    # ==========================================
    # 4. Plot results
    # ==========================================
    print("\n4. Creating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Gap spectrum
    ax = axes[0, 0]
    ax.plot(Gamma_values, gaps, 'b-', linewidth=2)
    ax.axvline(Gamma_c, color='red', linestyle='--', 
              label=f'$\\Gamma_c = {Gamma_c:.2f}$')
    ax.set_xlabel(r'$\Gamma/J$')
    ax.set_ylabel(r'Gap $\Delta$')
    ax.set_title('Energy Gap Spectrum')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Success probability vs annealing time
    ax = axes[0, 1]
    success_probs = [r['success_probability'] for r in results_list]
    ax.semilogx(t_anneal_values, success_probs, 'go-', markersize=8, linewidth=2)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel(r'Annealing time $t_{\rm anneal}$')
    ax.set_ylabel('Success Probability')
    ax.set_title('Adiabatic Time Scaling')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    
    # Energy evolution for different times
    ax = axes[1, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(t_anneal_values)))
    for result, color, t in zip(results_list, colors, t_anneal_values):
        ax.plot(result['times'] / result['times'][-1], result['energy'],
               color=color, label=f't={t}')
    ax.axhline(results_list[0]['classical_gs_energy'], color='red', 
              linestyle='--', label='Classical GS')
    ax.set_xlabel(r'$t / t_{\rm anneal}$')
    ax.set_ylabel('Energy')
    ax.set_title('Energy Evolution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Schedule comparison
    ax = axes[1, 1]
    for schedule, result in schedule_results.items():
        ax.plot(result['times'], result['overlap_gs'], 
               label=f'{schedule}: P={result["success_probability"]:.3f}',
               linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Ground State Overlap')
    ax.set_title(f'Schedule Comparison (t={t_anneal})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quantum_annealing_results.png', dpi=150)
    print("\nSaved: quantum_annealing_results.png")
    plt.show()
    
    # ==========================================
    # Summary
    # ==========================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"\nClassical ground state energy: {qa.classical_gs_energy:.4f}")
    print(f"Minimum gap: {min_gap:.6f}")
    print(f"Critical field: Γ_c = {Gamma_c:.4f}")
    print("\nSuccess probabilities by annealing time:")
    for t, p in zip(t_anneal_values, success_probs):
        print(f"  t = {t:5.1f}: P = {p:.4f}")
    print("\nBest schedule:", max(schedule_results.items(), 
                                 key=lambda x: x[1]['success_probability'])[0])


if __name__ == "__main__":
    main()
