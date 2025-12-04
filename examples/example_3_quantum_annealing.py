"""
Example 3: Quantum Annealing

This example demonstrates:
- Quantum annealing protocol
- Time evolution from easy to hard problem
- Different annealing schedules
- Convergence to ground state
"""

import sys
sys.path.append('..')
from quantum_ising_3d import QuantumIsing3D, QuantumAnnealing, visualize_results
import numpy as np
import matplotlib.pyplot as plt


def main():
    print("\n" + "="*70)
    print("EXAMPLE 3: Quantum Annealing")
    print("="*70 + "\n")
    
    print("Quantum Annealing Protocol:")
    print("  1. Start with large Γ (easy quantum ground state)")
    print("  2. Slowly decrease Γ → 0 (hard classical problem)")
    print("  3. System remains in ground state (adiabatic theorem)")
    print("  4. Final state solves optimization problem\n")
    
    # Use small system
    Lx, Ly, Lz = 3, 1, 1  # 3 spins
    J = 1.0
    
    model = QuantumIsing3D(Lx, Ly, Lz, J=J, periodic=False)
    annealer = QuantumAnnealing(model)
    
    # First, find the target ground state (Γ = 0)
    print("Finding target ground state (Γ → 0)...")
    model_target = QuantumIsing3D(Lx, Ly, Lz, J=J, Gamma=0.01, periodic=False)
    target_eigenvalues, target_eigenvectors = model_target.find_ground_state(k=1)
    target_energy = target_eigenvalues[0]
    target_state = target_eigenvectors[:, 0]
    
    print(f"Target ground state energy: {target_energy:.6f}")
    print(f"Target M_z: {model_target.magnetization_z(target_state):.6f}\n")
    
    # Perform annealing with different schedules
    print("="*70)
    print("QUANTUM ANNEALING SIMULATIONS")
    print("="*70)
    
    schedules = ['linear', 'polynomial']
    colors = ['blue', 'red', 'green']
    results_all = {}
    
    for i, schedule in enumerate(schedules):
        print(f"\n{'-'*70}")
        print(f"Schedule {i+1}: {schedule.upper()}")
        print(f"{'-'*70}")
        
        # Annealing parameters
        T = 10.0  # Total time
        num_steps = 50  # Number of time steps
        Gamma_init = 5.0  # Start with large Γ
        Gamma_final = 0.1  # End with small Γ
        
        results = annealer.anneal(
            T=T,
            num_steps=num_steps,
            Gamma_init=Gamma_init,
            Gamma_final=Gamma_final,
            schedule=schedule
        )
        
        results_all[schedule] = results
        
        # Analyze final state
        final_state = results['states'][-1]
        final_energy = results['energies'][-1]
        
        # Overlap with target ground state
        overlap = np.abs(np.vdot(target_state, final_state))**2
        
        print(f"\nFinal Results:")
        print(f"  Final energy: {final_energy:.6f}")
        print(f"  Target energy: {target_energy:.6f}")
        print(f"  Energy error: {abs(final_energy - target_energy):.6f}")
        print(f"  Ground state overlap: {overlap:.6f}")
        print(f"  Success: {'YES' if overlap > 0.95 else 'NO'}")
    
    # Visualization
    print("\n" + "="*70)
    print("VISUALIZATION")
    print("="*70 + "\n")
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('Quantum Annealing: Different Schedules', fontsize=14, fontweight='bold')
    
    for idx, (schedule, results) in enumerate(results_all.items()):
        col = idx
        
        # Transverse field schedule
        ax = axes[0, col]
        ax.plot(results['times'], results['gammas'], 
                color=colors[idx], linewidth=2)
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Γ(t)', fontsize=11)
        ax.set_title(f'{schedule.capitalize()} Schedule', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Energy evolution
        ax = axes[1, col]
        ax.plot(results['times'], results['energies'], 
                color=colors[idx], linewidth=2, label='E(t)')
        ax.axhline(y=target_energy, color='k', linestyle='--', 
                   alpha=0.5, label='Target E₀')
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Energy', fontsize=11)
        ax.set_title('Energy Evolution', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Magnetizations
        ax = axes[2, col]
        ax.plot(results['times'], results['mag_z'], 
                'b-', linewidth=2, label='M_z')
        ax.plot(results['times'], results['mag_x'], 
                'r-', linewidth=2, label='M_x')
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Magnetization', fontsize=11)
        ax.set_title('Order Parameters', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig('quantum_annealing.png', dpi=150, bbox_inches='tight')
    
    print("Figure saved: quantum_annealing.png\n")
    
    # Compare convergence
    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig2.suptitle('Annealing Schedule Comparison', fontsize=14, fontweight='bold')
    
    ax = axes[0]
    for schedule, results in results_all.items():
        energy_error = np.abs(results['energies'] - target_energy)
        ax.semilogy(results['times'], energy_error, 
                   linewidth=2, label=schedule.capitalize())
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('|E(t) - E₀|', fontsize=12)
    ax.set_title('Energy Convergence (log scale)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    for schedule, results in results_all.items():
        ax.plot(results['gammas'], results['energies'], 
               linewidth=2, label=schedule.capitalize())
    ax.axhline(y=target_energy, color='k', linestyle='--', 
               alpha=0.5, label='Target E₀')
    ax.set_xlabel('Γ', fontsize=12)
    ax.set_ylabel('Energy', fontsize=12)
    ax.set_title('Energy vs Transverse Field', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig2.savefig('annealing_comparison.png', dpi=150, bbox_inches='tight')
    
    print("Figure saved: annealing_comparison.png\n")
    
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print("\nQuantum annealing successfully finds ground state by:")
    print("  1. Starting in easy-to-prepare quantum state (large Γ)")
    print("  2. Adiabatically evolving to classical ground state (small Γ)")
    print("  3. Maintaining ground state occupation throughout")
    print("\nDifferent schedules affect convergence speed:")
    print("  - Linear: constant rate")
    print("  - Polynomial: slower near transition")
    print("\nFor optimization: slower annealing near Γ_c improves success.")
    print("="*70 + "\n")
    
    plt.show()


if __name__ == "__main__":
    main()
