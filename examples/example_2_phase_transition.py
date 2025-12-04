"""
Example 2: Quantum Phase Transition Analysis

This example demonstrates:
- Scanning ground state properties across Γ values
- Identifying critical point Γ_c
- Visualizing order parameters
- Energy gap analysis
"""

import sys
sys.path.append('..')
from quantum_ising_3d import PhaseTransitionAnalyzer, visualize_results
import numpy as np
import matplotlib.pyplot as plt


def main():
    print("\n" + "="*70)
    print("EXAMPLE 2: Quantum Phase Transition")
    print("="*70 + "\n")
    
    # Use small system for demonstration
    # For 3D, critical behavior is different than 1D
    # We'll use a quasi-1D system (chain): 4x1x1
    print("Using 4x1x1 quasi-1D lattice for faster computation")
    print("(True 3D critical point analysis requires larger systems)\n")
    
    Lx, Ly, Lz = 4, 1, 1
    J = 1.0
    
    # Scan transverse field from 0 to 3J
    gamma_values = np.linspace(0.1, 3.0, 15)
    
    analyzer = PhaseTransitionAnalyzer(Lx, Ly, Lz, J=J, periodic=True)
    
    results = analyzer.scan_transverse_field(gamma_values, compute_gap=True)
    
    # Analyze results
    print("\n" + "="*70)
    print("PHASE TRANSITION ANALYSIS")
    print("="*70 + "\n")
    
    # Find approximate critical point from magnetization
    mag_z_abs = np.abs(results['mag_z'])
    
    # Critical point is where magnetization drops most rapidly
    dmag_dgamma = np.gradient(mag_z_abs, results['gamma'])
    critical_idx = np.argmin(dmag_dgamma)
    gamma_critical = results['gamma'][critical_idx]
    
    print(f"Approximate critical point: Γ_c/J ≈ {gamma_critical:.3f}")
    print(f"(Exact result for 1D: Γ_c/J = 1.000)\n")
    
    # Analyze phases
    ferro_idx = 0
    para_idx = -1
    
    print("Ferromagnetic Phase (Γ << J):")
    print(f"  Γ/J = {results['gamma'][ferro_idx]:.3f}")
    print(f"  M_z = {results['mag_z'][ferro_idx]:.4f} (large)")
    print(f"  M_x = {results['mag_x'][ferro_idx]:.4f} (small)")
    print(f"  Gap = {results['gap'][ferro_idx]:.4f}")
    
    print("\nParametric Phase (Γ >> J):")
    print(f"  Γ/J = {results['gamma'][para_idx]:.3f}")
    print(f"  M_z = {results['mag_z'][para_idx]:.4f} (small)")
    print(f"  M_x = {results['mag_x'][para_idx]:.4f} (large)")
    print(f"  Gap = {results['gap'][para_idx]:.4f}")
    
    # Minimum gap location
    min_gap_idx = np.argmin(results['gap'])
    print(f"\nMinimum gap at Γ/J = {results['gamma'][min_gap_idx]:.3f}")
    print(f"  Gap_min = {results['gap'][min_gap_idx]:.4f}")
    print("  (Gap closes at quantum critical point in thermodynamic limit)")
    
    # Create comprehensive visualization
    fig = visualize_results(results, 
                           title=f"Quantum Phase Transition - {Lx}x{Ly}x{Lz} Lattice")
    
    # Additional analysis plot
    fig2, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig2.suptitle('Detailed Phase Transition Analysis', fontsize=14, fontweight='bold')
    
    # Order parameter
    ax = axes[0, 0]
    ax.plot(results['gamma'], np.abs(results['mag_z']), 'b-o', 
            linewidth=2, label='|M_z|')
    ax.plot(results['gamma'], np.abs(results['mag_x']), 'r-s', 
            linewidth=2, label='|M_x|')
    ax.axvline(x=1.0, color='k', linestyle='--', alpha=0.5, label='Γ_c = J')
    ax.axvline(x=gamma_critical, color='g', linestyle=':', alpha=0.5, 
               label=f'Γ_c ≈ {gamma_critical:.2f}J')
    ax.set_xlabel('Γ/J', fontsize=12)
    ax.set_ylabel('Magnetization', fontsize=12)
    ax.set_title('Order Parameters')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Energy derivative (specific heat analog)
    ax = axes[0, 1]
    dE_dgamma = np.gradient(results['energy'], results['gamma'])
    ax.plot(results['gamma'], -dE_dgamma, 'g-o', linewidth=2)
    ax.axvline(x=1.0, color='k', linestyle='--', alpha=0.5)
    ax.axvline(x=gamma_critical, color='g', linestyle=':', alpha=0.5)
    ax.set_xlabel('Γ/J', fontsize=12)
    ax.set_ylabel('-dE/dΓ', fontsize=12)
    ax.set_title('Energy Susceptibility')
    ax.grid(True, alpha=0.3)
    
    # Magnetization derivative (susceptibility)
    ax = axes[1, 0]
    dmz_dgamma = np.abs(np.gradient(results['mag_z'], results['gamma']))
    ax.plot(results['gamma'], dmz_dgamma, 'm-o', linewidth=2)
    ax.axvline(x=1.0, color='k', linestyle='--', alpha=0.5)
    ax.axvline(x=gamma_critical, color='g', linestyle=':', alpha=0.5)
    ax.set_xlabel('Γ/J', fontsize=12)
    ax.set_ylabel('|dM_z/dΓ|', fontsize=12)
    ax.set_title('Magnetic Susceptibility')
    ax.grid(True, alpha=0.3)
    
    # Gap analysis
    ax = axes[1, 1]
    ax.semilogy(results['gamma'], results['gap'], 'c-o', linewidth=2)
    ax.axvline(x=1.0, color='k', linestyle='--', alpha=0.5, label='Γ_c = J (exact)')
    ax.axvline(x=gamma_critical, color='g', linestyle=':', alpha=0.5, 
               label=f'Γ_c ≈ {gamma_critical:.2f}J (numerical)')
    ax.set_xlabel('Γ/J', fontsize=12)
    ax.set_ylabel('Energy Gap ΔE', fontsize=12)
    ax.set_title('Excitation Gap (log scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figures
    fig.savefig('phase_transition_overview.png', dpi=150, bbox_inches='tight')
    fig2.savefig('phase_transition_analysis.png', dpi=150, bbox_inches='tight')
    
    print("\n" + "="*70)
    print("Figures saved:")
    print("  - phase_transition_overview.png")
    print("  - phase_transition_analysis.png")
    print("="*70 + "\n")
    
    plt.show()


if __name__ == "__main__":
    main()
