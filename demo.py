#!/usr/bin/env python3
"""
Quick Demonstration of Quantum Ising 3D Simulator

Run this to see a quick overview of capabilities.
"""

from quantum_ising_3d import QuantumIsing3D, QuantumAnnealing, PhaseTransitionAnalyzer
import numpy as np
import matplotlib.pyplot as plt


def demo_banner():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║     QUANTUM ISING 3D SIMULATOR - DEMONSTRATION                ║
    ║                                                                ║
    ║     H = -J Σ_<i,j> σ_i^z σ_j^z - Γ Σ_i σ_i^x                 ║
    ║                                                                ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)


def demo_1_basic():
    """Demo 1: Basic ground state calculation"""
    print("\n" + "="*70)
    print("DEMO 1: Ground State of Small System")
    print("="*70 + "\n")
    
    print("Creating a 2x2x1 lattice (4 spins)...")
    model = QuantumIsing3D(2, 2, 1, J=1.0, Gamma=0.3)
    
    print("Finding ground state...")
    eigenvalues, eigenvectors = model.find_ground_state(k=2)
    
    ground_state = eigenvectors[:, 0]
    
    print(f"\n✓ Results:")
    print(f"  Ground state energy: E_0 = {eigenvalues[0]:.6f}")
    print(f"  Energy gap: ΔE = {eigenvalues[1] - eigenvalues[0]:.6f}")
    print(f"  Z-magnetization: M_z = {model.magnetization_z(ground_state):.6f}")
    print(f"  X-magnetization: M_x = {model.magnetization_x(ground_state):.6f}")
    
    print("\nInterpretation:")
    mag_z = abs(model.magnetization_z(ground_state))
    if mag_z > 0.8:
        print("  → Strong ferromagnetic order (spins aligned)")
    elif mag_z < 0.2:
        print("  → Paramagnetic phase (quantum fluctuations dominate)")
    else:
        print("  → Near quantum critical point")


def demo_2_phase_transition():
    """Demo 2: Quantum phase transition"""
    print("\n" + "="*70)
    print("DEMO 2: Quantum Phase Transition Scan")
    print("="*70 + "\n")
    
    print("Scanning transverse field from Γ/J = 0.1 to 3.0...")
    print("(Using 4x1x1 lattice for speed)")
    
    analyzer = PhaseTransitionAnalyzer(4, 1, 1, J=1.0)
    gamma_values = np.linspace(0.1, 3.0, 10)
    
    results = analyzer.scan_transverse_field(gamma_values, compute_gap=False)
    
    # Find transition
    mag_z_abs = np.abs(results['mag_z'])
    dmag = np.gradient(mag_z_abs)
    transition_idx = np.argmin(dmag)
    
    print(f"\n✓ Analysis complete!")
    print(f"  Approximate Γ_c/J ≈ {results['gamma'][transition_idx]:.2f}")
    print(f"  (Exact 1D value: Γ_c/J = 1.00)")
    
    # Simple plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    ax = axes[0]
    ax.plot(results['gamma'], results['mag_z'], 'bo-', linewidth=2, markersize=6)
    ax.axvline(x=1.0, color='r', linestyle='--', label='Γ_c = J (exact)')
    ax.set_xlabel('Γ/J', fontsize=12)
    ax.set_ylabel('M_z', fontsize=12)
    ax.set_title('Order Parameter', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.plot(results['gamma'], results['energy'] / 4, 'ro-', linewidth=2, markersize=6)
    ax.set_xlabel('Γ/J', fontsize=12)
    ax.set_ylabel('E₀/N', fontsize=12)
    ax.set_title('Ground State Energy per Spin', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig('demo_phase_transition.png', dpi=120)
    print("\n  Saved: demo_phase_transition.png")


def demo_3_annealing():
    """Demo 3: Quantum annealing"""
    print("\n" + "="*70)
    print("DEMO 3: Quantum Annealing")
    print("="*70 + "\n")
    
    print("Setting up annealing protocol...")
    print("  Initial: Γ = 5.0 (easy quantum state)")
    print("  Final: Γ = 0.1 (classical ground state)")
    print("  Using 3x1x1 lattice (3 spins)\n")
    
    model = QuantumIsing3D(3, 1, 1, J=1.0)
    annealer = QuantumAnnealing(model)
    
    results = annealer.anneal(
        T=8.0,
        num_steps=30,
        Gamma_init=5.0,
        Gamma_final=0.1,
        schedule='linear'
    )
    
    print(f"\n✓ Annealing complete!")
    print(f"  Initial energy: {results['energies'][0]:.6f}")
    print(f"  Final energy: {results['energies'][-1]:.6f}")
    print(f"  Final M_z: {results['mag_z'][-1]:.6f}")
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    ax = axes[0]
    ax.plot(results['times'], results['energies'], 'b-', linewidth=2)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Energy', fontsize=12)
    ax.set_title('Energy Evolution During Annealing', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.plot(results['times'], results['gammas'], 'm-', linewidth=2, label='Γ(t)')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Γ', fontsize=12)
    ax.set_title('Transverse Field Schedule', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig('demo_annealing.png', dpi=120)
    print("\n  Saved: demo_annealing.png")


def demo_4_3d():
    """Demo 4: True 3D system"""
    print("\n" + "="*70)
    print("DEMO 4: True 3D System")
    print("="*70 + "\n")
    
    print("Creating 2x2x2 cubic lattice...")
    model = QuantumIsing3D(2, 2, 2, J=1.0, Gamma=0.5)
    
    print(f"  Total spins: {model.N}")
    print(f"  Hilbert space dimension: 2^{model.N} = {2**model.N}")
    
    # Count bonds
    bonds = 0
    for i in range(model.N):
        neighbors = model._get_neighbors(i)
        bonds += len([j for j in neighbors if j > i])
    
    print(f"  Total bonds: {bonds}")
    
    print("\nFinding ground state...")
    eigenvalues, eigenvectors = model.find_ground_state(k=1)
    ground_state = eigenvectors[:, 0]
    
    print(f"\n✓ Results:")
    print(f"  E_0 = {eigenvalues[0]:.6f}")
    print(f"  M_z = {model.magnetization_z(ground_state):.6f}")
    
    # Compute all correlations
    print("\n  Computing correlation matrix...")
    corr_matrix = np.zeros((model.N, model.N))
    for i in range(model.N):
        for j in range(model.N):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                corr_matrix[i, j] = model.correlation_zz(i, j, ground_state)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1)
    ax.set_xlabel('Spin j', fontsize=12)
    ax.set_ylabel('Spin i', fontsize=12)
    ax.set_title('Spin Correlations <σ_i^z σ_j^z> in 2×2×2 Cube', 
                 fontsize=13, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', fontsize=12)
    
    plt.tight_layout()
    fig.savefig('demo_3d_correlations.png', dpi=120)
    print("\n  Saved: demo_3d_correlations.png")


def main():
    """Run all demos"""
    demo_banner()
    
    demos = [
        ("Basic Ground State", demo_1_basic),
        ("Phase Transition", demo_2_phase_transition),
        ("Quantum Annealing", demo_3_annealing),
        ("3D System", demo_4_3d),
    ]
    
    print("\nThis demo will run 4 quick simulations showcasing key features.")
    print("Figures will be saved to the current directory.\n")
    
    input("Press Enter to continue...")
    
    for i, (name, demo_func) in enumerate(demos, 1):
        try:
            demo_func()
        except Exception as e:
            print(f"\n✗ Demo {i} failed: {e}")
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - demo_phase_transition.png")
    print("  - demo_annealing.png")
    print("  - demo_3d_correlations.png")
    print("\nFor more detailed examples, see the examples/ directory:")
    print("  - example_1_ground_state.py")
    print("  - example_2_phase_transition.py")
    print("  - example_3_quantum_annealing.py")
    print("  - example_4_3d_system.py")
    print("\nFor full documentation, see README.md")
    print("="*70 + "\n")
    
    try:
        plt.show()
    except:
        pass


if __name__ == "__main__":
    main()
