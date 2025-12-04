#!/usr/bin/env python3
"""
Main Simulation Runner for 3D Transverse Field Ising Model

This script provides a comprehensive simulation of the quantum 3D Transverse Field
Ising Model (TFIM) with the following capabilities:

1. Exact Diagonalization: For small systems (L ≤ 2)
2. Quantum Monte Carlo: For larger systems using SSE algorithm
3. Quantum Annealing: Time-dependent simulation for optimization
4. Phase Diagram: Mapping the quantum phase transition
5. Visualization: Publication-quality plots

The Hamiltonian is:
    H = -J Σ_{<i,j>} σ_i^z σ_j^z - Γ Σ_i σ_i^x

where the sum is over nearest neighbors on a 3D cubic lattice.

Usage:
    python run_simulation.py [mode] [options]
    
Modes:
    exact    - Run exact diagonalization (default)
    qmc      - Run Quantum Monte Carlo
    anneal   - Run quantum annealing
    phase    - Compute phase diagram
    all      - Run all simulations

Examples:
    python run_simulation.py exact --L 2 --Gamma 1.0
    python run_simulation.py qmc --L 4 --beta 2.0
    python run_simulation.py phase --L 2 --n_points 30
    python run_simulation.py all
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantum_ising_3d import (
    TransverseFieldIsing3D,
    ExactDiagonalization,
    QuantumMonteCarlo,
    QuantumAnnealing,
    Observables,
    Visualization
)
from quantum_ising_3d.hamiltonian import TransverseFieldIsing1D
from quantum_ising_3d.annealing import SimulatedQuantumAnnealing
from quantum_ising_3d.utils import save_results, check_feasibility, timer


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


@timer
def run_exact_diagonalization(L: int = 2, J: float = 1.0, Gamma: float = 1.0,
                               show_plots: bool = True) -> dict:
    """
    Run exact diagonalization for small 3D TFIM.
    
    Parameters
    ----------
    L : int
        Linear lattice size (L x L x L)
    J : float
        Ising coupling
    Gamma : float
        Transverse field
    show_plots : bool
        Whether to display plots
    """
    print_header("Exact Diagonalization")
    
    # Check feasibility
    check_feasibility(L, method='exact')
    
    # Create model
    model = TransverseFieldIsing3D(L=L, J=J, Gamma=Gamma)
    model.analyze_system()
    
    # Create solver
    solver = ExactDiagonalization(model)
    
    # Analyze ground state
    results = solver.analyze_ground_state()
    
    print("\n" + "-" * 50)
    print("Ground State Properties")
    print("-" * 50)
    for key, value in results.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.6f}")
    
    return results


@timer
def run_phase_diagram(L: int = 2, J: float = 1.0, n_points: int = 30,
                       Gamma_min: float = 0.1, Gamma_max: float = 8.0,
                       show_plots: bool = True, save_plots: bool = True) -> dict:
    """
    Compute phase diagram by scanning Γ values.
    
    Parameters
    ----------
    L : int
        Linear lattice size
    J : float
        Ising coupling
    n_points : int
        Number of Γ values to scan
    Gamma_min, Gamma_max : float
        Range of Γ values
    """
    print_header("Phase Diagram Computation")
    
    check_feasibility(L, method='exact')
    
    model = TransverseFieldIsing3D(L=L, J=J, Gamma=1.0)
    solver = ExactDiagonalization(model)
    
    Gamma_values = np.linspace(Gamma_min, Gamma_max, n_points)
    
    results = solver.phase_diagram_scan(
        Gamma_values,
        observables=['energy_per_site', 'magnetization_z', 'magnetization_x', 
                    'magnetization_z_squared', 'gap']
    )
    
    # Find critical point estimate
    mz2 = results.get('magnetization_z_squared', results['magnetization_z']**2)
    gap_min_idx = np.argmin(results['gap'])
    Gamma_c = Gamma_values[gap_min_idx]
    
    print("\n" + "-" * 50)
    print("Phase Diagram Summary")
    print("-" * 50)
    print(f"Γ range: [{Gamma_min}, {Gamma_max}]")
    print(f"Number of points: {n_points}")
    print(f"Estimated critical point Γ_c/J ≈ {Gamma_c/J:.3f}")
    print(f"Minimum gap: {results['gap'][gap_min_idx]:.6f}")
    
    # Plot
    if show_plots or save_plots:
        viz = Visualization(save_dir='./figures')
        
        # Add susceptibility calculation
        chi = L**3 * (mz2 - results['magnetization_z']**2)
        results['susceptibility'] = chi
        
        fig = viz.plot_phase_diagram(results, 
                                     title=f"3D TFIM Phase Diagram (L={L})")
        
        if save_plots:
            os.makedirs('./figures', exist_ok=True)
            fig.savefig(f'./figures/phase_diagram_L{L}.png', dpi=150, bbox_inches='tight')
            print(f"\nSaved: ./figures/phase_diagram_L{L}.png")
        
        if show_plots:
            plt.show()
    
    return results


@timer
def run_quantum_monte_carlo(L: int = 4, J: float = 1.0, Gamma: float = 3.0,
                             beta: float = 2.0, n_measurements: int = 2000,
                             show_plots: bool = True) -> dict:
    """
    Run Quantum Monte Carlo simulation using SSE.
    
    Parameters
    ----------
    L : int
        Linear lattice size
    J : float
        Ising coupling
    Gamma : float
        Transverse field
    beta : float
        Inverse temperature
    n_measurements : int
        Number of measurements
    """
    print_header("Quantum Monte Carlo (SSE)")
    
    print(f"System: {L}x{L}x{L} = {L**3} sites")
    print(f"Parameters: J = {J}, Γ = {Gamma}, β = {beta}")
    print(f"Measurements: {n_measurements}")
    
    qmc = QuantumMonteCarlo(L=L, J=J, Gamma=Gamma, beta=beta)
    
    results = qmc.run(
        n_thermalization=500,
        n_measurements=n_measurements,
        n_skip=10
    )
    
    print("\n" + "-" * 50)
    print("QMC Results")
    print("-" * 50)
    for key, value in results.items():
        if isinstance(value, dict) and 'mean' in value:
            print(f"  {key}: {value['mean']:.6f} ± {value['error']:.6f}")
    
    return results


@timer
def run_qmc_phase_diagram(L: int = 4, J: float = 1.0, beta: float = 2.0,
                           n_points: int = 15, n_measurements: int = 1000,
                           show_plots: bool = True, save_plots: bool = True) -> dict:
    """
    Compute phase diagram using QMC for larger systems.
    """
    print_header("QMC Phase Diagram")
    
    Gamma_values = np.linspace(0.5, 8.0, n_points)
    
    qmc = QuantumMonteCarlo(L=L, J=J, Gamma=1.0, beta=beta)
    
    results = qmc.phase_diagram_scan(
        Gamma_values,
        n_thermalization=300,
        n_measurements=n_measurements,
        n_skip=5
    )
    
    if show_plots or save_plots:
        viz = Visualization(save_dir='./figures')
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.errorbar(results['Gamma'], results['magnetization_z_squared'],
                   yerr=results.get('magnetization_z_error', None),
                   fmt='o-', capsize=3, label=r'$\langle M_z^2 \rangle$')
        ax.set_xlabel(r'$\Gamma/J$', fontsize=12)
        ax.set_ylabel(r'$\langle M_z^2 \rangle$', fontsize=12)
        ax.set_title(f'3D TFIM Order Parameter (L={L}, QMC)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        if save_plots:
            os.makedirs('./figures', exist_ok=True)
            fig.savefig(f'./figures/qmc_phase_diagram_L{L}.png', dpi=150, bbox_inches='tight')
            print(f"\nSaved: ./figures/qmc_phase_diagram_L{L}.png")
        
        if show_plots:
            plt.show()
    
    return results


@timer  
def run_quantum_annealing(L: int = 2, J: float = 1.0, t_anneal: float = 10.0,
                           n_steps: int = 200, schedule: str = 'linear',
                           show_plots: bool = True, save_plots: bool = True) -> dict:
    """
    Run quantum annealing simulation.
    
    Parameters
    ----------
    L : int
        Linear lattice size
    J : float
        Ising coupling
    t_anneal : float
        Total annealing time
    n_steps : int
        Number of time steps
    schedule : str
        'linear', 'exponential', or 'polynomial'
    """
    print_header("Quantum Annealing")
    
    check_feasibility(L, method='exact')
    
    model = TransverseFieldIsing3D(L=L, J=J, Gamma=0.0)
    
    qa = QuantumAnnealing(model)
    
    # Compute minimum gap
    print("Computing energy gap spectrum...")
    Gamma_scan = np.linspace(0.1, 15.0, 50)
    gaps, min_gap, Gamma_c = qa.compute_minimum_gap(Gamma_scan)
    
    print(f"\nMinimum gap: {min_gap:.6f} at Γ_c = {Gamma_c:.4f}")
    
    # Run annealing
    results = qa.run_exact_annealing(
        t_anneal=t_anneal,
        n_steps=n_steps,
        Gamma_0=15.0,
        schedule=schedule
    )
    
    results['gaps'] = gaps
    results['Gamma_scan'] = Gamma_scan
    results['min_gap'] = min_gap
    results['Gamma_critical'] = Gamma_c
    
    if show_plots or save_plots:
        viz = Visualization(save_dir='./figures')
        fig = viz.plot_annealing_dynamics(results, 
                                          title=f"Quantum Annealing (L={L}, t={t_anneal})")
        
        if save_plots:
            os.makedirs('./figures', exist_ok=True)
            fig.savefig(f'./figures/annealing_L{L}_t{t_anneal}.png', 
                       dpi=150, bbox_inches='tight')
            print(f"\nSaved: ./figures/annealing_L{L}_t{t_anneal}.png")
        
        # Also plot gap spectrum
        fig2 = viz.plot_gap_vs_Gamma(Gamma_scan, gaps, Gamma_c=Gamma_c,
                                     title=f"Energy Gap (L={L})")
        
        if save_plots:
            fig2.savefig(f'./figures/gap_spectrum_L{L}.png', 
                        dpi=150, bbox_inches='tight')
            print(f"Saved: ./figures/gap_spectrum_L{L}.png")
        
        if show_plots:
            plt.show()
    
    return results


@timer
def run_simulated_quantum_annealing(L: int = 4, J: float = 1.0,
                                     n_steps: int = 500,
                                     show_plots: bool = True) -> dict:
    """
    Run simulated quantum annealing using Path Integral Monte Carlo.
    
    This is suitable for larger systems where exact simulation is infeasible.
    """
    print_header("Simulated Quantum Annealing")
    
    sqa = SimulatedQuantumAnnealing(L=L, J=J, n_replicas=20)
    
    results = sqa.run_annealing(
        beta=10.0,
        Gamma_0=10.0,
        n_steps=n_steps,
        mc_sweeps_per_step=10
    )
    
    print("\n" + "-" * 50)
    print("SQA Results")
    print("-" * 50)
    print(f"Final energy: {results['final_energy']:.4f}")
    print(f"Final magnetization: {results['magnetization']:.4f}")
    
    if show_plots:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(results['Gamma'], results['energy'], '-')
        ax.set_xlabel(r'$\Gamma$', fontsize=12)
        ax.set_ylabel('Energy', fontsize=12)
        ax.set_title(f'Simulated Quantum Annealing (L={L})', fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.show()
    
    return results


@timer
def run_1d_comparison(L: int = 10, J: float = 1.0, n_points: int = 30,
                       show_plots: bool = True) -> dict:
    """
    Run 1D TFIM for comparison (exactly solvable via Jordan-Wigner).
    
    Critical point at Γ_c = J (exact).
    """
    print_header("1D TFIM Comparison")
    
    print(f"1D Chain: L = {L} sites")
    print(f"Exact critical point: Γ_c/J = 1.0")
    
    model = TransverseFieldIsing1D(L=L, J=J, Gamma=1.0)
    solver = ExactDiagonalization(model)
    
    Gamma_values = np.linspace(0.1, 3.0, n_points)
    
    results = solver.phase_diagram_scan(
        Gamma_values,
        observables=['energy_per_site', 'magnetization_z', 'gap']
    )
    
    # Compare with exact Jordan-Wigner solution
    E_exact = []
    for Gamma in Gamma_values:
        model_jw = TransverseFieldIsing1D(L=L, J=J, Gamma=Gamma)
        E_exact.append(model_jw.exact_ground_state_energy())
    
    results['energy_exact'] = np.array(E_exact)
    
    print("\nComparison with Jordan-Wigner exact solution:")
    print(f"  Max energy difference: {np.max(np.abs(results['energy_per_site'] - results['energy_exact'])):.6f}")
    
    if show_plots:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1 = axes[0]
        ax1.plot(Gamma_values, results['energy_per_site'], 'o', label='Numerical', markersize=4)
        ax1.plot(Gamma_values, results['energy_exact'], '-', label='Jordan-Wigner exact')
        ax1.set_xlabel(r'$\Gamma/J$', fontsize=12)
        ax1.set_ylabel('Energy per site', fontsize=12)
        ax1.set_title('1D TFIM Energy', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        ax2.plot(Gamma_values, results['gap'], 'o-', markersize=4)
        ax2.axvline(1.0, color='red', linestyle='--', label=r'$\Gamma_c = J$')
        ax2.set_xlabel(r'$\Gamma/J$', fontsize=12)
        ax2.set_ylabel('Energy gap', fontsize=12)
        ax2.set_title('1D TFIM Gap (critical at Γ=J)', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return results


def run_full_demonstration(L: int = 2, show_plots: bool = True):
    """Run complete demonstration of all simulation capabilities."""
    
    print_header("FULL DEMONSTRATION: 3D Transverse Field Ising Model")
    
    print("""
This simulation studies the quantum phase transition in the 3D Transverse
Field Ising Model:

    H = -J Σ_{<i,j>} σ_i^z σ_j^z - Γ Σ_i σ_i^x

Key physics:
- Ferromagnetic phase (Γ << J): spins ordered along z-axis
- Paramagnetic phase (Γ >> J): spins aligned with transverse field
- Quantum critical point: gap closes, correlation length diverges

Methods used:
1. Exact Diagonalization: Ground state and excited states
2. Quantum Monte Carlo: Finite temperature properties
3. Quantum Annealing: Adiabatic state preparation
    """)
    
    # 1. Basic exact diagonalization
    print("\n[1/5] Exact Diagonalization...")
    ed_results = run_exact_diagonalization(L=L, Gamma=1.0, show_plots=False)
    
    # 2. Phase diagram
    print("\n[2/5] Phase Diagram...")
    phase_results = run_phase_diagram(L=L, n_points=25, show_plots=False, save_plots=True)
    
    # 3. Quantum annealing
    print("\n[3/5] Quantum Annealing...")
    anneal_results = run_quantum_annealing(L=L, t_anneal=10.0, show_plots=False, save_plots=True)
    
    # 4. 1D comparison
    print("\n[4/5] 1D Comparison...")
    results_1d = run_1d_comparison(L=8, n_points=20, show_plots=False)
    
    # 5. Summary
    print_header("SIMULATION COMPLETE")
    
    print("Results saved to ./figures/")
    print("\nKey findings:")
    print(f"  - 3D TFIM ground state energy (Γ=J): {ed_results['energy_per_site']:.4f}")
    print(f"  - Estimated critical point: Γ_c/J ≈ {anneal_results['Gamma_critical']/1.0:.2f}")
    print(f"  - Minimum gap: {anneal_results['min_gap']:.4f}")
    print(f"  - Annealing success probability: {anneal_results['success_probability']:.4f}")
    
    # Create summary figure
    if show_plots:
        viz = Visualization(save_dir='./figures')
        fig = viz.create_summary_figure(
            phase_results, 
            anneal_data=anneal_results,
            title="3D TFIM Simulation Summary"
        )
        fig.savefig('./figures/summary.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    return {
        'exact_diag': ed_results,
        'phase_diagram': phase_results,
        'annealing': anneal_results,
        '1d_comparison': results_1d
    }


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="3D Transverse Field Ising Model Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('mode', nargs='?', default='all',
                       choices=['exact', 'phase', 'qmc', 'anneal', 'sqa', '1d', 'all'],
                       help='Simulation mode')
    parser.add_argument('--L', type=int, default=2,
                       help='Linear lattice size (default: 2)')
    parser.add_argument('--J', type=float, default=1.0,
                       help='Ising coupling (default: 1.0)')
    parser.add_argument('--Gamma', type=float, default=1.0,
                       help='Transverse field (default: 1.0)')
    parser.add_argument('--beta', type=float, default=2.0,
                       help='Inverse temperature for QMC (default: 2.0)')
    parser.add_argument('--t-anneal', type=float, default=10.0,
                       help='Annealing time (default: 10.0)')
    parser.add_argument('--n-points', type=int, default=30,
                       help='Number of parameter points (default: 30)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Disable plotting')
    parser.add_argument('--save-results', action='store_true',
                       help='Save results to JSON')
    
    args = parser.parse_args()
    
    show_plots = not args.no_plots
    
    if args.mode == 'exact':
        results = run_exact_diagonalization(
            L=args.L, J=args.J, Gamma=args.Gamma, show_plots=show_plots
        )
    elif args.mode == 'phase':
        results = run_phase_diagram(
            L=args.L, J=args.J, n_points=args.n_points, show_plots=show_plots
        )
    elif args.mode == 'qmc':
        results = run_qmc_phase_diagram(
            L=args.L, J=args.J, beta=args.beta, n_points=args.n_points,
            show_plots=show_plots
        )
    elif args.mode == 'anneal':
        results = run_quantum_annealing(
            L=args.L, J=args.J, t_anneal=args.t_anneal, show_plots=show_plots
        )
    elif args.mode == 'sqa':
        results = run_simulated_quantum_annealing(
            L=args.L, J=args.J, show_plots=show_plots
        )
    elif args.mode == '1d':
        results = run_1d_comparison(
            L=args.L, J=args.J, n_points=args.n_points, show_plots=show_plots
        )
    else:  # 'all'
        results = run_full_demonstration(L=args.L, show_plots=show_plots)
    
    if args.save_results:
        save_results(results, f'simulation_{args.mode}.json')
    
    return results


if __name__ == "__main__":
    main()
