"""
Visualization Tools for 3D Transverse Field Ising Model

Provides publication-quality plots for:
- Phase diagrams
- Order parameters vs. control parameter
- Energy spectra and gaps
- Correlation functions
- Annealing dynamics
- Spin configurations (3D)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Optional, Tuple, Any
import os


class Visualization:
    """
    Visualization suite for TFIM simulations.
    
    Parameters
    ----------
    save_dir : str, optional
        Directory to save figures
    style : str
        Matplotlib style ('default', 'seaborn', 'ggplot', etc.)
    """
    
    def __init__(self, save_dir: str = './figures', style: str = 'default'):
        self.save_dir = save_dir
        self.style = style
        
        # Create save directory if needed
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Set style
        if style != 'default':
            try:
                plt.style.use(style)
            except:
                pass
        
        # Default figure parameters
        self.figsize = (8, 6)
        self.dpi = 150
        self.fontsize = 12
        
        # Color scheme
        self.colors = {
            'energy': '#1f77b4',
            'magnetization_z': '#d62728',
            'magnetization_x': '#2ca02c',
            'gap': '#9467bd',
            'susceptibility': '#ff7f0e',
            'correlation': '#17becf'
        }
    
    def plot_phase_diagram(self, data: Dict[str, np.ndarray],
                            title: str = "Phase Diagram",
                            save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot phase diagram showing order parameters vs Γ/J.
        
        Parameters
        ----------
        data : Dict
            Dictionary with 'Gamma', 'magnetization_z', 'magnetization_x', etc.
        title : str
            Plot title
        save_name : str, optional
            Filename to save figure
            
        Returns
        -------
        plt.Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        Gamma = data['Gamma']
        
        # Top left: Order parameters
        ax1 = axes[0, 0]
        if 'magnetization_z' in data:
            mz = np.array(data['magnetization_z'])
            ax1.plot(Gamma, np.sqrt(np.abs(mz)), 'o-', color=self.colors['magnetization_z'],
                    label=r'$\sqrt{|\langle M_z \rangle|}$', markersize=4)
        if 'magnetization_x' in data:
            ax1.plot(Gamma, data['magnetization_x'], 's-', color=self.colors['magnetization_x'],
                    label=r'$\langle M_x \rangle$', markersize=4)
        ax1.set_xlabel(r'$\Gamma/J$', fontsize=self.fontsize)
        ax1.set_ylabel('Magnetization', fontsize=self.fontsize)
        ax1.legend(fontsize=self.fontsize - 2)
        ax1.set_title('Order Parameters', fontsize=self.fontsize)
        ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax1.grid(True, alpha=0.3)
        
        # Top right: Energy
        ax2 = axes[0, 1]
        if 'energy_per_site' in data:
            ax2.plot(Gamma, data['energy_per_site'], 'o-', color=self.colors['energy'],
                    markersize=4)
        elif 'energy' in data:
            ax2.plot(Gamma, data['energy'], 'o-', color=self.colors['energy'],
                    markersize=4)
        ax2.set_xlabel(r'$\Gamma/J$', fontsize=self.fontsize)
        ax2.set_ylabel(r'Energy per site $E/N$', fontsize=self.fontsize)
        ax2.set_title('Ground State Energy', fontsize=self.fontsize)
        ax2.grid(True, alpha=0.3)
        
        # Bottom left: Energy gap
        ax3 = axes[1, 0]
        if 'gap' in data:
            ax3.plot(Gamma, data['gap'], 'o-', color=self.colors['gap'], markersize=4)
            ax3.set_xlabel(r'$\Gamma/J$', fontsize=self.fontsize)
            ax3.set_ylabel(r'Energy gap $\Delta$', fontsize=self.fontsize)
            ax3.set_title('Excitation Gap', fontsize=self.fontsize)
            ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax3.grid(True, alpha=0.3)
        
        # Bottom right: Susceptibility
        ax4 = axes[1, 1]
        if 'susceptibility' in data:
            ax4.plot(Gamma, data['susceptibility'], 'o-', 
                    color=self.colors['susceptibility'], markersize=4)
            ax4.set_xlabel(r'$\Gamma/J$', fontsize=self.fontsize)
            ax4.set_ylabel(r'Susceptibility $\chi$', fontsize=self.fontsize)
            ax4.set_title('Magnetic Susceptibility', fontsize=self.fontsize)
            ax4.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=self.fontsize + 2)
        plt.tight_layout()
        
        if save_name:
            filepath = os.path.join(self.save_dir, save_name)
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        return fig
    
    def plot_annealing_dynamics(self, data: Dict[str, np.ndarray],
                                  title: str = "Quantum Annealing",
                                  save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot quantum annealing dynamics.
        
        Parameters
        ----------
        data : Dict
            Dictionary with 'times', 'Gamma', 'energy', 'overlap_gs', etc.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        times = data['times']
        
        # Top left: Transverse field schedule
        ax1 = axes[0, 0]
        ax1.plot(times, data['Gamma'], '-', color='#1f77b4', linewidth=2)
        ax1.set_xlabel('Time $t$', fontsize=self.fontsize)
        ax1.set_ylabel(r'$\Gamma(t)$', fontsize=self.fontsize)
        ax1.set_title('Annealing Schedule', fontsize=self.fontsize)
        ax1.grid(True, alpha=0.3)
        
        # Top right: Energy
        ax2 = axes[0, 1]
        ax2.plot(times, data['energy'], '-', color=self.colors['energy'], linewidth=2)
        if 'classical_gs_energy' in data:
            ax2.axhline(data['classical_gs_energy'], color='red', linestyle='--',
                       label='Classical GS', alpha=0.7)
        ax2.set_xlabel('Time $t$', fontsize=self.fontsize)
        ax2.set_ylabel('Energy', fontsize=self.fontsize)
        ax2.set_title('Energy Evolution', fontsize=self.fontsize)
        ax2.legend(fontsize=self.fontsize - 2)
        ax2.grid(True, alpha=0.3)
        
        # Bottom left: Ground state overlap
        ax3 = axes[1, 0]
        if 'overlap_gs' in data:
            ax3.plot(times, data['overlap_gs'], '-', color='#2ca02c', linewidth=2)
            ax3.set_xlabel('Time $t$', fontsize=self.fontsize)
            ax3.set_ylabel(r'$|\langle \psi | \psi_{GS} \rangle|^2$', fontsize=self.fontsize)
            ax3.set_title('Ground State Overlap', fontsize=self.fontsize)
            ax3.set_ylim([0, 1.05])
            ax3.grid(True, alpha=0.3)
        
        # Bottom right: Magnetization
        ax4 = axes[1, 1]
        if 'magnetization_z' in data:
            ax4.plot(times, data['magnetization_z'], '-', 
                    color=self.colors['magnetization_z'], linewidth=2,
                    label=r'$\langle M_z \rangle$')
            ax4.set_xlabel('Time $t$', fontsize=self.fontsize)
            ax4.set_ylabel('Magnetization', fontsize=self.fontsize)
            ax4.set_title('Magnetization Evolution', fontsize=self.fontsize)
            ax4.legend(fontsize=self.fontsize - 2)
            ax4.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=self.fontsize + 2)
        plt.tight_layout()
        
        if save_name:
            filepath = os.path.join(self.save_dir, save_name)
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        return fig
    
    def plot_energy_spectrum(self, Gamma_values: np.ndarray,
                              eigenvalues: List[np.ndarray],
                              n_show: int = 5,
                              title: str = "Energy Spectrum",
                              save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot energy levels as function of Γ.
        
        Parameters
        ----------
        Gamma_values : np.ndarray
            Array of Γ values
        eigenvalues : List[np.ndarray]
            List of eigenvalue arrays for each Γ
        n_show : int
            Number of levels to show
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Convert to arrays
        n_Gamma = len(Gamma_values)
        n_levels = min(n_show, len(eigenvalues[0]))
        
        E = np.zeros((n_Gamma, n_levels))
        for i, evs in enumerate(eigenvalues):
            E[i, :] = evs[:n_levels]
        
        # Plot levels
        colors = plt.cm.viridis(np.linspace(0, 1, n_levels))
        for n in range(n_levels):
            label = 'Ground' if n == 0 else f'Excited {n}'
            ax.plot(Gamma_values, E[:, n], '-', color=colors[n], 
                   label=label, linewidth=2)
        
        ax.set_xlabel(r'$\Gamma/J$', fontsize=self.fontsize)
        ax.set_ylabel('Energy', fontsize=self.fontsize)
        ax.set_title(title, fontsize=self.fontsize)
        ax.legend(fontsize=self.fontsize - 2, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            filepath = os.path.join(self.save_dir, save_name)
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        return fig
    
    def plot_gap_vs_Gamma(self, Gamma_values: np.ndarray,
                           gaps: np.ndarray,
                           Gamma_c: Optional[float] = None,
                           title: str = "Energy Gap",
                           save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot energy gap as function of Γ.
        
        Parameters
        ----------
        Gamma_values : np.ndarray
            Array of Γ values
        gaps : np.ndarray
            Energy gap values
        Gamma_c : float, optional
            Critical point to mark
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(Gamma_values, gaps, 'o-', color=self.colors['gap'], 
               markersize=4, linewidth=1.5)
        
        if Gamma_c is not None:
            ax.axvline(Gamma_c, color='red', linestyle='--', 
                      label=f'$\\Gamma_c = {Gamma_c:.2f}$', alpha=0.7)
            ax.legend(fontsize=self.fontsize - 2)
        
        ax.set_xlabel(r'$\Gamma/J$', fontsize=self.fontsize)
        ax.set_ylabel(r'Gap $\Delta$', fontsize=self.fontsize)
        ax.set_title(title, fontsize=self.fontsize)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        
        if save_name:
            filepath = os.path.join(self.save_dir, save_name)
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        return fig
    
    def plot_correlation_function(self, distances: np.ndarray,
                                    correlations: np.ndarray,
                                    title: str = "Correlation Function",
                                    save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot spin-spin correlation function C(r).
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(distances, correlations, 'o-', color=self.colors['correlation'],
               markersize=6, linewidth=1.5)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Distance $r$', fontsize=self.fontsize)
        ax.set_ylabel(r'$C(r) = \langle \sigma_0^z \sigma_r^z \rangle - \langle \sigma_0^z \rangle \langle \sigma_r^z \rangle$',
                     fontsize=self.fontsize)
        ax.set_title(title, fontsize=self.fontsize)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            filepath = os.path.join(self.save_dir, save_name)
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        return fig
    
    def plot_spin_configuration_3d(self, spins: np.ndarray, L: int,
                                     title: str = "Spin Configuration",
                                     save_name: Optional[str] = None) -> plt.Figure:
        """
        3D visualization of spin configuration.
        
        Parameters
        ----------
        spins : np.ndarray
            Array of ±1 spins
        L : int
            Linear lattice size
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create coordinate arrays
        coords_up = []
        coords_down = []
        
        for z in range(L):
            for y in range(L):
                for x in range(L):
                    idx = x + L * y + L * L * z
                    if spins[idx] > 0:
                        coords_up.append([x, y, z])
                    else:
                        coords_down.append([x, y, z])
        
        coords_up = np.array(coords_up) if coords_up else np.empty((0, 3))
        coords_down = np.array(coords_down) if coords_down else np.empty((0, 3))
        
        # Plot spins
        if len(coords_up) > 0:
            ax.scatter(coords_up[:, 0], coords_up[:, 1], coords_up[:, 2],
                      c='blue', s=200, marker='^', label='Spin up', alpha=0.8)
        if len(coords_down) > 0:
            ax.scatter(coords_down[:, 0], coords_down[:, 1], coords_down[:, 2],
                      c='red', s=200, marker='v', label='Spin down', alpha=0.8)
        
        ax.set_xlabel('X', fontsize=self.fontsize)
        ax.set_ylabel('Y', fontsize=self.fontsize)
        ax.set_zlabel('Z', fontsize=self.fontsize)
        ax.set_title(title, fontsize=self.fontsize)
        ax.legend(fontsize=self.fontsize - 2)
        
        if save_name:
            filepath = os.path.join(self.save_dir, save_name)
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        return fig
    
    def plot_wavefunction_distribution(self, state: np.ndarray,
                                         n_top: int = 20,
                                         title: str = "Wavefunction",
                                         save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot probability distribution of wavefunction.
        
        Shows top n_top basis states by probability.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        probs = np.abs(state) ** 2
        top_indices = np.argsort(probs)[-n_top:][::-1]
        top_probs = probs[top_indices]
        
        # Create labels for basis states
        N = int(np.log2(len(state)))
        labels = [f'|{idx:0{N}b}⟩' for idx in top_indices]
        
        bars = ax.bar(range(n_top), top_probs, color=plt.cm.viridis(top_probs / max(top_probs)))
        ax.set_xticks(range(n_top))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_xlabel('Basis State', fontsize=self.fontsize)
        ax.set_ylabel('Probability', fontsize=self.fontsize)
        ax.set_title(f'{title} (top {n_top} states)', fontsize=self.fontsize)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_name:
            filepath = os.path.join(self.save_dir, save_name)
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        return fig
    
    def plot_annealing_success_vs_time(self, t_anneal_values: np.ndarray,
                                         success_probs: np.ndarray,
                                         title: str = "Annealing Time Scaling",
                                         save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot success probability vs annealing time.
        
        Useful for understanding adiabatic scaling.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.semilogx(t_anneal_values, success_probs, 'o-', 
                   color='#2ca02c', markersize=6, linewidth=1.5)
        ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel(r'Annealing time $t_{\rm anneal}$', fontsize=self.fontsize)
        ax.set_ylabel('Success Probability', fontsize=self.fontsize)
        ax.set_title(title, fontsize=self.fontsize)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        if save_name:
            filepath = os.path.join(self.save_dir, save_name)
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        return fig
    
    def plot_qmc_convergence(self, data: Dict[str, np.ndarray],
                              title: str = "QMC Convergence",
                              save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot QMC observable convergence.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        for ax, (key, values) in zip(axes.flat, data.items()):
            if isinstance(values, (list, np.ndarray)) and len(values) > 1:
                ax.plot(values, '-', alpha=0.7)
                ax.axhline(np.mean(values), color='red', linestyle='--', 
                          label=f'Mean = {np.mean(values):.4f}')
                ax.set_xlabel('MC Step', fontsize=self.fontsize)
                ax.set_ylabel(key, fontsize=self.fontsize)
                ax.set_title(key, fontsize=self.fontsize)
                ax.legend(fontsize=self.fontsize - 2)
                ax.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=self.fontsize + 2)
        plt.tight_layout()
        
        if save_name:
            filepath = os.path.join(self.save_dir, save_name)
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        return fig
    
    def create_summary_figure(self, ed_data: Dict, qmc_data: Optional[Dict] = None,
                               anneal_data: Optional[Dict] = None,
                               title: str = "TFIM Simulation Summary",
                               save_name: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive summary figure with all key results.
        """
        n_plots = 3 if qmc_data or anneal_data else 2
        fig, axes = plt.subplots(2, n_plots, figsize=(4 * n_plots + 2, 8))
        axes = axes.flatten()
        
        # Plot 1: Phase diagram from ED
        ax = axes[0]
        Gamma = ed_data['Gamma']
        if 'magnetization_z' in ed_data:
            mz = ed_data['magnetization_z']
            ax.plot(Gamma, np.sqrt(np.abs(mz)), 'o-', label=r'$\sqrt{|M_z|}$', color='blue')
        if 'magnetization_x' in ed_data:
            ax.plot(Gamma, ed_data['magnetization_x'], 's-', label=r'$M_x$', color='red')
        ax.set_xlabel(r'$\Gamma/J$')
        ax.set_ylabel('Order Parameter')
        ax.set_title('Phase Diagram (ED)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Energy gap
        ax = axes[1]
        if 'gap' in ed_data:
            ax.plot(Gamma, ed_data['gap'], 'o-', color='purple')
        ax.set_xlabel(r'$\Gamma/J$')
        ax.set_ylabel(r'Gap $\Delta$')
        ax.set_title('Energy Gap')
        ax.grid(True, alpha=0.3)
        
        idx = 2
        
        # Plot 3: QMC comparison (if available)
        if qmc_data:
            ax = axes[idx]
            if 'magnetization_z' in qmc_data:
                ax.errorbar(qmc_data['Gamma'], qmc_data['magnetization_z'],
                           yerr=qmc_data.get('magnetization_z_error', None),
                           fmt='o-', capsize=3, label='QMC')
            ax.set_xlabel(r'$\Gamma/J$')
            ax.set_ylabel(r'$\langle M_z \rangle$')
            ax.set_title('QMC Results')
            ax.legend()
            ax.grid(True, alpha=0.3)
            idx += 1
        
        # Plot 4: Annealing (if available)
        if anneal_data:
            ax = axes[idx]
            if 'times' in anneal_data and 'overlap_gs' in anneal_data:
                ax.plot(anneal_data['times'], anneal_data['overlap_gs'], '-')
            ax.set_xlabel('Time')
            ax.set_ylabel('GS Overlap')
            ax.set_title('Quantum Annealing')
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_name:
            filepath = os.path.join(self.save_dir, save_name)
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        return fig


def quick_plot_demo():
    """Quick demonstration of visualization capabilities."""
    # Generate sample data
    Gamma = np.linspace(0.1, 5.0, 30)
    
    # Mock phase diagram data
    data = {
        'Gamma': Gamma,
        'magnetization_z': np.exp(-Gamma / 2),
        'magnetization_x': 1 - np.exp(-Gamma / 2),
        'energy_per_site': -1.5 - 0.5 * Gamma - 0.1 * np.sin(Gamma),
        'gap': 0.5 * np.abs(Gamma - 2.5) + 0.1
    }
    
    viz = Visualization(save_dir='./figures')
    fig = viz.plot_phase_diagram(data, title="Sample Phase Diagram")
    plt.show()
    
    return fig


if __name__ == "__main__":
    quick_plot_demo()
