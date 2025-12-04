"""
Example 4: True 3D System Analysis

This example demonstrates:
- Proper 3D lattice (not quasi-1D)
- 3D nearest-neighbor structure
- Comparison of different lattice geometries
- Scaling with system size
"""

import sys
sys.path.append('..')
from quantum_ising_3d import QuantumIsing3D
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_lattice_3d(Lx, Ly, Lz):
    """Visualize the 3D lattice structure"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot sites
    for x in range(Lx):
        for y in range(Ly):
            for z in range(Lz):
                ax.scatter(x, y, z, c='blue', s=200, marker='o', alpha=0.8)
                
    # Plot bonds
    for x in range(Lx):
        for y in range(Ly):
            for z in range(Lz):
                # X bonds
                if x < Lx - 1:
                    ax.plot([x, x+1], [y, y], [z, z], 'k-', linewidth=2, alpha=0.5)
                # Y bonds
                if y < Ly - 1:
                    ax.plot([x, x], [y, y+1], [z, z], 'k-', linewidth=2, alpha=0.5)
                # Z bonds
                if z < Lz - 1:
                    ax.plot([x, x], [y, y], [z, z+1], 'k-', linewidth=2, alpha=0.5)
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title(f'{Lx}×{Ly}×{Lz} Cubic Lattice', fontsize=14, fontweight='bold')
    
    return fig


def main():
    print("\n" + "="*70)
    print("EXAMPLE 4: 3D Lattice Analysis")
    print("="*70 + "\n")
    
    print("This example explores true 3D systems where each spin has")
    print("up to 6 nearest neighbors (±x, ±y, ±z directions).\n")
    
    # Visualize lattice structure
    print("Visualizing 2x2x2 cubic lattice...")
    fig_lattice = visualize_lattice_3d(2, 2, 2)
    fig_lattice.savefig('lattice_3d_structure.png', dpi=150, bbox_inches='tight')
    print("Saved: lattice_3d_structure.png\n")
    
    # Analyze coordination number (number of neighbors)
    print("="*70)
    print("COORDINATION NUMBER ANALYSIS")
    print("="*70 + "\n")
    
    configs = [
        (4, 1, 1, "1D chain"),
        (2, 2, 1, "2D square"),
        (2, 2, 2, "3D cube"),
    ]
    
    for Lx, Ly, Lz, description in configs:
        model = QuantumIsing3D(Lx, Ly, Lz, J=1.0, Gamma=0.5, periodic=False)
        
        # Count neighbors for each site
        neighbor_counts = [len(model._get_neighbors(i)) for i in range(model.N)]
        avg_neighbors = np.mean(neighbor_counts)
        
        # Count bonds
        num_bonds = 0
        for i in range(model.N):
            neighbors = model._get_neighbors(i)
            num_bonds += len([j for j in neighbors if j > i])
        
        print(f"{description} ({Lx}×{Ly}×{Lz}):")
        print(f"  Sites: {model.N}")
        print(f"  Bonds: {num_bonds}")
        print(f"  Avg neighbors: {avg_neighbors:.2f}")
        print(f"  Hilbert space dim: 2^{model.N} = {2**model.N}")
        print()
    
    # Compare ground states for different geometries
    print("="*70)
    print("GROUND STATE COMPARISON")
    print("="*70 + "\n")
    
    print("Same number of spins (N=4), different geometries:\n")
    
    geometries = [
        (4, 1, 1, "1D chain"),
        (2, 2, 1, "2D square"),
        (2, 1, 2, "quasi-2D"),
    ]
    
    J = 1.0
    Gamma = 0.5
    
    results = []
    
    for Lx, Ly, Lz, description in geometries:
        print(f"{description} ({Lx}×{Ly}×{Lz}):")
        model = QuantumIsing3D(Lx, Ly, Lz, J=J, Gamma=Gamma, periodic=False)
        
        eigenvalues, eigenvectors = model.find_ground_state(k=2)
        ground_state = eigenvectors[:, 0]
        
        E0 = eigenvalues[0]
        gap = eigenvalues[1] - eigenvalues[0]
        mag_z = model.magnetization_z(ground_state)
        mag_x = model.magnetization_x(ground_state)
        
        results.append({
            'desc': description,
            'dims': (Lx, Ly, Lz),
            'E0': E0,
            'gap': gap,
            'mag_z': mag_z,
            'mag_x': mag_x
        })
        
        print(f"  E₀/N: {E0/model.N:.6f}")
        print(f"  Gap: {gap:.6f}")
        print(f"  M_z: {mag_z:.6f}")
        print(f"  M_x: {mag_x:.6f}")
        print()
    
    # Comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Geometry Comparison (N=4 spins, J=1, Γ=0.5)', 
                 fontsize=14, fontweight='bold')
    
    descriptions = [r['desc'] for r in results]
    
    ax = axes[0, 0]
    energies_per_site = [r['E0']/4 for r in results]
    bars = ax.bar(descriptions, energies_per_site, color=['blue', 'red', 'green'])
    ax.set_ylabel('E₀/N', fontsize=12)
    ax.set_title('Ground State Energy per Spin', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    ax = axes[0, 1]
    gaps = [r['gap'] for r in results]
    bars = ax.bar(descriptions, gaps, color=['blue', 'red', 'green'])
    ax.set_ylabel('ΔE', fontsize=12)
    ax.set_title('Energy Gap', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    ax = axes[1, 0]
    mag_z = [abs(r['mag_z']) for r in results]
    bars = ax.bar(descriptions, mag_z, color=['blue', 'red', 'green'])
    ax.set_ylabel('|M_z|', fontsize=12)
    ax.set_title('Z-Magnetization (order parameter)', fontsize=12)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    ax = axes[1, 1]
    mag_x = [abs(r['mag_x']) for r in results]
    bars = ax.bar(descriptions, mag_x, color=['blue', 'red', 'green'])
    ax.set_ylabel('|M_x|', fontsize=12)
    ax.set_title('X-Magnetization', fontsize=12)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    fig.savefig('geometry_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: geometry_comparison.png\n")
    
    # True 3D system analysis
    print("="*70)
    print("TRUE 3D SYSTEM: 2×2×2 Cube")
    print("="*70 + "\n")
    
    Lx, Ly, Lz = 2, 2, 2
    model_3d = QuantumIsing3D(Lx, Ly, Lz, J=J, Gamma=Gamma, periodic=False)
    
    print(f"System: {Lx}×{Ly}×{Lz} = {model_3d.N} spins")
    print(f"Each spin has 3 neighbors (corner of cube)")
    print(f"Total bonds: 12")
    print(f"Hilbert space: 2^{model_3d.N} = {2**model_3d.N} dimensions\n")
    
    eigenvalues_3d, eigenvectors_3d = model_3d.find_ground_state(k=3)
    ground_state_3d = eigenvectors_3d[:, 0]
    
    print(f"\nGround state properties:")
    print(f"  E₀ = {eigenvalues_3d[0]:.6f}")
    print(f"  E₀/N = {eigenvalues_3d[0]/model_3d.N:.6f}")
    print(f"  M_z = {model_3d.magnetization_z(ground_state_3d):.6f}")
    print(f"  M_x = {model_3d.magnetization_x(ground_state_3d):.6f}")
    
    print(f"\nEnergy spectrum:")
    for i in range(len(eigenvalues_3d)):
        print(f"  E_{i} = {eigenvalues_3d[i]:.6f}")
    
    # Correlation analysis in 3D
    print("\n" + "="*70)
    print("3D CORRELATION STRUCTURE")
    print("="*70 + "\n")
    
    print("Spin-spin correlations <σ_i^z σ_j^z>:")
    print("(Site labeling: x + y*Lx + z*Lx*Ly)\n")
    
    # Compute all correlations
    corr_matrix = np.zeros((model_3d.N, model_3d.N))
    for i in range(model_3d.N):
        for j in range(model_3d.N):
            if i == j:
                corr_matrix[i, j] = 1.0  # <σ_i^z σ_i^z> = 1
            else:
                corr_matrix[i, j] = model_3d.correlation_zz(i, j, ground_state_3d)
    
    # Visualize correlation matrix
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1)
    ax.set_xlabel('Spin j', fontsize=12)
    ax.set_ylabel('Spin i', fontsize=12)
    ax.set_title('Spin-Spin Correlations <σ_i^z σ_j^z>', fontsize=13, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', fontsize=12)
    
    # Add grid
    ax.set_xticks(range(model_3d.N))
    ax.set_yticks(range(model_3d.N))
    ax.grid(False)
    
    plt.tight_layout()
    fig.savefig('correlations_3d.png', dpi=150, bbox_inches='tight')
    print("Saved: correlations_3d.png\n")
    
    # Print correlations by distance
    print("Correlations grouped by distance:\n")
    
    def manhattan_distance(i, j, model):
        xi, yi, zi = model._index_to_coords(i)
        xj, yj, zj = model._index_to_coords(j)
        return abs(xi - xj) + abs(yi - yj) + abs(zi - zj)
    
    distances = {}
    for i in range(model_3d.N):
        for j in range(i+1, model_3d.N):
            d = manhattan_distance(i, j, model_3d)
            if d not in distances:
                distances[d] = []
            distances[d].append(corr_matrix[i, j])
    
    for d in sorted(distances.keys()):
        avg_corr = np.mean(distances[d])
        std_corr = np.std(distances[d])
        print(f"  Distance {d}: <C> = {avg_corr:.6f} ± {std_corr:.6f} ({len(distances[d])} pairs)")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\n3D Ising model shows:")
    print("  - Higher coordination → stronger ordering")
    print("  - Geometry affects critical behavior")
    print("  - Correlations decay with distance")
    print("  - True 3D critical point differs from 1D/2D")
    print("\nFor large systems, critical point Γ_c depends on dimension:")
    print("  - 1D: Γ_c/J = 1.000 (exact)")
    print("  - 2D: Γ_c/J ≈ 3.0-3.5")
    print("  - 3D: Γ_c/J ≈ 4.5-5.5")
    print("="*70 + "\n")
    
    plt.show()


if __name__ == "__main__":
    main()
