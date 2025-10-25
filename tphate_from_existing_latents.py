#!/usr/bin/env python3
"""
Apply T-PHATE to existing latent vectors (extracted with correct sampling)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from glob import glob

def apply_pca(data, n_components=2):
    """Apply PCA for dimensionality reduction"""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    embedding = pca.fit_transform(data)
    print(f"PCA embedding shape: {embedding.shape}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    return embedding

def apply_tsne(data, n_components=2):
    """Apply t-SNE for non-linear dimensionality reduction"""
    from sklearn.manifold import TSNE
    n_samples = data.shape[0]
    perplexity = min(30, max(5, n_samples // 3))
    
    print(f"Applying t-SNE with perplexity={perplexity}...")
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    embedding = tsne.fit_transform(data)
    print(f"t-SNE embedding shape: {embedding.shape}")
    return embedding

def build_adaptive_graph(data, k=5, decay=40, n_pca=None):
    """Build adaptive k-NN graph like in PHATE"""
    print("Building adaptive k-NN graph...")
    
    n_samples, n_features = data.shape
    
    # PCA preprocessing if needed
    if n_pca is not None and n_features > n_pca and n_samples > n_pca:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(n_pca, n_samples-1, n_features))
        data_pca = pca.fit_transform(data)
        print(f"PCA preprocessing: {n_features} -> {pca.n_components_} features")
    else:
        data_pca = data
    
    # Compute pairwise distances
    from scipy.spatial.distance import pdist, squareform
    distances = pdist(data_pca, metric='euclidean')
    dist_matrix = squareform(distances)
    
    # Adaptive bandwidth selection
    sorted_dists = np.sort(dist_matrix, axis=1)
    adaptive_bandwidths = sorted_dists[:, k]
    
    # Build adaptive k-NN graph
    n = len(data_pca)
    kernel_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                sigma_i = adaptive_bandwidths[i]
                kernel_matrix[i, j] = np.exp(-(dist_matrix[i, j]**2) / (2 * sigma_i**2))
    
    # Keep only k nearest neighbors
    for i in range(n):
        neighbor_indices = np.argsort(dist_matrix[i])[1:k+1]
        mask = np.ones(n, dtype=bool)
        mask[neighbor_indices] = False
        mask[i] = True
        kernel_matrix[i, mask] = 0
    
    # Make symmetric
    kernel_matrix = (kernel_matrix + kernel_matrix.T) / 2
    
    n_edges = np.sum(kernel_matrix > 0) // 2
    print(f"Adaptive graph built: {n_edges} edges")
    
    return kernel_matrix

def apply_diffusion(kernel_matrix, t=1):
    """Apply diffusion operator"""
    print(f"Applying diffusion operator with t={t}...")
    
    row_sums = kernel_matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1
    transition_matrix = kernel_matrix / row_sums[:, np.newaxis]
    
    diffused_matrix = np.linalg.matrix_power(transition_matrix, t)
    
    print(f"Diffusion applied: t={t}")
    return diffused_matrix

def apply_phate_embedding(diffused_matrix, n_components=2):
    """Apply PHATE embedding using MDS"""
    print("Applying PHATE embedding...")
    
    epsilon = 1e-6
    potential_matrix = -np.log(diffused_matrix + epsilon)
    
    # Ensure matrix is symmetric
    potential_matrix = (potential_matrix + potential_matrix.T) / 2
    
    from sklearn.manifold import MDS
    mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=42)
    embedding = mds.fit_transform(potential_matrix)
    
    print(f"PHATE embedding shape: {embedding.shape}")
    return embedding

def apply_tphate(data, n_components=2, k=5, decay=40, t=1, n_pca=None):
    """Apply the complete T-PHATE algorithm"""
    print("Applying T-PHATE algorithm...")
    
    # Step 1: Build adaptive graph
    kernel_matrix = build_adaptive_graph(data, k=k, decay=decay, n_pca=n_pca)
    
    # Step 2: Apply diffusion
    diffused_matrix = apply_diffusion(kernel_matrix, t=t)
    
    # Step 3: Apply PHATE embedding
    embedding = apply_phate_embedding(diffused_matrix, n_components=n_components)
    
    print("T-PHATE completed successfully!")
    return embedding

def plot_trajectory(z_embedding, cell_id, method="PCA", save_path=None):
    """Plot trajectory with clear visualization"""
    print(f"Creating {method} trajectory plot...")
    
    plt.figure(figsize=(12, 10))
    
    # Plot trajectory line
    plt.plot(z_embedding[:, 0], z_embedding[:, 1], 'b-', alpha=0.7, linewidth=3, label='Developmental Trajectory')
    
    # Plot points with time coloring
    scatter = plt.scatter(z_embedding[:, 0], z_embedding[:, 1], 
                        c=range(len(z_embedding)), 
                        cmap='plasma', 
                        s=150, 
                        alpha=0.8,
                        edgecolors='black',
                        linewidth=2)
    
    # Mark start and end points clearly
    plt.scatter(z_embedding[0, 0], z_embedding[0, 1], 
               color='green', s=300, marker='o', 
               label='Start (t=0)', edgecolors='black', linewidth=3)
    plt.scatter(z_embedding[-1, 0], z_embedding[-1, 1], 
               color='red', s=300, marker='s', 
               label='End (t=15)', edgecolors='black', linewidth=3)
    
    # Add time labels
    for i, (x, y) in enumerate(z_embedding):
        plt.annotate(f't{i}', (x, y), xytext=(8, 8), 
                    textcoords='offset points', fontsize=10, 
                    fontweight='bold', alpha=0.8)
    
    # No arrows needed - trajectory line already shows direction clearly
    
    plt.colorbar(scatter, label='Time Step')
    plt.xlabel(f'{method} Component 1', fontsize=14, fontweight='bold')
    plt.ylabel(f'{method} Component 2', fontsize=14, fontweight='bold')
    plt.title(f'Embryo Development Trajectory: {cell_id}\n({method} Visualization, 16 time points)', 
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Calculate trajectory length
    trajectory_length = np.sum(np.sqrt(np.sum(np.diff(z_embedding, axis=0)**2, axis=1)))
    plt.text(0.02, 0.98, f'Time Points: {len(z_embedding)}\nTrajectory Length: {trajectory_length:.3f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.close()

def main():
    print("=== T-PHATE Analysis on Existing Latent Vectors ===")
    
    # Create output directory
    output_dir = Path("tphate_from_latents_results")
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Find all latent vector files
    latent_files = glob("latents_unique/*_z.npy")
    latent_files.sort()
    
    print(f"Found {len(latent_files)} latent vector files")
    
    # Process first 5 embryos
    for i, latent_file in enumerate(latent_files[:5]):
        cell_id = Path(latent_file).stem.replace('_z', '')
        print(f"\n--- Processing Embryo {i+1}/5: {cell_id} ---")
        
        try:
            # Load latent vectors
            z_latent = np.load(latent_file)
            print(f"Loaded latent vectors shape: {z_latent.shape}")
            
            # Apply PCA
            z_pca = apply_pca(z_latent, n_components=2)
            
            # Apply t-SNE
            z_tsne = apply_tsne(z_latent, n_components=2)
            
            # Apply T-PHATE
            z_tphate = apply_tphate(z_latent, n_components=2, k=5, t=1)
            
            # Plot trajectories
            plot_trajectory(z_pca, cell_id, "PCA", 
                          str(output_dir / f"pca_embryo_{cell_id}.png"))
            plot_trajectory(z_tsne, cell_id, "t-SNE", 
                          str(output_dir / f"tsne_embryo_{cell_id}.png"))
            plot_trajectory(z_tphate, cell_id, "T-PHATE", 
                          str(output_dir / f"tphate_embryo_{cell_id}.png"))
            
            print(f"✅ Completed: {cell_id}")
            
        except Exception as e:
            print(f"❌ Error processing {cell_id}: {e}")
            continue
    
    print(f"\n=== Analysis Complete ===")
    print(f"Generated plots saved in: {output_dir}")
    print(f"Total plots generated: {len(list(output_dir.glob('*.png')))}")

if __name__ == "__main__":
    main()
