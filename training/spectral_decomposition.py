"""
Spectral Decomposition / Unmixing (Fixed Version).

Uses sklearn NMF only (RamanSPy NFINDR/VCA have scipy compatibility issues).
Increased iterations for convergence.
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.pipeline import PreprocessingPipeline


def generate_synthetic_mixtures(n_endmembers=3, n_samples=200, n_wavenumbers=1000, seed=42):
    """Generate synthetic mixed spectra simulating lunar regolith."""
    np.random.seed(seed)
    wavenumbers = np.linspace(400, 1800, n_wavenumbers)
    
    # Lunar mineral-like peak profiles
    endmember_peaks = [
        [(520, 1.0, 25), (670, 0.4, 20), (1000, 0.6, 30)],   # Olivine
        [(660, 0.9, 22), (1000, 1.0, 28), (1350, 0.3, 18)],  # Pyroxene
        [(510, 0.7, 20), (780, 1.0, 25), (1050, 0.5, 22)],   # Plagioclase
    ]
    
    endmember_names = ["Olivine", "Pyroxene", "Plagioclase"][:n_endmembers]
    
    # Generate endmembers
    true_endmembers = []
    for i in range(n_endmembers):
        spectrum = np.zeros(n_wavenumbers)
        for center, amp, width in endmember_peaks[i]:
            spectrum += amp * np.exp(-((wavenumbers - center) ** 2) / (2 * width ** 2))
        true_endmembers.append(spectrum)
    true_endmembers = np.array(true_endmembers)
    
    # Random abundances (sum to 1)
    true_abundances = np.random.dirichlet(np.ones(n_endmembers) * 2, n_samples)  # Smoother distribution
    
    # Create mixtures
    mixtures = true_abundances @ true_endmembers
    
    # Add realistic noise
    mixtures += np.random.normal(0, 0.01, mixtures.shape)
    mixtures = np.maximum(mixtures, 0)
    
    metadata = {
        'source': 'synthetic',
        'n_endmembers': n_endmembers,
        'n_samples': n_samples,
        'seed': seed
    }
    
    return mixtures, true_endmembers, true_abundances, wavenumbers, endmember_names, metadata


def unmix_nmf(spectra, n_endmembers, max_iter=1000, tol=1e-4):
    """
    NMF-based unmixing with improved convergence settings.
    """
    spectra = np.maximum(spectra, 0)
    
    nmf = NMF(
        n_components=n_endmembers, 
        init='nndsvda',  # Better initialization for sparse data
        max_iter=max_iter,
        tol=tol,
        random_state=42,
        solver='mu',  # Multiplicative update, more stable
        beta_loss='frobenius'
    )
    
    abundances = nmf.fit_transform(spectra)
    endmembers = nmf.components_
    
    # Normalize abundances to sum to 1
    row_sums = abundances.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    abundances = abundances / row_sums
    
    reconstruction_error = nmf.reconstruction_err_
    n_iter = nmf.n_iter_
    
    print(f"  NMF converged in {n_iter} iterations (max: {max_iter})")
    print(f"  Reconstruction error: {reconstruction_error:.6f}")
    
    return abundances, endmembers, n_iter


def evaluate_unmixing(true_abundances, pred_abundances, true_endmembers, pred_endmembers):
    """Evaluate unmixing quality with proper endmember matching."""
    # Match predicted to true endmembers using Hungarian algorithm
    cost_matrix = cdist(pred_endmembers, true_endmembers, 'cosine')
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Reorder predictions
    pred_endmembers_aligned = pred_endmembers[row_ind]
    pred_abundances_aligned = pred_abundances[:, row_ind]
    
    # Metrics
    abundance_mse = np.mean((true_abundances - pred_abundances_aligned) ** 2)
    abundance_mae = np.mean(np.abs(true_abundances - pred_abundances_aligned))
    
    # Spectral Angle Distance for endmembers
    sad_values = []
    for i in range(len(true_endmembers)):
        dot = np.dot(true_endmembers[i], pred_endmembers_aligned[i])
        norm_t = np.linalg.norm(true_endmembers[i])
        norm_p = np.linalg.norm(pred_endmembers_aligned[i])
        cos_sim = np.clip(dot / (norm_t * norm_p), -1, 1)
        sad_values.append(np.arccos(cos_sim))
    
    metrics = {
        'abundance_mse': float(abundance_mse),
        'abundance_mae': float(abundance_mae),
        'endmember_sad_mean': float(np.mean(sad_values)),
        'endmember_sad_std': float(np.std(sad_values))
    }
    
    return metrics, pred_abundances_aligned, pred_endmembers_aligned


def main():
    print("=" * 60)
    print("SPECTRAL DECOMPOSITION / UNMIXING (FIXED)")
    print("=" * 60)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'unmixing')
    os.makedirs(os.path.join(RESULTS_DIR, 'plots'), exist_ok=True)
    
    # Generate synthetic lunar regolith mixtures
    print("\n[1/4] Generating synthetic lunar regolith mixtures...")
    mixtures, true_endmembers, true_abundances, wavenumbers, endmember_names, synth_metadata = \
        generate_synthetic_mixtures(n_endmembers=3, n_samples=200, n_wavenumbers=1000)
    
    print(f"  Generated {len(mixtures)} mixtures of {len(endmember_names)} minerals")
    print(f"  Minerals: {', '.join(endmember_names)}")
    
    # Preprocess
    print("\n[2/4] Preprocessing...")
    pipeline = PreprocessingPipeline(use_baseline_correction=True, use_normalization=True)
    mixtures_processed = pipeline.process_batch(mixtures)
    true_endmembers_processed = pipeline.process_batch(true_endmembers)
    
    # Unmix with improved NMF
    print("\n[3/4] Performing NMF unmixing...")
    pred_abundances, pred_endmembers, n_iter = unmix_nmf(
        mixtures_processed, 
        n_endmembers=len(endmember_names),
        max_iter=1000
    )
    
    metrics, pred_abundances_aligned, pred_endmembers_aligned = evaluate_unmixing(
        true_abundances, pred_abundances, 
        true_endmembers_processed, pred_endmembers
    )
    
    print(f"\n  Abundance MAE: {metrics['abundance_mae']:.4f}")
    print(f"  Endmember SAD: {metrics['endmember_sad_mean']:.4f} ± {metrics['endmember_sad_std']:.4f} rad")
    
    # Visualize
    print("\n[4/4] Generating plots...")
    
    # Endmember comparison
    fig, axes = plt.subplots(len(endmember_names), 1, figsize=(12, 3*len(endmember_names)))
    for i in range(len(endmember_names)):
        axes[i].plot(wavenumbers, true_endmembers_processed[i], 'b-', label='True', linewidth=2)
        axes[i].plot(wavenumbers, pred_endmembers_aligned[i], 'r--', label='Predicted', linewidth=2)
        axes[i].set_title(f'{endmember_names[i]}')
        axes[i].set_xlabel('Wavenumber (cm⁻¹)')
        axes[i].set_ylabel('Intensity')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'plots', 'endmembers_comparison.png'), dpi=150)
    plt.close()
    
    # Abundance scatter
    fig, axes = plt.subplots(1, len(endmember_names), figsize=(4*len(endmember_names), 4))
    for i in range(len(endmember_names)):
        axes[i].scatter(true_abundances[:, i], pred_abundances_aligned[:, i], alpha=0.5, s=20)
        axes[i].plot([0, 1], [0, 1], 'r--', linewidth=2)
        axes[i].set_xlabel('True Abundance')
        axes[i].set_ylabel('Predicted Abundance')
        axes[i].set_title(f'{endmember_names[i]}')
        axes[i].set_xlim(0, 1)
        axes[i].set_ylim(0, 1)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'plots', 'abundance_scatter.png'), dpi=150)
    plt.close()
    
    # Save results with provenance
    results = {
        'method': 'NMF',
        'metrics': metrics,
        'nmf_iterations': n_iter,
        'data_source': 'synthetic',
        'endmember_names': endmember_names,
        'n_mixtures': len(mixtures),
        'synth_metadata': synth_metadata,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(RESULTS_DIR, 'unmixing_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Method: NMF (sklearn)")
    print(f"  Abundance MAE: {metrics['abundance_mae']:.4f}")
    print(f"  Endmember SAD: {metrics['endmember_sad_mean']:.4f} rad")
    print(f"  Data source: SYNTHETIC")
    print(f"\n  Results saved to: {RESULTS_DIR}")
    
    return results


if __name__ == "__main__":
    main()
