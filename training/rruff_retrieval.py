"""
Spectral Angle Distance (SAD) Retrieval for RRUFF.

Scientifically correct approach: treat RRUFF as a reference library,
use SAD/cosine similarity for nearest-neighbor retrieval.

Metrics:
- Top-1, Top-3, Top-5 family retrieval accuracy
- Mean Reciprocal Rank (MRR)
- SAD distribution between/within families
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.pipeline import PreprocessingPipeline
from utils.reproducibility import set_all_seeds


def spectral_angle_distance(spec1: np.ndarray, spec2: np.ndarray) -> float:
    """
    Compute Spectral Angle Distance between two spectra.
    
    SAD = arccos(cos_similarity) in radians
    Range: [0, π] where 0 = identical, π/2 = orthogonal
    """
    dot = np.dot(spec1, spec2)
    norm1 = np.linalg.norm(spec1)
    norm2 = np.linalg.norm(spec2)
    
    if norm1 == 0 or norm2 == 0:
        return np.pi / 2  # Orthogonal if zero spectrum
    
    cos_sim = np.clip(dot / (norm1 * norm2), -1, 1)
    return np.arccos(cos_sim)


def compute_sad_matrix(spectra: np.ndarray) -> np.ndarray:
    """Compute pairwise SAD matrix."""
    n = len(spectra)
    sad_matrix = np.zeros((n, n))
    
    # Normalize for faster computation
    norms = np.linalg.norm(spectra, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = spectra / norms
    
    # Cosine similarity matrix
    cos_sim = normalized @ normalized.T
    cos_sim = np.clip(cos_sim, -1, 1)
    
    # Convert to SAD
    sad_matrix = np.arccos(cos_sim)
    
    return sad_matrix


def retrieve_top_k(query_idx: int, sad_matrix: np.ndarray, labels: np.ndarray, 
                   k: int = 5, exclude_self: bool = True) -> list:
    """
    Retrieve top-k nearest neighbors by SAD.
    
    Returns: list of (idx, label, sad) tuples
    """
    distances = sad_matrix[query_idx].copy()
    
    if exclude_self:
        distances[query_idx] = np.inf
    
    top_k_idx = np.argsort(distances)[:k]
    
    return [(idx, labels[idx], distances[idx]) for idx in top_k_idx]


def evaluate_retrieval(test_indices: np.ndarray, sad_matrix: np.ndarray, 
                       labels: np.ndarray, class_names: list, k_values: list = [1, 3, 5]):
    """
    Evaluate retrieval performance.
    
    Returns: dict with top-k accuracy and MRR
    """
    results = {f'top{k}': [] for k in k_values}
    reciprocal_ranks = []
    
    for query_idx in test_indices:
        query_label = labels[query_idx]
        
        # Get all neighbors sorted by distance
        distances = sad_matrix[query_idx].copy()
        distances[query_idx] = np.inf  # Exclude self
        sorted_idx = np.argsort(distances)
        
        # Find rank of first correct match
        first_correct_rank = None
        for rank, idx in enumerate(sorted_idx, 1):
            if labels[idx] == query_label:
                first_correct_rank = rank
                break
        
        if first_correct_rank:
            reciprocal_ranks.append(1.0 / first_correct_rank)
        else:
            reciprocal_ranks.append(0.0)
        
        # Check top-k accuracy
        for k in k_values:
            top_k_labels = labels[sorted_idx[:k]]
            results[f'top{k}'].append(query_label in top_k_labels)
    
    # Aggregate
    metrics = {}
    for k in k_values:
        metrics[f'top{k}_accuracy'] = np.mean(results[f'top{k}'])
    metrics['mrr'] = np.mean(reciprocal_ranks)
    
    return metrics


def compute_inter_intra_sad(sad_matrix: np.ndarray, labels: np.ndarray, class_names: list):
    """
    Compute intra-class and inter-class SAD distributions.
    """
    intra_class = defaultdict(list)
    inter_class = []
    
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            sad = sad_matrix[i, j]
            if labels[i] == labels[j]:
                intra_class[class_names[labels[i]]].append(sad)
            else:
                inter_class.append(sad)
    
    return dict(intra_class), inter_class


def load_rruff_families_clean(
    rruff_dir: str = '/home/john/projects/raman_spectroscopy_pipeline/rruff',
    subset: str = 'excellent_unoriented',
    min_samples: int = 30
):
    """Load RRUFF with cleaned family labels (no Unknown/Other)."""
    from pathlib import Path
    from scipy.interpolate import interp1d
    from training.train_rruff_families import classify_mineral_family
    
    subset_dir = Path(rruff_dir) / subset
    txt_files = [f for f in subset_dir.glob('*.txt') 
                 if not f.name.endswith('Zone.Identifier') and 'Processed' in f.name]
    
    all_spectra = []
    all_families = []
    
    for filepath in txt_files:
        try:
            metadata = {}
            wavenumbers = []
            intensities = []
            
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('##') and '=' in line:
                        key, value = line[2:].split('=', 1)
                        metadata[key.strip()] = value.strip()
                    elif ',' in line and not line.startswith('#'):
                        try:
                            parts = line.split(',')
                            if len(parts) >= 2:
                                wavenumbers.append(float(parts[0].strip()))
                                intensities.append(float(parts[1].strip()))
                        except ValueError:
                            continue
            
            if len(wavenumbers) < 100:
                continue
            
            ideal_chem = metadata.get('IDEAL CHEMISTRY', '')
            family = classify_mineral_family(ideal_chem)
            
            # Skip ambiguous classes
            if family in ['Unknown', 'Other']:
                continue
            
            all_spectra.append((np.array(wavenumbers), np.array(intensities)))
            all_families.append(family)
            
        except Exception:
            continue
    
    # Filter by sample count
    from collections import Counter
    family_counts = Counter(all_families)
    valid_families = {f for f, c in family_counts.items() if c >= min_samples}
    
    mask = [f in valid_families for f in all_families]
    filtered_spectra = [s for s, m in zip(all_spectra, mask) if m]
    filtered_families = [f for f, m in zip(all_families, mask) if m]
    
    # Resample to common grid
    min_wn = max(wn[0] for wn, _ in filtered_spectra)
    max_wn = min(wn[-1] for wn, _ in filtered_spectra)
    target_wn = np.linspace(max(min_wn, 100), min(max_wn, 2000), 1000)
    
    resampled = []
    valid_families_final = []
    
    for (wn, intensity), family in zip(filtered_spectra, filtered_families):
        if wn.min() > target_wn[0] or wn.max() < target_wn[-1]:
            continue
        f = interp1d(wn, intensity, kind='linear', bounds_error=False, fill_value=0)
        resampled.append(f(target_wn))
        valid_families_final.append(family)
    
    spectra = np.array(resampled)
    unique_families = sorted(set(valid_families_final))
    family_to_idx = {f: i for i, f in enumerate(unique_families)}
    labels = np.array([family_to_idx[f] for f in valid_families_final])
    
    return spectra, labels, target_wn, unique_families


def main():
    SEED = 42
    set_all_seeds(SEED)
    
    print("=" * 70)
    print("RRUFF SPECTRAL RETRIEVAL (SAD-Based)")
    print("Library Matching Approach")
    print("=" * 70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'rruff_retrieval')
    os.makedirs(os.path.join(RESULTS_DIR, 'plots'), exist_ok=True)
    
    # Load data
    print("\n[1/4] Loading RRUFF (cleaned families)...")
    spectra, labels, wavenumbers, class_names = load_rruff_families_clean()
    n_classes = len(class_names)
    
    print(f"  Loaded: {len(spectra)} spectra, {n_classes} families")
    print(f"  Families: {class_names}")
    
    # Preprocess
    print("\n[2/4] Preprocessing...")
    pipeline = PreprocessingPipeline()
    spectra = pipeline.process_batch(spectra)
    
    # Normalize for SAD
    norms = np.linalg.norm(spectra, axis=1, keepdims=True)
    norms[norms == 0] = 1
    spectra_normalized = spectra / norms
    
    # Split for evaluation (but keep all as reference library)
    _, test_idx = train_test_split(
        np.arange(len(labels)), test_size=0.2, random_state=SEED, stratify=labels
    )
    
    print(f"  Test queries: {len(test_idx)}")
    
    # Compute SAD matrix
    print("\n[3/4] Computing SAD matrix...")
    sad_matrix = compute_sad_matrix(spectra_normalized)
    print(f"  SAD matrix shape: {sad_matrix.shape}")
    print(f"  Mean SAD: {sad_matrix[np.triu_indices_from(sad_matrix, k=1)].mean():.4f} rad")
    
    # Evaluate retrieval
    print("\n[4/4] Evaluating retrieval...")
    metrics = evaluate_retrieval(test_idx, sad_matrix, labels, class_names, k_values=[1, 3, 5, 10])
    
    print(f"\n  Top-1 Accuracy: {metrics['top1_accuracy']:.4f}")
    print(f"  Top-3 Accuracy: {metrics['top3_accuracy']:.4f}")
    print(f"  Top-5 Accuracy: {metrics['top5_accuracy']:.4f}")
    print(f"  Top-10 Accuracy: {metrics['top10_accuracy']:.4f}")
    print(f"  MRR: {metrics['mrr']:.4f}")
    
    # Intra vs inter class SAD
    intra_class, inter_class = compute_inter_intra_sad(sad_matrix, labels, class_names)
    
    all_intra = []
    for family, sads in intra_class.items():
        all_intra.extend(sads)
    
    print(f"\n  Intra-class mean SAD: {np.mean(all_intra):.4f} rad")
    print(f"  Inter-class mean SAD: {np.mean(inter_class):.4f} rad")
    print(f"  Separation ratio: {np.mean(inter_class) / np.mean(all_intra):.2f}x")
    
    # Plot SAD distributions
    plt.figure(figsize=(10, 6))
    plt.hist(all_intra, bins=50, alpha=0.7, label=f'Intra-class (μ={np.mean(all_intra):.3f})', density=True)
    plt.hist(inter_class, bins=50, alpha=0.7, label=f'Inter-class (μ={np.mean(inter_class):.3f})', density=True)
    plt.xlabel('Spectral Angle Distance (radians)')
    plt.ylabel('Density')
    plt.title('SAD Distribution: Within vs Between Families')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'plots', 'sad_distribution.png'), dpi=150)
    plt.close()
    
    # Per-family intra-class SAD
    plt.figure(figsize=(12, 6))
    family_means = [(f, np.mean(sads)) for f, sads in intra_class.items() if len(sads) > 10]
    family_means.sort(key=lambda x: x[1])
    plt.barh([f[0] for f in family_means], [f[1] for f in family_means])
    plt.xlabel('Mean Intra-class SAD (radians)')
    plt.title('Within-Family Spectral Variability')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'plots', 'family_variability.png'), dpi=150)
    plt.close()
    
    # Save results
    results = {
        'metrics': metrics,
        'intra_class_mean_sad': float(np.mean(all_intra)),
        'inter_class_mean_sad': float(np.mean(inter_class)),
        'separation_ratio': float(np.mean(inter_class) / np.mean(all_intra)),
        'n_spectra': len(spectra),
        'n_families': n_classes,
        'families': class_names,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(RESULTS_DIR, 'retrieval_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "=" * 70)
    print("RETRIEVAL RESULTS")
    print("=" * 70)
    print(f"\n| Metric | Value |")
    print(f"|--------|-------|")
    print(f"| Top-1 | {metrics['top1_accuracy']:.4f} |")
    print(f"| Top-3 | {metrics['top3_accuracy']:.4f} |")
    print(f"| Top-5 | {metrics['top5_accuracy']:.4f} |")
    print(f"| MRR | {metrics['mrr']:.4f} |")
    print(f"| Separation | {np.mean(inter_class) / np.mean(all_intra):.2f}x |")
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    
    return results


if __name__ == "__main__":
    main()
