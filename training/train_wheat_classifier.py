"""
Wheat Lines Classification Training (Fixed Version).

Features:
- Hard fail on data load (no silent synthetic fallback)
- Uses cached preprocessed data when available  
- Exports provenance metadata with all results
- LightweightCNN with baseline comparison
"""

import sys
import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.pipeline import PreprocessingPipeline
from models.cnn_classifier import LightweightCNN, SpectralDataset, train_model
from data.data_loader import load_wheat_lines, DataBundle
from data.cache_manager import (compute_config_hash, get_cache_path, save_to_cache, 
                                 load_from_cache, cache_exists, get_preprocessing_config)
from data.exceptions import DataLoadError, CacheInvalidError


class Timer:
    def __init__(self):
        self.times = {}
        self.start_time = None
        self.total_start = time.time()
    
    def start(self, name):
        self.start_time = time.time()
        print(f"\n⏱️  {name}")
    
    def stop(self, name):
        elapsed = time.time() - self.start_time
        self.times[name] = elapsed
        print(f"   ✓ {elapsed:.1f}s ({elapsed/60:.2f} min)")
    
    def summary(self):
        total = time.time() - self.total_start
        print("\n" + "=" * 50)
        print("TIMING SUMMARY")
        for name, t in self.times.items():
            print(f"  {name}: {t:.1f}s")
        print(f"  TOTAL: {total:.1f}s ({total/60:.2f} min)")
        return total


def train_pls_da(X_train, y_train, X_test, y_test, n_components=10):
    """PLS-DA baseline with proper handling."""
    lb = LabelBinarizer()
    Y_train = lb.fit_transform(y_train)
    if Y_train.shape[1] == 1:
        Y_train = np.hstack([1 - Y_train, Y_train])
    
    n_comp = min(n_components, X_train.shape[1], X_train.shape[0] - 1)
    pls = PLSRegression(n_components=n_comp)
    pls.fit(X_train, Y_train)
    y_pred = pls.predict(X_test).argmax(axis=1)
    return accuracy_score(y_test, y_pred)


def train_svm(X_train, y_train, X_test, y_test, max_samples=5000):
    """SVM baseline (subsampled for speed)."""
    if len(X_train) > max_samples:
        idx = np.random.choice(len(X_train), max_samples, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
    
    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    svm.fit(X_train, y_train)
    return accuracy_score(y_test, svm.predict(X_test))


def main():
    print("=" * 60)
    print("WHEAT LINES CLASSIFICATION (FIXED)")
    print("=" * 60)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    timer = Timer()
    
    # Configuration
    N_WAVENUMBERS = 1000
    SPLIT_SEED = 42
    BATCH_SIZE = 64
    EPOCHS = 50
    PATIENCE = 10
    
    RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'wheat')
    os.makedirs(os.path.join(RESULTS_DIR, 'models'), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'plots'), exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Initialize pipeline for config hash
    pipeline = PreprocessingPipeline()
    preprocessing_config = get_preprocessing_config(pipeline)
    config_hash = compute_config_hash('wheat_lines', preprocessing_config, SPLIT_SEED, N_WAVENUMBERS)
    
    # Check cache
    cache_path = get_cache_path('wheat_lines', config_hash)
    
    if cache_exists('wheat_lines', config_hash):
        timer.start("Loading from cache")
        try:
            (spectra, labels, wavenumbers, class_names, metadata,
             train_idx, val_idx, test_idx) = load_from_cache(cache_path)
            
            X_train, X_val, X_test = spectra[train_idx], spectra[val_idx], spectra[test_idx]
            y_train, y_val, y_test = labels[train_idx], labels[val_idx], labels[test_idx]
            timer.stop("Loading from cache")
        except CacheInvalidError as e:
            print(f"  Cache invalid: {e}")
            cache_exists_flag = False
    else:
        # Load and preprocess from scratch
        timer.start("Loading wheat_lines")
        target_wn = np.linspace(400, 1800, N_WAVENUMBERS)
        
        try:
            bundle = load_wheat_lines(allow_synthetic=False, target_wavenumbers=target_wn)
        except DataLoadError as e:
            print(f"\n❌ FATAL: {e}")
            print("Cannot proceed without real wheat_lines data.")
            return None
        
        spectra = bundle.spectra
        labels = bundle.labels
        wavenumbers = bundle.wavenumbers
        class_names = bundle.class_names
        metadata = bundle.metadata
        timer.stop("Loading wheat_lines")
        
        # Preprocess
        timer.start("Preprocessing")
        spectra = pipeline.process_batch(spectra)
        timer.stop("Preprocessing")
        
        # Create deterministic splits
        X_train_full, X_test, y_train_full, y_test, train_full_idx, test_idx = train_test_split(
            spectra, labels, np.arange(len(labels)),
            test_size=0.2, random_state=SPLIT_SEED, stratify=labels
        )
        X_train, X_val, y_train, y_val, train_idx_rel, val_idx_rel = train_test_split(
            X_train_full, y_train_full, np.arange(len(y_train_full)),
            test_size=0.15, random_state=SPLIT_SEED, stratify=y_train_full
        )
        
        # Convert relative indices to absolute
        train_idx = train_full_idx[train_idx_rel]
        val_idx = train_full_idx[val_idx_rel]
        
        # Save to cache
        timer.start("Saving to cache")
        save_to_cache(cache_path, spectra, labels, wavenumbers, class_names, metadata,
                      train_idx, val_idx, test_idx)
        timer.stop("Saving to cache")
    
    n_classes = len(class_names)
    print(f"\n  Data: {len(spectra)} samples, {n_classes} classes")
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    print(f"  Source: {metadata.get('source', 'unknown').upper()}")
    
    results = {}
    
    # PLS-DA
    timer.start("PLS-DA Training")
    results['PLS-DA'] = train_pls_da(X_train, y_train, X_test, y_test)
    print(f"   Accuracy: {results['PLS-DA']:.4f}")
    timer.stop("PLS-DA Training")
    
    # SVM
    timer.start("SVM Training")
    results['SVM'] = train_svm(X_train, y_train, X_test, y_test)
    print(f"   Accuracy: {results['SVM']:.4f}")
    timer.stop("SVM Training")
    
    # LightweightCNN
    timer.start("LightweightCNN Training")
    train_loader = DataLoader(SpectralDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SpectralDataset(X_val, y_val), batch_size=BATCH_SIZE)
    test_loader = DataLoader(SpectralDataset(X_test, y_test), batch_size=BATCH_SIZE)
    
    model = LightweightCNN(input_length=N_WAVENUMBERS, n_classes=n_classes)
    print(f"   Parameters: {model.count_parameters():,}")
    
    model_path = os.path.join(RESULTS_DIR, 'models', 'wheat_cnn.pth')
    history = train_model(model, train_loader, val_loader, epochs=EPOCHS, 
                         learning_rate=0.001, patience=PATIENCE, device=device, save_path=model_path)
    
    model.eval()
    correct = sum((model.predict(x.to(device)).cpu() == y).sum().item() for x, y in test_loader)
    results['LightCNN'] = correct / len(y_test)
    print(f"   Accuracy: {results['LightCNN']:.4f}")
    timer.stop("LightweightCNN Training")
    
    # Total timing
    total_time = timer.summary()
    
    # Save results with provenance
    results_df = pd.DataFrame([{
        'Model': k, 
        'Accuracy': v,
        'data_source': metadata.get('source', 'unknown'),
        'dataset': 'wheat_lines',
        'n_samples': len(spectra),
        'n_classes': n_classes,
        'config_hash': config_hash,
        'timestamp': datetime.now().isoformat(),
        'total_time_seconds': total_time
    } for k, v in results.items()])
    
    results_df.to_csv(os.path.join(RESULTS_DIR, 'wheat_results.csv'), index=False)
    
    # Save metadata
    with open(os.path.join(RESULTS_DIR, 'provenance.json'), 'w') as f:
        json.dump({
            'data_metadata': metadata,
            'preprocessing_config': preprocessing_config,
            'config_hash': config_hash,
            'split_seed': SPLIT_SEED,
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'test_size': len(test_idx),
            'results': results,
            'timing': timer.times,
            'completed_at': datetime.now().isoformat()
        }, f, indent=2, default=str)
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n| Model | Accuracy |")
    print(f"|-------|----------|")
    for m, acc in results.items():
        print(f"| {m} | {acc:.4f} |")
    
    print(f"\nData source: {metadata.get('source', 'unknown').upper()}")
    print(f"Results saved to: {RESULTS_DIR}")
    
    return results, metadata


if __name__ == "__main__":
    main()
