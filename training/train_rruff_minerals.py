"""
RRUFF Mineral Classification Training (Fixed Version).

Features:
- Hard fail on data load (no silent synthetic fallback)
- Proper RRUFF API handling
- Provenance metadata in all outputs
- LightweightCNN with baseline comparison
"""

import sys
import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, confusion_matrix

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.pipeline import PreprocessingPipeline
from models.cnn_classifier import LightweightCNN, SpectralDataset, train_model
from data.data_loader import load_rruff, DataBundle
from data.exceptions import DataLoadError


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
        print(f"   ✓ {elapsed:.1f}s")
    
    def summary(self):
        total = time.time() - self.total_start
        print("\n" + "=" * 50)
        print("TIMING SUMMARY")
        for name, t in self.times.items():
            print(f"  {name}: {t:.1f}s")
        print(f"  TOTAL: {total:.1f}s ({total/60:.2f} min)")
        return total


def train_pls_da(X_train, y_train, X_test, y_test, n_components=10):
    """PLS-DA baseline."""
    lb = LabelBinarizer()
    Y_train = lb.fit_transform(y_train)
    if Y_train.shape[1] == 1:
        Y_train = np.hstack([1 - Y_train, Y_train])
    
    n_comp = min(n_components, X_train.shape[1], X_train.shape[0] - 1)
    pls = PLSRegression(n_components=n_comp)
    pls.fit(X_train, Y_train)
    y_pred = pls.predict(X_test).argmax(axis=1)
    return accuracy_score(y_test, y_pred)


def main():
    print("=" * 60)
    print("RRUFF MINERAL CLASSIFICATION (FIXED)")
    print("For Lunar/Planetary Regolith Analysis")
    print("=" * 60)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    timer = Timer()
    
    N_WAVENUMBERS = 1000
    SPLIT_SEED = 42
    
    RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'rruff')
    os.makedirs(os.path.join(RESULTS_DIR, 'models'), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'plots'), exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load RRUFF - HARD FAIL if unavailable
    timer.start("Loading RRUFF")
    target_wn = np.linspace(400, 1800, N_WAVENUMBERS)
    
    try:
        bundle = load_rruff(
            dataset='excellent_unoriented',
            allow_synthetic=False,  # HARD FAIL
            target_wavenumbers=target_wn
        )
    except DataLoadError as e:
        print(f"\n❌ FATAL: {e}")
        print("Cannot proceed without real RRUFF data.")
        print("This experiment is invalid without mineral spectra.")
        return None
    
    spectra = bundle.spectra
    labels = bundle.labels
    wavenumbers = bundle.wavenumbers
    class_names = bundle.class_names
    metadata = bundle.metadata
    timer.stop("Loading RRUFF")
    
    # Preprocess
    timer.start("Preprocessing")
    pipeline = PreprocessingPipeline()
    spectra = pipeline.process_batch(spectra)
    timer.stop("Preprocessing")
    
    # Filter classes with too few samples
    unique, counts = np.unique(labels, return_counts=True)
    valid_classes = unique[counts >= 3]
    if len(valid_classes) < len(unique):
        mask = np.isin(labels, valid_classes)
        spectra, labels = spectra[mask], labels[mask]
        label_map = {old: new for new, old in enumerate(valid_classes)}
        labels = np.array([label_map[l] for l in labels])
        class_names = [class_names[i] for i in valid_classes]
        print(f"  Filtered to {len(valid_classes)} classes with ≥3 samples")
    
    n_classes = len(class_names)
    print(f"\n  Data: {len(spectra)} samples, {n_classes} mineral classes")
    print(f"  Source: {metadata.get('source', 'unknown').upper()}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        spectra, labels, test_size=0.2, random_state=SPLIT_SEED, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=SPLIT_SEED, stratify=y_train
    )
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    results = {}
    
    # PLS-DA baseline
    timer.start("PLS-DA Training")
    results['PLS-DA'] = train_pls_da(X_train, y_train, X_test, y_test)
    print(f"   Accuracy: {results['PLS-DA']:.4f}")
    timer.stop("PLS-DA Training")
    
    # LightweightCNN
    timer.start("LightweightCNN Training")
    train_loader = DataLoader(SpectralDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(SpectralDataset(X_val, y_val), batch_size=32)
    test_loader = DataLoader(SpectralDataset(X_test, y_test), batch_size=32)
    
    model = LightweightCNN(input_length=N_WAVENUMBERS, n_classes=n_classes)
    print(f"   Parameters: {model.count_parameters():,}")
    
    model_path = os.path.join(RESULTS_DIR, 'models', 'mineral_cnn.pth')
    history = train_model(model, train_loader, val_loader, epochs=100, 
                         learning_rate=0.001, patience=15, device=device, save_path=model_path)
    
    model.eval()
    correct = sum((model.predict(x.to(device)).cpu() == y).sum().item() for x, y in test_loader)
    results['LightCNN'] = correct / len(y_test)
    print(f"   Accuracy: {results['LightCNN']:.4f}")
    timer.stop("LightweightCNN Training")
    
    total_time = timer.summary()
    
    # Save results with provenance
    results_df = pd.DataFrame([{
        'Model': k,
        'Accuracy': v,
        'data_source': metadata.get('source', 'unknown'),
        'dataset': 'rruff_excellent_unoriented',
        'n_samples': len(spectra),
        'n_classes': n_classes,
        'timestamp': datetime.now().isoformat()
    } for k, v in results.items()])
    
    results_df.to_csv(os.path.join(RESULTS_DIR, 'mineral_results.csv'), index=False)
    
    # Save provenance
    with open(os.path.join(RESULTS_DIR, 'provenance.json'), 'w') as f:
        json.dump({
            'data_metadata': metadata,
            'class_names': class_names,
            'split_seed': SPLIT_SEED,
            'results': results,
            'timing': timer.times,
            'completed_at': datetime.now().isoformat()
        }, f, indent=2, default=str)
    
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
