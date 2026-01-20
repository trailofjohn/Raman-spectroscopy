"""
Uncertainty Quantification with MC Dropout (Fixed Version).

Features:
- Reuses trained wheat CNN weights for comparability
- Hard fail on data load
- Proper calibration metrics
- Provenance tracking
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.pipeline import PreprocessingPipeline
from models.cnn_classifier import LightweightCNN, SpectralDataset
from data.data_loader import load_wheat_lines
from data.exceptions import DataLoadError


class UncertaintyCNN(LightweightCNN):
    """
    LightweightCNN with MC Dropout for uncertainty estimation.
    Can optionally load pretrained weights from wheat classifier.
    """
    
    def predict_with_uncertainty(self, x, n_samples=50):
        """
        Monte Carlo Dropout prediction.
        Keeps dropout ON during inference for uncertainty estimation.
        """
        self.train()  # Keep dropout active
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.forward(x)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs.cpu().numpy())
        
        predictions = np.array(predictions)  # (n_samples, batch, n_classes)
        
        mean_probs = predictions.mean(axis=0)
        std_probs = predictions.std(axis=0)
        
        predicted_class = mean_probs.argmax(axis=1)
        confidence = mean_probs.max(axis=1)
        uncertainty = std_probs.max(axis=1)
        
        return predicted_class, confidence, uncertainty


def load_pretrained_weights(model, weights_path, device):
    """Load pretrained weights from wheat classifier if available."""
    if os.path.exists(weights_path):
        print(f"  Loading pretrained weights from: {weights_path}")
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return True
    return False


def main():
    print("=" * 60)
    print("UNCERTAINTY QUANTIFICATION (FIXED)")
    print("MC Dropout with Pretrained Weights")
    print("=" * 60)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    N_WAVENUMBERS = 1000
    MC_SAMPLES = 50
    SPLIT_SEED = 42
    
    RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'uncertainty')
    os.makedirs(os.path.join(RESULTS_DIR, 'models'), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'plots'), exist_ok=True)
    
    # Check for pretrained wheat model
    WHEAT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                     'results', 'wheat', 'models', 'wheat_cnn.pth')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load data - HARD FAIL
    print("\n[1/4] Loading data...")
    target_wn = np.linspace(400, 1800, N_WAVENUMBERS)
    
    try:
        bundle = load_wheat_lines(allow_synthetic=False, target_wavenumbers=target_wn)
    except DataLoadError as e:
        print(f"\n❌ FATAL: {e}")
        return None
    
    spectra = bundle.spectra
    labels = bundle.labels
    metadata = bundle.metadata
    n_classes = len(bundle.class_names)
    
    print(f"  Loaded: {len(spectra)} samples, {n_classes} classes")
    print(f"  Source: {metadata.get('source', 'unknown').upper()}")
    
    # Preprocess
    print("\n[2/4] Preprocessing...")
    pipeline = PreprocessingPipeline()
    spectra = pipeline.process_batch(spectra)
    
    # Use SAME splits as wheat training for fair comparison
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        spectra, labels, test_size=0.2, random_state=SPLIT_SEED, stratify=labels
    )
    
    test_loader = DataLoader(SpectralDataset(X_test, y_test), batch_size=64)
    
    # Create model and try to load pretrained weights
    print("\n[3/4] Setting up model...")
    model = UncertaintyCNN(input_length=N_WAVENUMBERS, n_classes=n_classes)
    
    used_pretrained = load_pretrained_weights(model, WHEAT_MODEL_PATH, device)
    
    if not used_pretrained:
        print("  ⚠️  No pretrained weights found, training from scratch...")
        # Quick training if no pretrained weights
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.15, random_state=SPLIT_SEED, stratify=y_train_full
        )
        train_loader = DataLoader(SpectralDataset(X_train, y_train), batch_size=64, shuffle=True)
        val_loader = DataLoader(SpectralDataset(X_val, y_val), batch_size=64)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model.to(device)
        
        for epoch in range(30):
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
    
    model.to(device)
    
    # Evaluate with MC Dropout
    print(f"\n[4/4] Evaluating with MC Dropout ({MC_SAMPLES} samples)...")
    all_preds, all_true, all_conf, all_uncert = [], [], [], []
    
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        pred_class, confidence, uncertainty = model.predict_with_uncertainty(batch_x, MC_SAMPLES)
        all_preds.extend(pred_class)
        all_true.extend(batch_y.numpy())
        all_conf.extend(confidence)
        all_uncert.extend(uncertainty)
    
    y_pred = np.array(all_preds)
    y_true = np.array(all_true)
    confidence = np.array(all_conf)
    uncertainty = np.array(all_uncert)
    
    correct_mask = y_pred == y_true
    acc = accuracy_score(y_true, y_pred)
    
    # Calibration metrics
    print(f"\n  Accuracy: {acc:.4f}")
    print(f"  Mean Confidence: {confidence.mean():.4f} ± {confidence.std():.4f}")
    print(f"  Mean Uncertainty: {uncertainty.mean():.4f} ± {uncertainty.std():.4f}")
    print(f"\n  Calibration Analysis:")
    print(f"    Correct predictions uncertainty: {uncertainty[correct_mask].mean():.4f}")
    print(f"    Wrong predictions uncertainty: {uncertainty[~correct_mask].mean():.4f}")
    
    # Uncertainty should be HIGHER for wrong predictions
    calibration_ok = uncertainty[~correct_mask].mean() > uncertainty[correct_mask].mean()
    print(f"    Calibration check: {'✓ PASS' if calibration_ok else '✗ FAIL'}")
    
    # Save plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(confidence[correct_mask], bins=30, alpha=0.7, label='Correct', color='green')
    if (~correct_mask).sum() > 0:
        axes[0].hist(confidence[~correct_mask], bins=30, alpha=0.7, label='Wrong', color='red')
    axes[0].set_xlabel('Confidence')
    axes[0].set_ylabel('Count')
    axes[0].legend()
    axes[0].set_title('Confidence Distribution')
    
    axes[1].hist(uncertainty[correct_mask], bins=30, alpha=0.7, label='Correct', color='green')
    if (~correct_mask).sum() > 0:
        axes[1].hist(uncertainty[~correct_mask], bins=30, alpha=0.7, label='Wrong', color='red')
    axes[1].set_xlabel('Uncertainty')
    axes[1].set_ylabel('Count')
    axes[1].legend()
    axes[1].set_title('Uncertainty Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'plots', 'uncertainty_analysis.png'), dpi=150)
    plt.close()
    
    # Save results with provenance
    results = {
        'accuracy': float(acc),
        'mean_confidence': float(confidence.mean()),
        'std_confidence': float(confidence.std()),
        'mean_uncertainty': float(uncertainty.mean()),
        'correct_uncertainty': float(uncertainty[correct_mask].mean()),
        'wrong_uncertainty': float(uncertainty[~correct_mask].mean()) if (~correct_mask).sum() > 0 else None,
        'calibration_passed': bool(calibration_ok),
        'mc_samples': MC_SAMPLES,
        'used_pretrained_weights': used_pretrained,
        'data_source': metadata.get('source', 'unknown'),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(RESULTS_DIR, 'uncertainty_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    pd.DataFrame([results]).to_csv(os.path.join(RESULTS_DIR, 'uncertainty_results.csv'), index=False)
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {RESULTS_DIR}")
    
    return results


if __name__ == "__main__":
    main()
