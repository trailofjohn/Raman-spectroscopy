"""
Wheat Lines Classification with Full Modeling Rigor.

Features:
- Full seed control for reproducibility
- StandardScaler for SVM/PLS-DA consistency
- Classwise metrics (per-class precision/recall/F1)
- ResidualCNN with temperature scaling calibration
- Proper train/val/test split (test held out completely)
- Expected Calibration Error (ECE) computation
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
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             precision_score, recall_score, f1_score)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.pipeline import PreprocessingPipeline
from models.cnn_classifier import ResidualCNN, SpectralDataset
from data.data_loader import load_wheat_lines
from data.cache_manager import (compute_config_hash, get_cache_path, save_to_cache, 
                                 load_from_cache, cache_exists, get_preprocessing_config)
from data.exceptions import DataLoadError, CacheInvalidError
from utils.reproducibility import set_all_seeds, get_deterministic_dataloader_kwargs


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error.
    
    ECE measures how well-calibrated the model's confidence is.
    Lower is better; 0 = perfectly calibrated.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = predictions == labels
    
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if in_bin.sum() > 0:
            bin_accuracy = accuracies[in_bin].mean()
            bin_confidence = confidences[in_bin].mean()
            ece += np.abs(bin_accuracy - bin_confidence) * in_bin.mean()
    
    return ece


def train_pls_da(X_train, y_train, X_test, y_test, scaler, n_components=10):
    """PLS-DA with StandardScaler for consistency."""
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lb = LabelBinarizer()
    Y_train = lb.fit_transform(y_train)
    if Y_train.shape[1] == 1:
        Y_train = np.hstack([1 - Y_train, Y_train])
    
    n_comp = min(n_components, X_train.shape[1], X_train.shape[0] - 1)
    pls = PLSRegression(n_components=n_comp)
    pls.fit(X_train_scaled, Y_train)
    
    y_pred = pls.predict(X_test_scaled).argmax(axis=1)
    return y_pred


def train_svm(X_train, y_train, X_test, scaler, max_samples=5000, seed=42):
    """SVM with StandardScaler and fixed random state."""
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if len(X_train_scaled) > max_samples:
        np.random.seed(seed)
        idx = np.random.choice(len(X_train_scaled), max_samples, replace=False)
        X_train_scaled, y_train = X_train_scaled[idx], y_train[idx]
    
    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=seed)
    svm.fit(X_train_scaled, y_train)
    return svm.predict(X_test_scaled)


def train_cnn(model, train_loader, val_loader, device, epochs=50, patience=15, lr=0.001):
    """Train CNN with early stopping."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    
    model.to(device)
    best_val_acc = 0
    best_state = None
    no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x).argmax(dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += len(batch_y)
        
        val_acc = val_correct / val_total
        scheduler.step(1 - val_acc)
        
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_state)
    return model, best_val_acc


def main():
    # === REPRODUCIBILITY ===
    SEED = 42
    set_all_seeds(SEED)
    
    print("=" * 70)
    print("WHEAT CLASSIFICATION - FULL MODELING RIGOR")
    print("=" * 70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seed: {SEED}")
    
    # === CONFIG ===
    N_WAVENUMBERS = 1000
    BATCH_SIZE = 64
    EPOCHS = 50
    PATIENCE = 15
    
    RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'wheat_rigorous')
    os.makedirs(os.path.join(RESULTS_DIR, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'models'), exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # === LOAD DATA ===
    print("\n[1/6] Loading data...")
    pipeline = PreprocessingPipeline()
    preprocessing_config = get_preprocessing_config(pipeline)
    config_hash = compute_config_hash('wheat_lines', preprocessing_config, SEED, N_WAVENUMBERS)
    
    cache_path = get_cache_path('wheat_lines', config_hash)
    target_wn = np.linspace(400, 1800, N_WAVENUMBERS)
    
    if cache_exists('wheat_lines', config_hash):
        print("  Loading from cache...")
        (spectra, labels, wavenumbers, class_names, metadata,
         train_idx, val_idx, test_idx) = load_from_cache(cache_path)
    else:
        bundle = load_wheat_lines(allow_synthetic=False, target_wavenumbers=target_wn)
        spectra = bundle.spectra
        labels = bundle.labels
        class_names = bundle.class_names
        metadata = bundle.metadata
        
        print("  Preprocessing...")
        spectra = pipeline.process_batch(spectra)
        
        # Create deterministic splits
        X_train_full, X_test, y_train_full, y_test, idx_train_full, test_idx = train_test_split(
            spectra, labels, np.arange(len(labels)),
            test_size=0.2, random_state=SEED, stratify=labels
        )
        _, _, _, _, train_idx_rel, val_idx_rel = train_test_split(
            X_train_full, y_train_full, np.arange(len(y_train_full)),
            test_size=0.15, random_state=SEED, stratify=y_train_full
        )
        train_idx = idx_train_full[train_idx_rel]
        val_idx = idx_train_full[val_idx_rel]
        
        save_to_cache(cache_path, spectra, labels, target_wn, class_names, metadata,
                      train_idx, val_idx, test_idx)
    
    X_train, X_val, X_test = spectra[train_idx], spectra[val_idx], spectra[test_idx]
    y_train, y_val, y_test = labels[train_idx], labels[val_idx], labels[test_idx]
    n_classes = len(class_names)
    
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"  Classes: {n_classes} ({', '.join(class_names[:3])}...)")
    
    results = {}
    
    # === PLS-DA ===
    print("\n[2/6] PLS-DA (with StandardScaler)...")
    scaler_pls = StandardScaler()
    y_pred_pls = train_pls_da(X_train, y_train, X_test, y_test, scaler_pls)
    results['PLS-DA'] = {
        'y_pred': y_pred_pls,
        'accuracy': accuracy_score(y_test, y_pred_pls),
        'precision': precision_score(y_test, y_pred_pls, average='weighted'),
        'recall': recall_score(y_test, y_pred_pls, average='weighted'),
        'f1': f1_score(y_test, y_pred_pls, average='weighted')
    }
    print(f"  Accuracy: {results['PLS-DA']['accuracy']:.4f}, F1: {results['PLS-DA']['f1']:.4f}")
    
    # === SVM ===
    print("\n[3/6] SVM (with StandardScaler, fixed seed)...")
    scaler_svm = StandardScaler()
    y_pred_svm = train_svm(X_train, y_train, X_test, scaler_svm, seed=SEED)
    results['SVM'] = {
        'y_pred': y_pred_svm,
        'accuracy': accuracy_score(y_test, y_pred_svm),
        'precision': precision_score(y_test, y_pred_svm, average='weighted'),
        'recall': recall_score(y_test, y_pred_svm, average='weighted'),
        'f1': f1_score(y_test, y_pred_svm, average='weighted')
    }
    print(f"  Accuracy: {results['SVM']['accuracy']:.4f}, F1: {results['SVM']['f1']:.4f}")
    
    # === ResidualCNN ===
    print("\n[4/6] ResidualCNN (with calibration)...")
    dl_kwargs = get_deterministic_dataloader_kwargs()
    train_loader = DataLoader(SpectralDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(SpectralDataset(X_val, y_val), batch_size=BATCH_SIZE, **dl_kwargs)
    test_loader = DataLoader(SpectralDataset(X_test, y_test), batch_size=BATCH_SIZE, **dl_kwargs)
    
    model = ResidualCNN(input_length=N_WAVENUMBERS, n_classes=n_classes, dropout_rate=0.2)
    print(f"  Parameters: {model.count_parameters():,}")
    
    model, best_val_acc = train_cnn(model, train_loader, val_loader, device, epochs=EPOCHS, patience=PATIENCE)
    print(f"  Best validation accuracy: {best_val_acc:.4f}")
    
    # Calibrate temperature
    print("  Calibrating temperature...")
    model.calibrate_temperature(val_loader, device)
    
    # Evaluate on test set
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            probs = model.predict_proba(batch_x, calibrated=True).cpu().numpy()
            all_probs.append(probs)
            all_preds.extend(probs.argmax(axis=1))
            all_labels.extend(batch_y.numpy())
    
    y_pred_cnn = np.array(all_preds)
    probs_cnn = np.vstack(all_probs)
    ece = compute_ece(probs_cnn, np.array(all_labels))
    
    results['ResidualCNN'] = {
        'y_pred': y_pred_cnn,
        'accuracy': accuracy_score(y_test, y_pred_cnn),
        'precision': precision_score(y_test, y_pred_cnn, average='weighted'),
        'recall': recall_score(y_test, y_pred_cnn, average='weighted'),
        'f1': f1_score(y_test, y_pred_cnn, average='weighted'),
        'ece': ece,
        'temperature': model.temperature.item()
    }
    print(f"  Accuracy: {results['ResidualCNN']['accuracy']:.4f}, F1: {results['ResidualCNN']['f1']:.4f}, ECE: {ece:.4f}")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'temperature': model.temperature.item(),
        'class_names': class_names,
        'seed': SEED
    }, os.path.join(RESULTS_DIR, 'models', 'residual_cnn.pth'))
    
    # === CLASSWISE METRICS ===
    print("\n[5/6] Generating classwise metrics...")
    for name, res in results.items():
        report = classification_report(y_test, res['y_pred'], target_names=class_names, output_dict=True)
        res['classwise'] = {class_names[i]: {
            'precision': report[class_names[i]]['precision'],
            'recall': report[class_names[i]]['recall'],
            'f1': report[class_names[i]]['f1-score']
        } for i in range(n_classes)}
    
    # Confusion matrix for best model
    cm = confusion_matrix(y_test, results['ResidualCNN']['y_pred'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('ResidualCNN Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'plots', 'confusion_matrix.png'), dpi=150)
    plt.close()
    
    # === SAVE RESULTS ===
    print("\n[6/6] Saving results...")
    
    # Summary table
    summary_df = pd.DataFrame([{
        'Model': name,
        'Accuracy': res['accuracy'],
        'Precision': res['precision'],
        'Recall': res['recall'],
        'F1': res['f1'],
        'ECE': res.get('ece', None),
        'Temperature': res.get('temperature', None)
    } for name, res in results.items()])
    summary_df.to_csv(os.path.join(RESULTS_DIR, 'results_summary.csv'), index=False)
    
    # Full provenance
    provenance = {
        'seed': SEED,
        'data_source': metadata.get('source', 'unknown'),
        'n_samples': {'train': len(X_train), 'val': len(X_val), 'test': len(X_test)},
        'class_names': class_names,
        'results': {name: {k: v if not isinstance(v, np.ndarray) else v.tolist() 
                          for k, v in res.items()} for name, res in results.items()},
        'timestamp': datetime.now().isoformat()
    }
    with open(os.path.join(RESULTS_DIR, 'provenance.json'), 'w') as f:
        json.dump(provenance, f, indent=2)
    
    # === SUMMARY ===
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<15} {'Accuracy':>10} {'F1':>10} {'ECE':>10}")
    print("-" * 50)
    for name, res in results.items():
        ece_str = f"{res.get('ece', 0):.4f}" if 'ece' in res else "N/A"
        print(f"{name:<15} {res['accuracy']:>10.4f} {res['f1']:>10.4f} {ece_str:>10}")
    
    print(f"\nClasswise F1 (ResidualCNN):")
    for cls, metrics in results['ResidualCNN']['classwise'].items():
        print(f"  {cls}: {metrics['f1']:.4f}")
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    
    return results


if __name__ == "__main__":
    main()
