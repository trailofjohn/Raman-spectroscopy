"""
RRUFF Mineral Classification using local files.

Uses downloaded RRUFF data from /home/john/projects/raman_spectroscopy_pipeline/rruff/
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.pipeline import PreprocessingPipeline
from models.cnn_classifier import ResidualCNN, SpectralDataset
from data.rruff_local_loader import load_rruff_local
from data.exceptions import DataLoadError
from utils.reproducibility import set_all_seeds, get_deterministic_dataloader_kwargs


def train_pls_da(X_train, y_train, X_test, scaler, n_components=10):
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lb = LabelBinarizer()
    Y_train = lb.fit_transform(y_train)
    if Y_train.shape[1] == 1:
        Y_train = np.hstack([1 - Y_train, Y_train])
    
    n_comp = min(n_components, X_train.shape[1], X_train.shape[0] - 1)
    pls = PLSRegression(n_components=n_comp)
    pls.fit(X_train_scaled, Y_train)
    return pls.predict(X_test_scaled).argmax(axis=1)


def train_svm(X_train, y_train, X_test, scaler, seed=42):
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=seed)
    svm.fit(X_train_scaled, y_train)
    return svm.predict(X_test_scaled)


def main():
    SEED = 42
    set_all_seeds(SEED)
    
    print("=" * 70)
    print("RRUFF MINERAL CLASSIFICATION (LOCAL FILES)")
    print("For Lunar/Planetary Regolith Analysis")
    print("=" * 70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'rruff_local')
    os.makedirs(os.path.join(RESULTS_DIR, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'models'), exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load from local files
    print("\n[1/5] Loading local RRUFF data...")
    try:
        spectra, labels, wavenumbers, class_names, metadata = load_rruff_local(
            subset='excellent_unoriented',
            processed_only=True
        )
    except DataLoadError as e:
        print(f"❌ {e}")
        return None
    
    n_classes = len(class_names)
    print(f"  {len(spectra)} samples, {n_classes} mineral classes")
    
    # Preprocess
    print("\n[2/5] Preprocessing...")
    pipeline = PreprocessingPipeline()
    spectra = pipeline.process_batch(spectra)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        spectra, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=SEED, stratify=y_train
    )
    
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    results = {}
    
    # PLS-DA
    print("\n[3/5] PLS-DA...")
    scaler_pls = StandardScaler()
    y_pred_pls = train_pls_da(X_train, y_train, X_test, scaler_pls)
    results['PLS-DA'] = accuracy_score(y_test, y_pred_pls)
    print(f"  Accuracy: {results['PLS-DA']:.4f}")
    
    # SVM
    print("\n[4/5] SVM...")
    scaler_svm = StandardScaler()
    y_pred_svm = train_svm(X_train, y_train, X_test, scaler_svm)
    results['SVM'] = accuracy_score(y_test, y_pred_svm)
    print(f"  Accuracy: {results['SVM']:.4f}")
    
    # ResidualCNN
    print("\n[5/5] ResidualCNN...")
    dl_kwargs = get_deterministic_dataloader_kwargs()
    train_loader = DataLoader(SpectralDataset(X_train, y_train), batch_size=32, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(SpectralDataset(X_val, y_val), batch_size=32, **dl_kwargs)
    test_loader = DataLoader(SpectralDataset(X_test, y_test), batch_size=32, **dl_kwargs)
    
    model = ResidualCNN(input_length=len(wavenumbers), n_classes=n_classes, dropout_rate=0.3)
    print(f"  Parameters: {model.count_parameters():,}")
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    model.to(device)
    
    best_val_acc = 0
    patience = 20
    no_improve = 0
    
    for epoch in range(100):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        correct = sum((model.predict(x.to(device)).cpu() == y).sum().item() for x, y in val_loader)
        val_acc = correct / len(y_val)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}, Val Acc: {val_acc:.4f}")
        
        if no_improve >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_state)
    model.eval()
    correct = sum((model.predict(x.to(device)).cpu() == y).sum().item() for x, y in test_loader)
    results['ResidualCNN'] = correct / len(y_test)
    print(f"  Test Accuracy: {results['ResidualCNN']:.4f}")
    
    # Confusion matrix
    all_preds, all_labels = [], []
    for x, y in test_loader:
        all_preds.extend(model.predict(x.to(device)).cpu().numpy())
        all_labels.extend(y.numpy())
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'n_classes': n_classes
    }, os.path.join(RESULTS_DIR, 'models', 'mineral_cnn.pth'))
    
    # Save results
    results_df = pd.DataFrame([{
        'Model': k,
        'Accuracy': v,
        'data_source': 'real_local',
        'n_samples': len(spectra),
        'n_classes': n_classes
    } for k, v in results.items()])
    results_df.to_csv(os.path.join(RESULTS_DIR, 'results.csv'), index=False)
    
    with open(os.path.join(RESULTS_DIR, 'provenance.json'), 'w') as f:
        json.dump({
            'metadata': metadata,
            'class_names': class_names,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n{'Model':<15} {'Accuracy':>10}")
    print("-" * 30)
    for m, acc in results.items():
        print(f"{m:<15} {acc:>10.4f}")
    
    print(f"\nData source: LOCAL REAL ({metadata['loaded_from']})")
    print(f"Results saved to: {RESULTS_DIR}")
    
    return results


if __name__ == "__main__":
    main()
