"""
Transfer Learning for Raman Spectroscopy.

Strategy:
1. Pre-train CNN on large RRUFF mineral database (general spectral features)
2. Fine-tune on specific target dataset (domain-specific)

Useful for when you have limited data for your specific minerals.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.pipeline import PreprocessingPipeline
from models.cnn_classifier import SpectralCNN, SpectralDataset, EarlyStopping
from data.data_loader import load_rruff, load_wheat_lines, generate_synthetic_custom


def pretrain_on_rruff(model, device, n_wavenumbers=1000, epochs=50, lr=0.001):
    """Pre-train on RRUFF mineral database to learn general spectral features."""
    print("\n--- Phase 1: Pre-training on RRUFF minerals ---")
    
    try:
        spectra, labels, wavenumbers, class_names = load_rruff('excellent_unoriented')
        if len(spectra) == 0:
            raise ValueError("No RRUFF data")
        
        # Resample
        from scipy.interpolate import interp1d
        if spectra.shape[1] != n_wavenumbers:
            x_old = np.linspace(0, 1, spectra.shape[1])
            x_new = np.linspace(0, 1, n_wavenumbers)
            resampled = np.zeros((spectra.shape[0], n_wavenumbers))
            for i in range(spectra.shape[0]):
                f = interp1d(x_old, spectra[i], kind='linear', fill_value='extrapolate')
                resampled[i] = f(x_new)
            spectra = resampled
    except Exception as e:
        print(f"  RRUFF failed ({e}), using synthetic pre-training data")
        spectra, labels, _ = generate_synthetic_custom(n_classes=20, samples_per_class=50, n_wavenumbers=n_wavenumbers)
    
    # Preprocess
    pipeline = PreprocessingPipeline()
    spectra = pipeline.process_batch(spectra)
    
    # Filter small classes
    unique, counts = np.unique(labels, return_counts=True)
    valid = unique[counts >= 3]
    mask = np.isin(labels, valid)
    spectra, labels = spectra[mask], labels[mask]
    label_map = {old: new for new, old in enumerate(np.unique(labels))}
    labels = np.array([label_map[l] for l in labels])
    
    n_classes = len(np.unique(labels))
    print(f"  Pre-training data: {len(spectra)} samples, {n_classes} classes")
    
    # Create new classifier head for pre-training
    model.fc2 = nn.Linear(256, n_classes).to(device)
    
    X_train, X_val, y_train, y_val = train_test_split(spectra, labels, test_size=0.2, random_state=42, stratify=labels)
    train_loader = DataLoader(SpectralDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(SpectralDataset(X_val, y_val), batch_size=32)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=10)
    
    model.to(device)
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                val_loss += criterion(model(batch_x.to(device)), batch_y.to(device)).item()
        val_loss /= len(val_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f"    Pre-train Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}")
        
        if early_stopping(val_loss):
            print(f"    Early stopping at epoch {epoch+1}")
            break
    
    print("  Pre-training complete. Feature extractor trained.")
    return model


def finetune_on_target(model, target_spectra, target_labels, device, n_classes, epochs=50, lr=0.0001):
    """Fine-tune on target dataset with frozen feature extractor."""
    print("\n--- Phase 2: Fine-tuning on target data ---")
    
    # Freeze convolutional layers (feature extractor)
    for name, param in model.named_parameters():
        if 'conv' in name or 'bn' in name:
            param.requires_grad = False
    
    # Replace classifier head for target classes
    model.fc2 = nn.Linear(256, n_classes).to(device)
    
    X_train, X_test, y_train, y_test = train_test_split(target_spectra, target_labels, test_size=0.2, random_state=42, stratify=target_labels)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42, stratify=y_train)
    
    print(f"  Fine-tuning data: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    train_loader = DataLoader(SpectralDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(SpectralDataset(X_val, y_val), batch_size=32)
    test_loader = DataLoader(SpectralDataset(X_test, y_test), batch_size=32)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    early_stopping = EarlyStopping(patience=10)
    
    best_val_acc = 0
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                preds = model.predict(batch_x.to(device))
                val_correct += (preds.cpu() == batch_y).sum().item()
                val_total += len(batch_y)
        val_acc = val_correct / val_total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        if (epoch + 1) % 10 == 0:
            print(f"    Fine-tune Epoch {epoch+1}/{epochs}, Val Acc: {val_acc:.4f}")
    
    # Final test evaluation
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            preds = model.predict(batch_x.to(device))
            test_correct += (preds.cpu() == batch_y).sum().item()
            test_total += len(batch_y)
    test_acc = test_correct / test_total
    
    print(f"  Fine-tuning complete. Test Accuracy: {test_acc:.4f}")
    return model, test_acc


def main():
    print("=" * 60)
    print("TRANSFER LEARNING FOR RAMAN SPECTROSCOPY")
    print("=" * 60)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    N_WAVENUMBERS = 1000
    RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'transfer_learning')
    os.makedirs(os.path.join(RESULTS_DIR, 'models'), exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Initialize model
    model = SpectralCNN(input_length=N_WAVENUMBERS, n_classes=10, dropout_rate=0.5)
    
    # Phase 1: Pre-train on RRUFF
    model = pretrain_on_rruff(model, device, N_WAVENUMBERS, epochs=30, lr=0.001)
    
    # Phase 2: Fine-tune on target (using synthetic as example - replace with your data)
    print("\nPreparing target dataset (synthetic lunar minerals)...")
    target_spectra, target_labels, target_names = generate_synthetic_custom(
        n_classes=5, samples_per_class=50, n_wavenumbers=N_WAVENUMBERS, seed=123
    )
    pipeline = PreprocessingPipeline()
    target_spectra = pipeline.process_batch(target_spectra)
    
    model, test_acc = finetune_on_target(model, target_spectra, target_labels, device, n_classes=5, epochs=30, lr=0.0001)
    
    # Save model
    model_path = os.path.join(RESULTS_DIR, 'models', 'transfer_learning_model.pth')
    torch.save({'model_state_dict': model.state_dict()}, model_path)
    
    print("\n" + "=" * 60)
    print("TRANSFER LEARNING COMPLETE")
    print("=" * 60)
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
