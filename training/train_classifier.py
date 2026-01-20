"""
Training Script for 1D-CNN Classification Model.
Uses RamanSPy datasets with synthetic fallback.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.pipeline import PreprocessingPipeline
from models.cnn_classifier import SpectralCNN, SpectralDataset, train_model, load_model
from data.data_loader import load_all_datasets, prepare_classification_data, generate_synthetic_ramanspy


def plot_confusion_matrix(y_true, y_pred, class_names, save_path, normalize=True):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        cm = np.divide(cm.astype('float'), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums!=0)
    
    n_classes = len(class_names)
    figsize = max(8, min(20, n_classes * 0.4))
    
    fig, ax = plt.subplots(figsize=(figsize, figsize * 0.8))
    sns.heatmap(cm, annot=n_classes < 15, fmt='.2f' if normalize else 'd',
                xticklabels=class_names, yticklabels=class_names, cmap='Blues', ax=ax)
    plt.xticks(rotation=45, ha='right', fontsize=max(6, 10 - n_classes//5))
    plt.yticks(fontsize=max(6, 10 - n_classes//5))
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_training_curves(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Val')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    print("=" * 60)
    print("1D-CNN CLASSIFICATION MODEL TRAINING")
    print("=" * 60)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    N_WAVENUMBERS = 1000
    BATCH_SIZE = 32
    EPOCHS = 100
    LR = 0.001
    PATIENCE = 15
    RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    os.makedirs(os.path.join(RESULTS_DIR, 'models'), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'plots'), exist_ok=True)
    
    # Load data
    print("\n[1/6] Loading data...")
    try:
        datasets = load_all_datasets()
        if datasets:
            spectra, labels, class_names = prepare_classification_data(datasets, target_wavenumbers=N_WAVENUMBERS)
        else:
            raise ValueError("No datasets loaded")
    except Exception as e:
        print(f"Real data failed ({e}), using synthetic...")
        spectra, labels, class_names = generate_synthetic_ramanspy(n_classes=6, samples_per_class=200, n_wavenumbers=N_WAVENUMBERS)
    
    N_CLASSES = len(class_names)
    print(f"  Total: {len(spectra)} samples, {N_CLASSES} classes")
    
    # Preprocess
    print("\n[2/6] Preprocessing...")
    pipeline = PreprocessingPipeline(use_cosmic_ray_removal=True, use_denoising=True, 
                                     use_baseline_correction=True, use_normalization=True)
    spectra = pipeline.process_batch(spectra)
    
    # Filter small classes
    unique, counts = np.unique(labels, return_counts=True)
    valid_classes = unique[counts >= 3]
    if len(valid_classes) < len(unique):
        print(f"  Filtering {len(unique) - len(valid_classes)} small classes...")
        mask = np.isin(labels, valid_classes)
        spectra, labels = spectra[mask], labels[mask]
        label_map = {old: new for new, old in enumerate(valid_classes)}
        labels = np.array([label_map[l] for l in labels])
        class_names = [class_names[i] for i in valid_classes]
        N_CLASSES = len(class_names)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(spectra, labels, test_size=0.2, random_state=42, stratify=labels)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42, stratify=y_train)
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    train_loader = DataLoader(SpectralDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SpectralDataset(X_val, y_val), batch_size=BATCH_SIZE)
    test_loader = DataLoader(SpectralDataset(X_test, y_test), batch_size=BATCH_SIZE)
    
    # Train
    print(f"\n[3/6] Model: {N_WAVENUMBERS} inputs, {N_CLASSES} classes")
    model = SpectralCNN(input_length=N_WAVENUMBERS, n_classes=N_CLASSES, dropout_rate=0.5)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n[4/6] Training...")
    model_path = os.path.join(RESULTS_DIR, 'models', 'cnn_classifier.pth')
    history = train_model(model, train_loader, val_loader, epochs=EPOCHS, learning_rate=LR, 
                         patience=PATIENCE, device=device, save_path=model_path)
    plot_training_curves(history, os.path.join(RESULTS_DIR, 'plots', 'training_curves.png'))
    
    # Evaluate
    print("\n[5/6] Evaluating...")
    model = load_model(model_path, N_WAVENUMBERS, N_CLASSES, device)
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            preds = model.predict(batch_x.to(device))
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    
    y_pred, y_true = np.array(all_preds), np.array(all_labels)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"  Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    
    print("\n[6/6] Saving results...")
    plot_confusion_matrix(y_true, y_pred, class_names, os.path.join(RESULTS_DIR, 'plots', 'confusion_matrix.png'))
    
    results = pd.DataFrame([{'Model': '1D-CNN', 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 
                            'F1_Score': f1, 'Epochs': len(history['train_loss']), 'Classes': N_CLASSES}])
    results.to_csv(os.path.join(RESULTS_DIR, 'classification_results.csv'), index=False)
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(results.to_string(index=False))
    return results, history


if __name__ == "__main__":
    main()
