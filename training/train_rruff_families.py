"""
RRUFF Mineral Family Classification (Scientifically Valid Approach).

Instead of 1415-class identification (invalid with ~3.7 samples/class),
this maps minerals to ~15 chemical families based on anion groups.

Families (from Nickel-Strunz classification):
- Silicates (SiO4) - largest group
- Oxides (O2-)
- Carbonates (CO3)
- Sulfates (SO4)
- Phosphates (PO4)
- Sulfides (S2-)
- Halides (Cl, F, Br)
- Hydroxides (OH)
- Borates (BO3)
- Native Elements
- Organic
- Other
"""

import sys
import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from collections import Counter

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.pipeline import PreprocessingPipeline
from models.cnn_classifier import ResidualCNN, SpectralDataset
from data.exceptions import DataLoadError
from utils.reproducibility import set_all_seeds, get_deterministic_dataloader_kwargs


# Mineral family classification based on dominant anion/chemistry
def classify_mineral_family(ideal_chemistry: str) -> str:
    """
    Classify mineral into chemical family based on IDEAL CHEMISTRY string.
    Uses pattern matching on common anion groups.
    """
    if not ideal_chemistry:
        return 'Unknown'
    
    chem = ideal_chemistry.upper()
    
    # Order matters - check more specific patterns first
    
    # Silicates (contain SiO4 or Si_x_O_y patterns)
    if 'SIO4' in chem or 'SI2O' in chem or 'SI3O' in chem or 'SI4O' in chem or 'SIO3' in chem:
        return 'Silicate'
    if '(SI' in chem and 'O' in chem:
        return 'Silicate'
    
    # Carbonates (CO3)
    if 'CO3' in chem or '(CO_3' in chem:
        return 'Carbonate'
    
    # Sulfates (SO4)
    if 'SO4' in chem or '(SO_4' in chem:
        return 'Sulfate'
    
    # Phosphates (PO4), Arsenates (AsO4), Vanadates (VO4)
    if 'PO4' in chem or '(PO_4' in chem:
        return 'Phosphate'
    if 'ASO4' in chem or 'VO4' in chem:
        return 'Phosphate'  # Group together
    
    # Borates (BO3, BO4)
    if 'BO3' in chem or 'BO4' in chem or 'B2O' in chem:
        return 'Borate'
    
    # Sulfides and Sulfosalts (contain S but not SO4)
    if 'S_' in chem or 'S2' in chem or re.search(r'[A-Z]S[^O]', chem):
        if 'SO4' not in chem and 'SO_4' not in chem:
            return 'Sulfide'
    
    # Halides (primarily F, Cl, Br, I)
    if 'CL' in chem or 'F_' in chem or 'BR' in chem:
        if 'O' not in chem:  # Pure halides
            return 'Halide'
    
    # Oxides (metal + oxygen, no complex anions)
    if 'O' in chem and not any(x in chem for x in ['CO3', 'SO4', 'PO4', 'SIO', 'BO']):
        # Check if it's a simple oxide pattern
        if re.search(r'O[_\d]', chem) or chem.endswith('O'):
            # Could be oxide or hydroxide
            if 'OH' in chem or '(OH)' in chem:
                return 'Hydroxide'
            return 'Oxide'
    
    # Native elements
    if re.match(r'^[A-Z][a-z]?$', ideal_chemistry.strip()):
        return 'Native Element'
    
    # Hydroxides
    if 'OH' in chem or '(OH)' in chem:
        return 'Hydroxide'
    
    return 'Other'


def load_rruff_with_families(
    rruff_dir: str = '/home/john/projects/raman_spectroscopy_pipeline/rruff',
    subset: str = 'excellent_unoriented',
    min_samples_per_family: int = 20
):
    """
    Load RRUFF data with family-level labels instead of mineral identities.
    """
    from pathlib import Path
    from scipy.interpolate import interp1d
    
    subset_dir = Path(rruff_dir) / subset
    if not subset_dir.exists():
        raise DataLoadError('rruff_local', f"Directory not found: {subset_dir}")
    
    txt_files = [f for f in subset_dir.glob('*.txt') 
                 if not f.name.endswith('Zone.Identifier') and 'Processed' in f.name]
    
    print(f"  Found {len(txt_files)} processed RRUFF files")
    
    all_spectra = []
    all_families = []
    all_minerals = []
    
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
            
            # Get chemistry and classify
            ideal_chem = metadata.get('IDEAL CHEMISTRY', '')
            family = classify_mineral_family(ideal_chem)
            mineral = metadata.get('NAMES', 'Unknown').split(',')[0].strip()
            
            all_spectra.append((np.array(wavenumbers), np.array(intensities)))
            all_families.append(family)
            all_minerals.append(mineral)
            
        except Exception:
            continue
    
    print(f"  Parsed {len(all_spectra)} spectra")
    
    # Show family distribution
    family_counts = Counter(all_families)
    print(f"\n  Family distribution:")
    for fam, count in family_counts.most_common():
        print(f"    {fam}: {count}")
    
    # Filter families with too few samples
    valid_families = {f for f, c in family_counts.items() if c >= min_samples_per_family}
    mask = [f in valid_families for f in all_families]
    
    filtered_spectra = [s for s, m in zip(all_spectra, mask) if m]
    filtered_families = [f for f, m in zip(all_families, mask) if m]
    
    print(f"\n  After filtering (≥{min_samples_per_family} samples): {len(filtered_spectra)} spectra, {len(set(filtered_families))} families")
    
    # Determine common wavenumber range
    min_wn = max(wn[0] for wn, _ in filtered_spectra)
    max_wn = min(wn[-1] for wn, _ in filtered_spectra)
    target_wn = np.linspace(max(min_wn, 100), min(max_wn, 2000), 1000)
    
    # Resample
    resampled = []
    valid_families_final = []
    
    for (wn, intensity), family in zip(filtered_spectra, filtered_families):
        if wn.min() > target_wn[0] or wn.max() < target_wn[-1]:
            continue
        f = interp1d(wn, intensity, kind='linear', bounds_error=False, fill_value=0)
        resampled.append(f(target_wn))
        valid_families_final.append(family)
    
    spectra = np.array(resampled)
    
    # Create integer labels
    unique_families = sorted(set(valid_families_final))
    family_to_idx = {f: i for i, f in enumerate(unique_families)}
    labels = np.array([family_to_idx[f] for f in valid_families_final])
    
    metadata = {
        'dataset': f'rruff_families_{subset}',
        'source': 'real',
        'n_samples': len(spectra),
        'n_classes': len(unique_families),
        'class_names': unique_families,
        'task': 'family_classification',
        'loaded_at': datetime.now().isoformat()
    }
    
    return spectra, labels, target_wn, unique_families, metadata


def main():
    SEED = 42
    set_all_seeds(SEED)
    
    print("=" * 70)
    print("RRUFF MINERAL FAMILY CLASSIFICATION")
    print("Scientifically Valid: ~10-15 Chemical Families")
    print("=" * 70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'rruff_families')
    os.makedirs(os.path.join(RESULTS_DIR, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'models'), exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load with family labels
    print("\n[1/5] Loading RRUFF with family labels...")
    spectra, labels, wavenumbers, class_names, metadata = load_rruff_with_families(
        min_samples_per_family=30
    )
    
    n_classes = len(class_names)
    print(f"\n  Final: {len(spectra)} spectra, {n_classes} families")
    print(f"  Families: {class_names}")
    
    # Preprocess
    print("\n[2/5] Preprocessing...")
    pipeline = PreprocessingPipeline()
    spectra = pipeline.process_batch(spectra)
    
    # Split - now valid because we have enough samples per class
    X_train, X_test, y_train, y_test = train_test_split(
        spectra, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=SEED, stratify=y_train
    )
    
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    results = {}
    
    # SVM baseline
    print("\n[3/5] SVM (with StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=SEED)
    svm.fit(X_train_scaled, y_train)
    y_pred_svm = svm.predict(X_test_scaled)
    
    results['SVM'] = {
        'accuracy': accuracy_score(y_test, y_pred_svm),
        'f1_weighted': f1_score(y_test, y_pred_svm, average='weighted')
    }
    print(f"  Accuracy: {results['SVM']['accuracy']:.4f}, F1: {results['SVM']['f1_weighted']:.4f}")
    
    # ResidualCNN
    print("\n[4/5] ResidualCNN...")
    dl_kwargs = get_deterministic_dataloader_kwargs()
    train_loader = DataLoader(SpectralDataset(X_train, y_train), batch_size=32, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(SpectralDataset(X_val, y_val), batch_size=32, **dl_kwargs)
    test_loader = DataLoader(SpectralDataset(X_test, y_test), batch_size=32, **dl_kwargs)
    
    model = ResidualCNN(input_length=len(wavenumbers), n_classes=n_classes, dropout_rate=0.2)
    print(f"  Parameters: {model.count_parameters():,}")
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    model.to(device)
    
    best_val_acc = 0
    patience = 15
    no_improve = 0
    
    for epoch in range(80):
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
    
    all_preds, all_labels = [], []
    for x, y in test_loader:
        all_preds.extend(model.predict(x.to(device)).cpu().numpy())
        all_labels.extend(y.numpy())
    
    y_pred_cnn = np.array(all_preds)
    results['ResidualCNN'] = {
        'accuracy': accuracy_score(y_test, y_pred_cnn),
        'f1_weighted': f1_score(y_test, y_pred_cnn, average='weighted')
    }
    print(f"  Accuracy: {results['ResidualCNN']['accuracy']:.4f}, F1: {results['ResidualCNN']['f1_weighted']:.4f}")
    
    # Confusion matrix
    print("\n[5/5] Saving results...")
    cm = confusion_matrix(y_test, y_pred_cnn)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Mineral Family Classification')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'plots', 'confusion_matrix.png'), dpi=150)
    plt.close()
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'n_classes': n_classes
    }, os.path.join(RESULTS_DIR, 'models', 'family_cnn.pth'))
    
    # Save results
    with open(os.path.join(RESULTS_DIR, 'provenance.json'), 'w') as f:
        json.dump({
            'metadata': metadata,
            'results': results,
            'class_names': class_names,
            'classification_report': classification_report(y_test, y_pred_cnn, 
                                                           target_names=class_names, output_dict=True),
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS - MINERAL FAMILY CLASSIFICATION")
    print("=" * 70)
    print(f"\n{'Model':<15} {'Accuracy':>10} {'F1':>10}")
    print("-" * 40)
    for m, res in results.items():
        print(f"{m:<15} {res['accuracy']:>10.4f} {res['f1_weighted']:>10.4f}")
    
    print(f"\n  Families: {class_names}")
    print(f"  Data source: REAL LOCAL")
    print(f"  Results saved to: {RESULTS_DIR}")
    
    return results


if __name__ == "__main__":
    main()
