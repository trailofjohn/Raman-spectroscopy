"""
Local RRUFF file loader.

Parses RRUFF .txt files downloaded directly from rruff.net.
Format: ## header lines, then wavenumber,intensity CSV data.
"""

import os
import re
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from datetime import datetime
from scipy.interpolate import interp1d

from data.exceptions import DataLoadError


def parse_rruff_file(filepath: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Parse a single RRUFF .txt file.
    
    Returns:
        wavenumbers: 1D array
        intensities: 1D array
        metadata: dict with mineral name, sample ID, etc.
    """
    metadata = {}
    wavenumbers = []
    intensities = []
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('##'):
                # Parse header
                if '=' in line:
                    key, value = line[2:].split('=', 1)
                    metadata[key.strip()] = value.strip()
            elif ',' in line and not line.startswith('#'):
                # Parse data line
                try:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        wn = float(parts[0].strip())
                        intensity = float(parts[1].strip())
                        wavenumbers.append(wn)
                        intensities.append(intensity)
                except ValueError:
                    continue
    
    return np.array(wavenumbers), np.array(intensities), metadata


def load_rruff_local(
    rruff_dir: str = '/home/john/projects/raman_spectroscopy_pipeline/rruff',
    subset: str = 'excellent_unoriented',
    target_wavenumbers: Optional[np.ndarray] = None,
    processed_only: bool = True,
    min_points: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Dict]:
    """
    Load RRUFF spectra from local files.
    
    Args:
        rruff_dir: Path to RRUFF data directory
        subset: Which subset to load (excellent_unoriented, excellent_oriented, etc.)
        target_wavenumbers: Resample all spectra to this grid (default: common range)
        processed_only: Only load "Processed" files (cleaner data)
        min_points: Minimum data points required per spectrum
    
    Returns:
        spectra: (n_samples, n_wavenumbers) array
        labels: (n_samples,) integer labels
        wavenumbers: (n_wavenumbers,) wavenumber axis
        class_names: list of mineral names
        metadata: provenance info
    """
    subset_dir = Path(rruff_dir) / subset
    
    if not subset_dir.exists():
        raise DataLoadError('rruff_local', f"Directory not found: {subset_dir}")
    
    # Find all .txt files (exclude Zone.Identifier Windows metadata)
    txt_files = [f for f in subset_dir.glob('*.txt') 
                 if not f.name.endswith('Zone.Identifier')]
    
    if processed_only:
        txt_files = [f for f in txt_files if 'Processed' in f.name]
    
    if not txt_files:
        raise DataLoadError('rruff_local', f"No files found in {subset_dir}")
    
    print(f"  Found {len(txt_files)} RRUFF files in {subset}")
    
    # Parse all files
    all_spectra = []
    all_minerals = []
    all_wavenumbers = []
    skipped = 0
    
    for filepath in txt_files:
        try:
            wn, intensity, meta = parse_rruff_file(filepath)
            
            if len(wn) < min_points:
                skipped += 1
                continue
            
            mineral = meta.get('NAMES', 'Unknown')
            # Take first name if multiple
            if ',' in mineral:
                mineral = mineral.split(',')[0].strip()
            
            all_spectra.append((wn, intensity))
            all_minerals.append(mineral)
            all_wavenumbers.append((wn.min(), wn.max()))
            
        except Exception as e:
            skipped += 1
            continue
    
    if not all_spectra:
        raise DataLoadError('rruff_local', 'No valid spectra loaded')
    
    print(f"  Parsed {len(all_spectra)} spectra, skipped {skipped}")
    
    # Determine common wavenumber range
    if target_wavenumbers is None:
        # Find overlap range
        min_wn = max(wn[0] for wn, _ in all_spectra)
        max_wn = min(wn[-1] for wn, _ in all_spectra)
        
        # Use reasonable Raman range
        min_wn = max(min_wn, 100)
        max_wn = min(max_wn, 2000)
        
        target_wavenumbers = np.linspace(min_wn, max_wn, 1000)
    
    print(f"  Resampling to {len(target_wavenumbers)} points ({target_wavenumbers[0]:.0f}-{target_wavenumbers[-1]:.0f} cm⁻¹)")
    
    # Resample all spectra
    resampled_spectra = []
    valid_minerals = []
    
    for (wn, intensity), mineral in zip(all_spectra, all_minerals):
        try:
            # Only use spectra that cover target range
            if wn.min() > target_wavenumbers[0] or wn.max() < target_wavenumbers[-1]:
                continue
            
            f = interp1d(wn, intensity, kind='linear', bounds_error=False, fill_value=0)
            resampled = f(target_wavenumbers)
            resampled_spectra.append(resampled)
            valid_minerals.append(mineral)
        except:
            continue
    
    if not resampled_spectra:
        raise DataLoadError('rruff_local', 'No spectra cover target wavenumber range')
    
    spectra = np.array(resampled_spectra)
    
    # Create integer labels
    unique_minerals = sorted(set(valid_minerals))
    mineral_to_idx = {m: i for i, m in enumerate(unique_minerals)}
    labels = np.array([mineral_to_idx[m] for m in valid_minerals])
    
    # Filter classes with too few samples
    unique, counts = np.unique(labels, return_counts=True)
    valid_classes = unique[counts >= 2]
    
    if len(valid_classes) < len(unique):
        mask = np.isin(labels, valid_classes)
        spectra = spectra[mask]
        labels_filtered = labels[mask]
        
        # Reindex
        old_to_new = {old: new for new, old in enumerate(sorted(valid_classes))}
        labels = np.array([old_to_new[l] for l in labels_filtered])
        unique_minerals = [unique_minerals[i] for i in sorted(valid_classes)]
        
        print(f"  Filtered to {len(unique_minerals)} minerals with ≥2 samples")
    
    metadata = {
        'dataset': f'rruff_local_{subset}',
        'source': 'real',
        'loaded_from': str(subset_dir),
        'n_samples': len(spectra),
        'n_classes': len(unique_minerals),
        'wavenumber_range': (float(target_wavenumbers[0]), float(target_wavenumbers[-1])),
        'loaded_at': datetime.now().isoformat()
    }
    
    print(f"  Loaded: {len(spectra)} spectra, {len(unique_minerals)} minerals [REAL DATA]")
    
    return spectra, labels, target_wavenumbers, unique_minerals, metadata


if __name__ == "__main__":
    # Test
    try:
        spectra, labels, wn, minerals, meta = load_rruff_local()
        print(f"\nSuccess!")
        print(f"  Shape: {spectra.shape}")
        print(f"  Minerals: {minerals[:5]}...")
        print(f"  Metadata: {meta}")
    except DataLoadError as e:
        print(f"Error: {e}")
