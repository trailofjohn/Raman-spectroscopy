"""
Data Loader for RamanSpy datasets with proper error handling and caching.

Features:
- Downloads datasets ONCE to data/raw/ directory
- Loads from local on subsequent runs (no re-download)
- Hard fail on dataset load errors (no silent synthetic fallback)
- Spectral resampling for uniform wavenumber grids
- Label normalization for RRUFF minerals
- Provenance metadata in returned data
"""

import os
import hashlib
import numpy as np
from typing import Tuple, Dict, List, Optional, NamedTuple
from datetime import datetime
from pathlib import Path
from scipy.interpolate import interp1d
import warnings

try:
    import ramanspy as rp
    RAMANSPY_AVAILABLE = True
except ImportError:
    RAMANSPY_AVAILABLE = False

from data.exceptions import DataLoadError, SpectralAlignmentError


# Local storage directory for downloaded datasets
DATA_DIR = Path(__file__).parent / 'raw'
DATA_DIR.mkdir(parents=True, exist_ok=True)


class DataBundle(NamedTuple):
    """Structured return type for loaded datasets with provenance."""
    spectra: np.ndarray
    labels: np.ndarray
    wavenumbers: np.ndarray
    class_names: List[str]
    metadata: Dict


# RRUFF mineral name normalization map (common variants -> canonical)
MINERAL_NORMALIZATION = {
    'quartz': 'Quartz',
    'QUARTZ': 'Quartz',
    'calcite': 'Calcite',
    'CALCITE': 'Calcite',
    'feldspar': 'Feldspar',
    'orthoclase': 'Feldspar',
    'plagioclase': 'Plagioclase',
    'olivine': 'Olivine',
    'forsterite': 'Olivine',
    'fayalite': 'Olivine',
    'pyroxene': 'Pyroxene',
    'augite': 'Pyroxene',
    'diopside': 'Pyroxene',
    'enstatite': 'Pyroxene',
}


def normalize_mineral_name(name: str) -> str:
    """Normalize mineral names to canonical form."""
    name_lower = name.lower().strip()
    return MINERAL_NORMALIZATION.get(name_lower, name.title())


def resample_spectrum(spectrum: np.ndarray, source_wavenumbers: np.ndarray, 
                      target_wavenumbers: np.ndarray) -> np.ndarray:
    """Resample a spectrum to a target wavenumber grid using linear interpolation."""
    if len(spectrum) == len(target_wavenumbers):
        return spectrum
    
    f = interp1d(source_wavenumbers, spectrum, kind='linear', 
                 bounds_error=False, fill_value='extrapolate')
    return f(target_wavenumbers)


def load_wheat_lines(allow_synthetic: bool = False, 
                     target_wavenumbers: Optional[np.ndarray] = None) -> DataBundle:
    """
    Load wheat_lines dataset.
    
    Downloads to data/raw/wheat_lines/ on first run, loads from local after.
    
    Args:
        allow_synthetic: If False, raises DataLoadError on failure
        target_wavenumbers: If provided, resample spectra to this grid
    
    Returns:
        DataBundle with spectra, labels, wavenumbers, class_names, metadata
    
    Raises:
        DataLoadError: If loading fails and allow_synthetic=False
    """
    if not RAMANSPY_AVAILABLE:
        if allow_synthetic:
            return _generate_synthetic_fallback('wheat_lines', n_classes=4)
        raise DataLoadError('wheat_lines', 'RamanSPy not installed', allow_synthetic_hint=True)
    
    wheat_dir = DATA_DIR / 'wheat_lines'
    wheat_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    cached_file = wheat_dir / 'wheat_data.npz'
    
    if cached_file.exists():
        # Load from local cache
        print(f"  Loading wheat_lines from local: {cached_file}")
        data = np.load(cached_file, allow_pickle=True)
        spectra = data['spectra']
        labels = data['labels']
        wavenumbers = data['wavenumbers']
        class_names = list(data['class_names'])
        loaded_from = 'local_cache'
    else:
        # Download and cache
        print(f"  Downloading wheat_lines (first run, ~700MB)...")
        try:
            container, labels, label_names = rp.datasets.wheat_lines()
            spectra = np.array(container.spectral_data)
            wavenumbers = np.array(container.spectral_axis)
            labels = np.array(labels).astype(int)
            class_names = [str(n) for n in label_names]
            
            # Save to local cache
            np.savez_compressed(
                cached_file,
                spectra=spectra,
                labels=labels,
                wavenumbers=wavenumbers,
                class_names=np.array(class_names, dtype=object)
            )
            print(f"  Saved to local cache: {cached_file}")
            loaded_from = 'download'
            
        except Exception as e:
            if allow_synthetic:
                warnings.warn(f"wheat_lines failed ({e}), using synthetic fallback")
                return _generate_synthetic_fallback('wheat_lines', n_classes=4)
            raise DataLoadError('wheat_lines', str(e), allow_synthetic_hint=True)
    
    # Resample if target wavenumbers provided
    if target_wavenumbers is not None and len(wavenumbers) != len(target_wavenumbers):
        resampled = np.zeros((spectra.shape[0], len(target_wavenumbers)))
        for i in range(spectra.shape[0]):
            resampled[i] = resample_spectrum(spectra[i], wavenumbers, target_wavenumbers)
        spectra = resampled
        wavenumbers = target_wavenumbers
    
    metadata = {
        'dataset': 'wheat_lines',
        'source': 'real',
        'loaded_from': loaded_from,
        'n_samples': len(spectra),
        'n_classes': len(class_names),
        'loaded_at': datetime.now().isoformat()
    }
    
    print(f"  wheat_lines: {spectra.shape[0]} samples, {len(class_names)} classes [REAL DATA]")
    return DataBundle(spectra, labels, wavenumbers, class_names, metadata)


def load_rruff(dataset: str = 'excellent_unoriented', 
               allow_synthetic: bool = False,
               target_wavenumbers: Optional[np.ndarray] = None) -> DataBundle:
    """
    Load RRUFF mineral Raman database.
    
    Downloads to data/raw/rruff/ on first run, loads from local after.
    
    Args:
        dataset: RRUFF subset name ('excellent_unoriented', etc.)
        allow_synthetic: If False, raises DataLoadError on failure
        target_wavenumbers: If provided, resample all spectra to this grid
    
    Returns:
        DataBundle with spectra, labels, wavenumbers, class_names, metadata
    
    Raises:
        DataLoadError: If loading fails and allow_synthetic=False
    """
    if not RAMANSPY_AVAILABLE:
        if allow_synthetic:
            return _generate_synthetic_fallback('rruff', n_classes=10)
        raise DataLoadError('rruff', 'RamanSPy not installed', allow_synthetic_hint=True)
    
    rruff_dir = DATA_DIR / 'rruff'
    rruff_dir.mkdir(parents=True, exist_ok=True)
    cached_file = rruff_dir / f'rruff_{dataset}.npz'
    
    if cached_file.exists():
        # Load from local cache
        print(f"  Loading RRUFF from local: {cached_file}")
        data = np.load(cached_file, allow_pickle=True)
        spectra = data['spectra']
        labels = data['labels']
        wavenumbers = data['wavenumbers']
        class_names = list(data['class_names'])
        loaded_from = 'local_cache'
    else:
        # Download and cache
        print(f"  Downloading RRUFF ({dataset})...")
        try:
            result = rp.datasets.rruff(dataset)
            
            if isinstance(result, tuple) and len(result) == 2:
                spectra_list, metadata_list = result
            else:
                raise DataLoadError('rruff', f"Unexpected return type: {type(result)}")
            
            if len(spectra_list) == 0:
                raise DataLoadError('rruff', 'Empty dataset returned')
            
            # Get reference wavenumber grid
            ref_wavenumbers = np.array(spectra_list[0].spectral_axis)
            
            all_spectra = []
            all_labels = []
            
            for spectrum, meta in zip(spectra_list, metadata_list):
                spec_data = np.array(spectrum.spectral_data).flatten()
                spec_wavenumbers = np.array(spectrum.spectral_axis)
                
                # Resample to reference grid if needed
                if len(spec_data) != len(ref_wavenumbers):
                    spec_data = resample_spectrum(spec_data, spec_wavenumbers, ref_wavenumbers)
                
                all_spectra.append(spec_data)
                
                # Extract and normalize mineral name
                mineral_name = 'unknown'
                if 'names' in meta:
                    names = meta['names']
                    if isinstance(names, list) and names:
                        mineral_name = names[0]
                    elif isinstance(names, str):
                        mineral_name = names
                elif 'name' in meta:
                    mineral_name = meta['name']
                
                mineral_name = normalize_mineral_name(mineral_name)
                all_labels.append(mineral_name)
            
            spectra = np.array(all_spectra)
            wavenumbers = ref_wavenumbers
            unique_labels = sorted(set(all_labels))
            label_map = {name: idx for idx, name in enumerate(unique_labels)}
            labels = np.array([label_map[l] for l in all_labels])
            class_names = unique_labels
            
            # Save to local cache
            np.savez_compressed(
                cached_file,
                spectra=spectra,
                labels=labels,
                wavenumbers=wavenumbers,
                class_names=np.array(class_names, dtype=object)
            )
            print(f"  Saved to local cache: {cached_file}")
            loaded_from = 'download'
            
        except DataLoadError:
            raise
        except Exception as e:
            if allow_synthetic:
                warnings.warn(f"rruff failed ({e}), using synthetic fallback")
                return _generate_synthetic_fallback('rruff', n_classes=10)
            raise DataLoadError('rruff', str(e), allow_synthetic_hint=True)
    
    # Resample to target wavenumbers if provided
    if target_wavenumbers is not None and len(wavenumbers) != len(target_wavenumbers):
        resampled = np.zeros((spectra.shape[0], len(target_wavenumbers)))
        for i in range(spectra.shape[0]):
            resampled[i] = resample_spectrum(spectra[i], wavenumbers, target_wavenumbers)
        spectra = resampled
        wavenumbers = target_wavenumbers
    
    # Convert string labels to integers if loaded from cache
    if labels.dtype.kind == 'U' or labels.dtype.kind == 'O':
        unique_labels = sorted(set(labels))
        label_map = {name: idx for idx, name in enumerate(unique_labels)}
        labels = np.array([label_map[str(l)] for l in labels])
        class_names = unique_labels
    
    metadata = {
        'dataset': f'rruff_{dataset}',
        'source': 'real',
        'loaded_from': loaded_from,
        'n_samples': len(spectra),
        'n_classes': len(class_names),
        'loaded_at': datetime.now().isoformat()
    }
    
    print(f"  rruff ({dataset}): {spectra.shape[0]} spectra, {len(class_names)} minerals [REAL DATA]")
    return DataBundle(spectra, labels, wavenumbers, class_names, metadata)


def _generate_synthetic_fallback(dataset_name: str, n_classes: int = 6, 
                                  samples_per_class: int = 100, 
                                  n_wavenumbers: int = 1000,
                                  seed: int = 42) -> DataBundle:
    """Generate synthetic data when real data fails. Clearly marked as synthetic."""
    np.random.seed(seed)
    wavenumbers = np.linspace(400, 1800, n_wavenumbers)
    
    class_peaks = [
        [(600, 0.8, 15), (1000, 1.0, 20), (1450, 0.6, 15)],
        [(700, 1.0, 18), (1100, 0.7, 15), (1500, 0.8, 20)],
        [(550, 0.6, 12), (850, 0.9, 18), (1200, 1.0, 22)],
        [(650, 0.85, 16), (950, 0.75, 14), (1350, 0.95, 18)],
        [(500, 0.7, 14), (800, 1.0, 20), (1250, 0.65, 16)],
        [(750, 0.9, 17), (1050, 0.8, 15), (1400, 0.75, 19)],
    ]
    
    spectra, labels = [], []
    class_names = [f"Synthetic_{i}" for i in range(n_classes)]
    
    for c in range(n_classes):
        peaks = class_peaks[c % len(class_peaks)]
        for _ in range(samples_per_class):
            spectrum = np.zeros(n_wavenumbers)
            for center, amp, width in peaks:
                spectrum += amp * np.random.uniform(0.8, 1.2) * \
                           np.exp(-((wavenumbers - center + np.random.normal(0, 5)) ** 2) / 
                                  (2 * (width * np.random.uniform(0.9, 1.1)) ** 2))
            spectrum += np.random.normal(0, 0.05, n_wavenumbers)
            spectra.append(np.maximum(spectrum, 0))
            labels.append(c)
    
    metadata = {
        'dataset': f'{dataset_name}_SYNTHETIC_FALLBACK',
        'source': 'synthetic',
        'n_samples': len(spectra),
        'n_classes': n_classes,
        'seed': seed,
        'WARNING': 'This is SYNTHETIC data, not real measurements!',
        'loaded_at': datetime.now().isoformat()
    }
    
    print(f"  ⚠️  {dataset_name}: {len(spectra)} SYNTHETIC samples (real data failed)")
    return DataBundle(np.array(spectra), np.array(labels), wavenumbers, class_names, metadata)


def load_all_datasets(allow_synthetic: bool = False, 
                      target_wavenumbers: int = 1000) -> Dict[str, DataBundle]:
    """
    Load all available RamanSPy datasets.
    
    Args:
        allow_synthetic: If False, skip datasets that fail (don't substitute)
        target_wavenumbers: Number of wavenumber points to resample to
    
    Returns:
        Dict mapping dataset name to DataBundle
    """
    print("Loading RamanSPy datasets...")
    print("=" * 50)
    
    target_wn = np.linspace(400, 1800, target_wavenumbers)
    datasets = {}
    
    # Try wheat_lines
    try:
        datasets['wheat_lines'] = load_wheat_lines(
            allow_synthetic=allow_synthetic, 
            target_wavenumbers=target_wn
        )
    except DataLoadError as e:
        print(f"  ✗ wheat_lines: {e.reason}")
    
    # Try RRUFF
    try:
        datasets['rruff'] = load_rruff(
            dataset='excellent_unoriented',
            allow_synthetic=allow_synthetic,
            target_wavenumbers=target_wn
        )
    except DataLoadError as e:
        print(f"  ✗ rruff: {e.reason}")
    
    print("=" * 50)
    print(f"Loaded {len(datasets)} datasets")
    
    if not datasets and not allow_synthetic:
        raise DataLoadError('all', 'No datasets could be loaded')
    
    return datasets


if __name__ == "__main__":
    # Test loading
    print("Testing data loaders...\n")
    
    try:
        bundle = load_wheat_lines(allow_synthetic=False)
        print(f"  Wheat: {bundle.metadata}")
    except DataLoadError as e:
        print(f"  Wheat failed: {e}")
    
    try:
        bundle = load_rruff(allow_synthetic=False)
        print(f"  RRUFF: {bundle.metadata}")
    except DataLoadError as e:
        print(f"  RRUFF failed: {e}")
