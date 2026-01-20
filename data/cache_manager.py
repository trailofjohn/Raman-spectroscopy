"""
Cache Manager for Raman Spectroscopy Pipeline.

Provides caching of preprocessed data with proper invalidation:
- Hash includes: dataset version, preprocessing params, code version, split indices
- Eliminates 16+ minute data download/preprocessing on repeated runs
"""

import os
import json
import hashlib
import numpy as np
from typing import Dict, Tuple, Optional, List
from datetime import datetime
from pathlib import Path

from data.exceptions import CacheInvalidError


CACHE_DIR = Path(__file__).parent / 'cache'


def get_code_version() -> str:
    """Get code version from git or fallback to static version."""
    try:
        import subprocess
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except:
        pass
    return "v1.0.0"  # Fallback static version


def compute_config_hash(dataset_name: str, preprocessing_config: Dict, 
                        split_seed: int, target_wavenumbers: int) -> str:
    """
    Compute hash of configuration for cache invalidation.
    
    Includes:
    - Dataset name
    - Preprocessing parameters
    - Random seed for splits
    - Target wavenumber count
    - Code version
    """
    config = {
        'dataset': dataset_name,
        'preprocessing': preprocessing_config,
        'split_seed': split_seed,
        'target_wavenumbers': target_wavenumbers,
        'code_version': get_code_version()
    }
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:12]


def get_cache_path(dataset_name: str, config_hash: str) -> Path:
    """Get path to cache file."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{dataset_name}_{config_hash}.npz"


def save_to_cache(cache_path: Path, 
                  spectra: np.ndarray, 
                  labels: np.ndarray,
                  wavenumbers: np.ndarray,
                  class_names: List[str],
                  metadata: Dict,
                  train_indices: np.ndarray,
                  val_indices: np.ndarray,
                  test_indices: np.ndarray) -> None:
    """
    Save preprocessed data and split indices to cache.
    
    Args:
        cache_path: Path to save cache file
        spectra: Preprocessed spectral data
        labels: Label array
        wavenumbers: Wavenumber axis
        class_names: Class name list
        metadata: Dataset metadata dict
        train_indices: Training set indices
        val_indices: Validation set indices
        test_indices: Test set indices
    """
    np.savez_compressed(
        cache_path,
        spectra=spectra,
        labels=labels,
        wavenumbers=wavenumbers,
        class_names=np.array(class_names, dtype=object),
        metadata=np.array([json.dumps(metadata)]),
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        saved_at=np.array([datetime.now().isoformat()])
    )
    print(f"  Cached to: {cache_path} ({cache_path.stat().st_size / 1024 / 1024:.1f} MB)")


def load_from_cache(cache_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                List[str], Dict, np.ndarray, 
                                                np.ndarray, np.ndarray]:
    """
    Load preprocessed data and split indices from cache.
    
    Returns:
        Tuple of (spectra, labels, wavenumbers, class_names, metadata,
                  train_indices, val_indices, test_indices)
    
    Raises:
        CacheInvalidError: If cache file is corrupted or incompatible
    """
    if not cache_path.exists():
        raise CacheInvalidError(str(cache_path), "File does not exist")
    
    try:
        data = np.load(cache_path, allow_pickle=True)
        
        spectra = data['spectra']
        labels = data['labels']
        wavenumbers = data['wavenumbers']
        class_names = list(data['class_names'])
        metadata = json.loads(data['metadata'][0])
        train_indices = data['train_indices']
        val_indices = data['val_indices']
        test_indices = data['test_indices']
        
        print(f"  Loaded from cache: {cache_path.name}")
        print(f"    {len(spectra)} samples, {len(class_names)} classes")
        print(f"    Saved at: {data['saved_at'][0]}")
        
        return (spectra, labels, wavenumbers, class_names, metadata,
                train_indices, val_indices, test_indices)
    
    except Exception as e:
        raise CacheInvalidError(str(cache_path), str(e))


def cache_exists(dataset_name: str, config_hash: str) -> bool:
    """Check if valid cache exists for given config."""
    cache_path = get_cache_path(dataset_name, config_hash)
    return cache_path.exists()


def clear_cache(dataset_name: Optional[str] = None) -> int:
    """
    Clear cache files.
    
    Args:
        dataset_name: If provided, only clear caches for this dataset
        
    Returns:
        Number of files deleted
    """
    if not CACHE_DIR.exists():
        return 0
    
    deleted = 0
    pattern = f"{dataset_name}_*.npz" if dataset_name else "*.npz"
    
    for cache_file in CACHE_DIR.glob(pattern):
        cache_file.unlink()
        deleted += 1
    
    print(f"Cleared {deleted} cache file(s)")
    return deleted


def get_preprocessing_config(pipeline) -> Dict:
    """Extract preprocessing configuration from pipeline object."""
    return {
        'use_cosmic_ray_removal': getattr(pipeline, 'use_cosmic_ray_removal', False),
        'use_denoising': getattr(pipeline, 'use_denoising', False),
        'use_baseline_correction': getattr(pipeline, 'use_baseline_correction', False),
        'use_normalization': getattr(pipeline, 'use_normalization', False),
        'normalization_method': getattr(pipeline, 'normalization_method', 'minmax'),
        'asls_lam': getattr(pipeline, 'asls_lam', 1e5),
        'asls_p': getattr(pipeline, 'asls_p', 0.01),
    }


if __name__ == "__main__":
    # Test cache operations
    print("Testing cache manager...")
    
    config_hash = compute_config_hash(
        'test_dataset', 
        {'use_normalization': True},
        seed=42,
        target_wavenumbers=1000
    )
    print(f"Config hash: {config_hash}")
    print(f"Code version: {get_code_version()}")
    
    cache_path = get_cache_path('test_dataset', config_hash)
    print(f"Cache path: {cache_path}")
