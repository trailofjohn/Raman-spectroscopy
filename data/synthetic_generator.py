"""
Synthetic Spectra Generator for Concentration Estimation.

This module generates synthetic Raman spectra mixtures with known concentrations
for training regression models.

Author: Spectroscopic ML Pipeline
Date: 2024-12
"""

import numpy as np
from typing import Tuple, List, Optional
from scipy.signal import find_peaks


def generate_gaussian_peak(
    wavenumbers: np.ndarray,
    center: float,
    amplitude: float,
    width: float
) -> np.ndarray:
    """
    Generate a Gaussian peak.
    
    Args:
        wavenumbers: Wavenumber axis
        center: Peak center position
        amplitude: Peak height
        width: Peak width (sigma)
    
    Returns:
        Gaussian peak intensity values
    """
    return amplitude * np.exp(-((wavenumbers - center) ** 2) / (2 * width ** 2))


def generate_lorentzian_peak(
    wavenumbers: np.ndarray,
    center: float,
    amplitude: float,
    width: float
) -> np.ndarray:
    """
    Generate a Lorentzian peak (more realistic for Raman).
    
    Args:
        wavenumbers: Wavenumber axis
        center: Peak center position
        amplitude: Peak height
        width: Half-width at half-maximum (HWHM)
    
    Returns:
        Lorentzian peak intensity values
    """
    return amplitude * (width ** 2) / ((wavenumbers - center) ** 2 + width ** 2)


def generate_component_spectrum(
    wavenumbers: np.ndarray,
    n_peaks: int = 5,
    peak_type: str = 'lorentzian',
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a synthetic pure component Raman spectrum.
    
    Args:
        wavenumbers: Wavenumber axis
        n_peaks: Number of peaks in the spectrum
        peak_type: 'gaussian' or 'lorentzian'
        seed: Random seed for reproducibility
    
    Returns:
        Synthetic spectrum intensity values
    """
    if seed is not None:
        np.random.seed(seed)
    
    spectrum = np.zeros_like(wavenumbers, dtype=float)
    
    # Generate random peak parameters
    wn_min, wn_max = wavenumbers.min(), wavenumbers.max()
    wn_range = wn_max - wn_min
    
    for _ in range(n_peaks):
        center = np.random.uniform(wn_min + 0.1 * wn_range, wn_max - 0.1 * wn_range)
        amplitude = np.random.uniform(0.3, 1.0)
        width = np.random.uniform(5, 30)
        
        if peak_type == 'gaussian':
            spectrum += generate_gaussian_peak(wavenumbers, center, amplitude, width)
        else:
            spectrum += generate_lorentzian_peak(wavenumbers, center, amplitude, width)
    
    return spectrum


def generate_mixture_spectra(
    n_samples: int = 1000,
    n_components: int = 4,
    n_wavenumbers: int = 1000,
    wavenumber_range: Tuple[float, float] = (400, 1800),
    noise_level: float = 0.02,
    concentration_range: Tuple[float, float] = (0.0, 1.0),
    normalize_concentrations: bool = True,
    seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic mixture Raman spectra with known concentrations.
    
    This function creates multi-component mixtures following Beer-Lambert law
    (linear combinations of pure component spectra).
    
    Args:
        n_samples: Number of mixture spectra to generate
        n_components: Number of chemical components
        n_wavenumbers: Number of wavenumber points
        wavenumber_range: (min, max) wavenumber values in cm^-1
        noise_level: Noise standard deviation (fraction of max intensity)
        concentration_range: (min, max) concentration values
        normalize_concentrations: If True, concentrations sum to 1.0
        seed: Random seed for reproducibility
    
    Returns:
        Tuple containing:
            - spectra (np.ndarray): Shape (n_samples, n_wavenumbers)
            - concentrations (np.ndarray): Shape (n_samples, n_components)
            - wavenumbers (np.ndarray): Shape (n_wavenumbers,)
            - pure_components (np.ndarray): Shape (n_components, n_wavenumbers)
    """
    np.random.seed(seed)
    
    # Create wavenumber axis
    wavenumbers = np.linspace(wavenumber_range[0], wavenumber_range[1], n_wavenumbers)
    
    # Generate pure component spectra
    pure_components = np.zeros((n_components, n_wavenumbers))
    for i in range(n_components):
        pure_components[i] = generate_component_spectrum(
            wavenumbers, 
            n_peaks=np.random.randint(3, 8),
            peak_type='lorentzian',
            seed=seed + i * 100
        )
        # Normalize each component
        pure_components[i] /= pure_components[i].max()
    
    # Generate random concentrations
    concentrations = np.random.uniform(
        concentration_range[0], 
        concentration_range[1], 
        size=(n_samples, n_components)
    )
    
    # Normalize concentrations to sum to 1 if requested
    if normalize_concentrations:
        concentrations = concentrations / concentrations.sum(axis=1, keepdims=True)
    
    # Generate mixture spectra (linear combinations)
    spectra = np.dot(concentrations, pure_components)
    
    # Add noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level * spectra.max(), spectra.shape)
        spectra = spectra + noise
        spectra = np.maximum(spectra, 0)  # Ensure non-negative
    
    print(f"Generated {n_samples} synthetic mixture spectra")
    print(f"  Components: {n_components}")
    print(f"  Wavenumber range: {wavenumber_range[0]}-{wavenumber_range[1]} cm^-1")
    print(f"  Noise level: {noise_level*100:.1f}%")
    
    return spectra, concentrations, wavenumbers, pure_components


def generate_training_dataset(
    n_train: int = 800,
    n_test: int = 200,
    **kwargs
) -> dict:
    """
    Generate train/test split synthetic dataset.
    
    Args:
        n_train: Number of training samples
        n_test: Number of test samples
        **kwargs: Additional arguments passed to generate_mixture_spectra
    
    Returns:
        Dictionary containing train and test data splits
    """
    total_samples = n_train + n_test
    
    spectra, concentrations, wavenumbers, pure_components = generate_mixture_spectra(
        n_samples=total_samples,
        **kwargs
    )
    
    # Split data
    indices = np.random.permutation(total_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    dataset = {
        'X_train': spectra[train_idx],
        'X_test': spectra[test_idx],
        'y_train': concentrations[train_idx],
        'y_test': concentrations[test_idx],
        'wavenumbers': wavenumbers,
        'pure_components': pure_components,
        'component_names': [f'Component_{i+1}' for i in range(kwargs.get('n_components', 4))]
    }
    
    print(f"\nDataset split:")
    print(f"  Training: {n_train} samples")
    print(f"  Testing: {n_test} samples")
    
    return dataset


if __name__ == "__main__":
    # Test synthetic data generation
    dataset = generate_training_dataset(
        n_train=800,
        n_test=200,
        n_components=4,
        n_wavenumbers=1000,
        wavenumber_range=(400, 1800),
        noise_level=0.02,
        seed=42
    )
    
    print("\n" + "=" * 50)
    print("Dataset Summary:")
    print(f"  X_train shape: {dataset['X_train'].shape}")
    print(f"  y_train shape: {dataset['y_train'].shape}")
    print(f"  X_test shape: {dataset['X_test'].shape}")
    print(f"  y_test shape: {dataset['y_test'].shape}")
    print(f"  Wavenumbers: {len(dataset['wavenumbers'])} points")
    print(f"  Components: {dataset['component_names']}")
