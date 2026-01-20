"""
Preprocessing Pipeline for Raman Spectroscopy.

This module implements a reproducible preprocessing pipeline using RamanSpy
with standard spectroscopic preprocessing methods.

Pipeline Steps:
    1. Cosmic Ray Removal (Whitaker-Hayes)
    2. Denoising (Savitzky-Golay filter)
    3. Baseline Correction (ASLS)
    4. Normalization (Min-Max or Vector)

Author: Spectroscopic ML Pipeline
Date: 2024-12
"""

import numpy as np
from typing import Optional, List, Tuple, Union
import warnings

try:
    import ramanspy as rp
    RAMANSPY_AVAILABLE = True
except ImportError:
    RAMANSPY_AVAILABLE = False
    warnings.warn("RamanSpy not available. Using fallback preprocessing methods.")

from scipy.signal import savgol_filter
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


class PreprocessingPipeline:
    """
    Comprehensive preprocessing pipeline for Raman spectra.
    
    Implements standard spectroscopic preprocessing methods either using
    RamanSpy (if available) or fallback scipy implementations.
    
    Attributes:
        use_cosmic_ray_removal (bool): Enable Whitaker-Hayes spike removal
        use_denoising (bool): Enable Savitzky-Golay denoising
        use_baseline_correction (bool): Enable ASLS baseline correction
        use_normalization (bool): Enable intensity normalization
        normalization_method (str): 'minmax' or 'vector'
    """
    
    def __init__(
        self,
        use_cosmic_ray_removal: bool = True,
        use_denoising: bool = True,
        use_baseline_correction: bool = True,
        use_normalization: bool = True,
        normalization_method: str = 'minmax',
        savgol_window: int = 9,
        savgol_polyorder: int = 3,
        asls_lam: float = 1e5,
        asls_p: float = 0.01,
        asls_niter: int = 10
    ):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            use_cosmic_ray_removal: Enable spike removal step
            use_denoising: Enable Savitzky-Golay smoothing
            use_baseline_correction: Enable ASLS baseline correction
            use_normalization: Enable intensity normalization
            normalization_method: 'minmax' (0-1 range) or 'vector' (unit norm)
            savgol_window: Savitzky-Golay filter window length (odd number)
            savgol_polyorder: Polynomial order for SavGol filter
            asls_lam: ASLS smoothness parameter (larger = smoother baseline)
            asls_p: ASLS asymmetry parameter (smaller = more asymmetric)
            asls_niter: Number of ASLS iterations
        """
        self.use_cosmic_ray_removal = use_cosmic_ray_removal
        self.use_denoising = use_denoising
        self.use_baseline_correction = use_baseline_correction
        self.use_normalization = use_normalization
        self.normalization_method = normalization_method
        
        # Savitzky-Golay parameters
        self.savgol_window = savgol_window
        self.savgol_polyorder = savgol_polyorder
        
        # ASLS parameters
        self.asls_lam = asls_lam
        self.asls_p = asls_p
        self.asls_niter = asls_niter
        
        # Build RamanSpy pipeline if available
        if RAMANSPY_AVAILABLE:
            self._build_ramanspy_pipeline()
        
    def _build_ramanspy_pipeline(self):
        """Build RamanSpy preprocessing pipeline."""
        steps = []
        
        if self.use_cosmic_ray_removal:
            steps.append(rp.preprocessing.despike.WhitakerHayes())
        
        if self.use_denoising:
            steps.append(rp.preprocessing.denoise.SavGol(
                window_length=self.savgol_window,
                polyorder=self.savgol_polyorder
            ))
        
        if self.use_baseline_correction:
            steps.append(rp.preprocessing.baseline.ASLS(
                lam=self.asls_lam,
                p=self.asls_p
            ))
        
        if self.use_normalization:
            if self.normalization_method == 'minmax':
                steps.append(rp.preprocessing.normalise.MinMax())
            else:
                steps.append(rp.preprocessing.normalise.Vector())
        
        self.rp_pipeline = rp.preprocessing.Pipeline(steps) if steps else None
    
    def _whitaker_hayes_despike(self, spectrum: np.ndarray, threshold: float = 6.0) -> np.ndarray:
        """
        Fallback Whitaker-Hayes cosmic ray removal.
        
        Uses modified Z-scores to detect and interpolate spikes.
        """
        # Calculate modified Z-score based on median absolute deviation
        spectrum = spectrum.copy()
        diff1 = np.diff(spectrum)
        median_diff = np.median(np.abs(diff1))
        
        if median_diff == 0:
            return spectrum
        
        mad = 0.6745 * diff1 / median_diff
        spike_indices = np.where(np.abs(mad) > threshold)[0]
        
        # Interpolate spikes
        for idx in spike_indices:
            if 0 < idx < len(spectrum) - 1:
                spectrum[idx] = (spectrum[idx - 1] + spectrum[idx + 1]) / 2
        
        return spectrum
    
    def _savgol_denoise(self, spectrum: np.ndarray) -> np.ndarray:
        """Apply Savitzky-Golay filter for denoising."""
        window = min(self.savgol_window, len(spectrum))
        if window % 2 == 0:
            window -= 1
        if window <= self.savgol_polyorder:
            return spectrum
        return savgol_filter(spectrum, window, self.savgol_polyorder)
    
    def _asls_baseline(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Asymmetric Least Squares baseline correction.
        
        Based on Eilers & Boelens (2005).
        Uses CSC format for sparse matrices to optimize spsolve performance.
        """
        L = len(spectrum)
        # Create difference matrix in CSC format for efficient spsolve
        D = diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2), format='csc')
        D = self.asls_lam * D.dot(D.transpose())
        w = np.ones(L)
        
        for _ in range(self.asls_niter):
            W = diags(w, 0, shape=(L, L), format='csc')
            Z = (W + D).tocsc()  # Ensure CSC format for spsolve
            z = spsolve(Z, w * spectrum)
            w = self.asls_p * (spectrum > z) + (1 - self.asls_p) * (spectrum < z)
        
        return spectrum - z
    
    def _normalize(self, spectrum: np.ndarray) -> np.ndarray:
        """Apply normalization to spectrum."""
        if self.normalization_method == 'minmax':
            min_val = spectrum.min()
            max_val = spectrum.max()
            if max_val - min_val > 0:
                return (spectrum - min_val) / (max_val - min_val)
            return spectrum
        else:  # vector normalization
            norm = np.linalg.norm(spectrum)
            if norm > 0:
                return spectrum / norm
            return spectrum
    
    def process_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Process a single spectrum through the pipeline.
        
        Args:
            spectrum: 1D array of intensity values
        
        Returns:
            Preprocessed spectrum
        """
        processed = spectrum.copy().astype(float)
        
        if self.use_cosmic_ray_removal:
            processed = self._whitaker_hayes_despike(processed)
        
        if self.use_denoising:
            processed = self._savgol_denoise(processed)
        
        if self.use_baseline_correction:
            processed = self._asls_baseline(processed)
        
        if self.use_normalization:
            processed = self._normalize(processed)
        
        return processed
    
    def process_batch(self, spectra: np.ndarray) -> np.ndarray:
        """
        Process multiple spectra through the pipeline.
        
        Args:
            spectra: 2D array of shape (n_samples, n_wavenumbers)
        
        Returns:
            Preprocessed spectra array
        """
        processed = np.zeros_like(spectra, dtype=float)
        
        for i in range(spectra.shape[0]):
            processed[i] = self.process_spectrum(spectra[i])
        
        return processed
    
    def __call__(self, spectra: Union[np.ndarray, List]) -> np.ndarray:
        """
        Apply preprocessing to spectra.
        
        Args:
            spectra: Single spectrum (1D) or batch of spectra (2D)
        
        Returns:
            Preprocessed spectra
        """
        spectra = np.atleast_2d(spectra)
        return self.process_batch(spectra)
    
    def get_config(self) -> dict:
        """Return pipeline configuration as dictionary."""
        return {
            'use_cosmic_ray_removal': self.use_cosmic_ray_removal,
            'use_denoising': self.use_denoising,
            'use_baseline_correction': self.use_baseline_correction,
            'use_normalization': self.use_normalization,
            'normalization_method': self.normalization_method,
            'savgol_window': self.savgol_window,
            'savgol_polyorder': self.savgol_polyorder,
            'asls_lam': self.asls_lam,
            'asls_p': self.asls_p,
            'asls_niter': self.asls_niter
        }


def get_default_pipeline() -> PreprocessingPipeline:
    """
    Get the default preprocessing pipeline configuration.
    
    Returns:
        PreprocessingPipeline with recommended settings
    """
    return PreprocessingPipeline(
        use_cosmic_ray_removal=True,
        use_denoising=True,
        use_baseline_correction=True,
        use_normalization=True,
        normalization_method='minmax',
        savgol_window=9,
        savgol_polyorder=3,
        asls_lam=1e5,
        asls_p=0.01
    )


def visualize_preprocessing(
    raw_spectrum: np.ndarray,
    processed_spectrum: np.ndarray,
    wavenumbers: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
):
    """
    Visualize before/after preprocessing comparison.
    
    Args:
        raw_spectrum: Original spectrum
        processed_spectrum: Preprocessed spectrum
        wavenumbers: Wavenumber axis (optional)
        save_path: Path to save figure (optional)
    """
    import matplotlib.pyplot as plt
    
    x = wavenumbers if wavenumbers is not None else np.arange(len(raw_spectrum))
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    axes[0].plot(x, raw_spectrum, 'b-', linewidth=0.8, label='Raw')
    axes[0].set_ylabel('Intensity (a.u.)')
    axes[0].set_title('Raw Spectrum')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(x, processed_spectrum, 'g-', linewidth=0.8, label='Preprocessed')
    axes[1].set_xlabel('Wavenumber (cm⁻¹)')
    axes[1].set_ylabel('Intensity (normalized)')
    axes[1].set_title('Preprocessed Spectrum')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved preprocessing comparison to: {save_path}")
    
    plt.close()


if __name__ == "__main__":
    # Test preprocessing pipeline
    import matplotlib.pyplot as plt
    
    # Generate test spectrum with noise and baseline
    np.random.seed(42)
    wavenumbers = np.linspace(400, 1800, 1000)
    
    # Create synthetic spectrum with peaks
    spectrum = np.zeros_like(wavenumbers)
    peaks = [(600, 0.8, 15), (900, 1.0, 20), (1200, 0.6, 12), (1500, 0.9, 18)]
    for center, amp, width in peaks:
        spectrum += amp * np.exp(-((wavenumbers - center) ** 2) / (2 * width ** 2))
    
    # Add baseline drift
    baseline = 0.3 * np.sin(wavenumbers / 300) + 0.1 * (wavenumbers - 400) / 1400
    spectrum += baseline
    
    # Add noise
    spectrum += np.random.normal(0, 0.02, len(spectrum))
    
    # Add cosmic ray spike
    spectrum[500] += 1.5
    
    # Process
    pipeline = get_default_pipeline()
    processed = pipeline.process_spectrum(spectrum)
    
    print("Preprocessing Pipeline Configuration:")
    for key, value in pipeline.get_config().items():
        print(f"  {key}: {value}")
    
    # Visualize
    visualize_preprocessing(
        spectrum, 
        processed, 
        wavenumbers,
        save_path='results/plots/preprocessing_demo.png'
    )
    
    print("\nPreprocessing test completed successfully!")
