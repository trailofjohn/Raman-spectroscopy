"""
Custom exceptions for the Raman Spectroscopy Pipeline.

Provides clear exception taxonomy for data loading and processing errors.
"""


class DataLoadError(Exception):
    """
    Raised when dataset loading fails and synthetic fallback is not allowed.
    
    This exception ensures that silent substitution of synthetic data
    cannot occur without explicit opt-in.
    """
    def __init__(self, dataset_name: str, reason: str, allow_synthetic_hint: bool = True):
        self.dataset_name = dataset_name
        self.reason = reason
        hint = " Set allow_synthetic=True to use synthetic fallback." if allow_synthetic_hint else ""
        super().__init__(f"Failed to load dataset '{dataset_name}': {reason}{hint}")


class PreprocessingError(Exception):
    """Raised when preprocessing fails on a spectrum."""
    pass


class CacheInvalidError(Exception):
    """Raised when cached data is stale or incompatible."""
    def __init__(self, cache_path: str, reason: str):
        self.cache_path = cache_path
        super().__init__(f"Cache invalid at '{cache_path}': {reason}")


class SpectralAlignmentError(Exception):
    """Raised when spectra have incompatible wavenumber grids."""
    pass
