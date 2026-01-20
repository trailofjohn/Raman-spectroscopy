"""
Chemometrics Models for Concentration Estimation.

This module implements PLS and SVM regression models for
predicting chemical concentrations from Raman spectra.

Author: Spectroscopic ML Pipeline
Date: 2024-12
"""

import numpy as np
import joblib
from typing import Tuple, Optional, Dict, Any
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings


class PLSModel:
    """
    Partial Least Squares Regression model for concentration prediction.
    
    PLS is well-suited for spectroscopic data where the number of features
    (wavenumbers) often exceeds the number of samples, and features are
    highly correlated.
    
    Attributes:
        n_components (int): Number of PLS components
        model: Fitted PLSRegression model
        scaler: StandardScaler for input normalization
    """
    
    def __init__(self, n_components: int = 10, scale_input: bool = True):
        """
        Initialize PLS model.
        
        Args:
            n_components: Number of PLS latent variables
            scale_input: Whether to standardize input features
        """
        self.n_components = n_components
        self.scale_input = scale_input
        self.model = PLSRegression(n_components=n_components)
        self.scaler = StandardScaler() if scale_input else None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PLSModel':
        """
        Fit PLS model to training data.
        
        Args:
            X: Spectra array of shape (n_samples, n_wavenumbers)
            y: Concentration array of shape (n_samples, n_components)
        
        Returns:
            Self for method chaining
        """
        if self.scaler:
            X = self.scaler.fit_transform(X)
        
        # Adjust n_components if needed
        max_components = min(X.shape[0], X.shape[1], self.n_components)
        if max_components < self.n_components:
            warnings.warn(f"Reducing n_components from {self.n_components} to {max_components}")
            self.model = PLSRegression(n_components=max_components)
        
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict concentrations from spectra.
        
        Args:
            X: Spectra array of shape (n_samples, n_wavenumbers)
        
        Returns:
            Predicted concentrations of shape (n_samples, n_components)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.scaler:
            X = self.scaler.transform(X)
        
        return self.model.predict(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Calculate regression metrics.
        
        Args:
            X: Test spectra
            y: True concentrations
        
        Returns:
            Dictionary of metrics (R², RMSE, MAE)
        """
        y_pred = self.predict(X)
        
        return {
            'r2': r2_score(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred)
        }
    
    def cross_validate(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        cv: int = 5
    ) -> Dict[str, Tuple[float, float]]:
        """
        Perform cross-validation.
        
        Args:
            X: Spectra array
            y: Concentration array
            cv: Number of folds
        
        Returns:
            Dictionary with mean and std of R² scores
        """
        if self.scaler:
            X = self.scaler.fit_transform(X)
        
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        r2_scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = PLSRegression(n_components=self.n_components)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            r2_scores.append(r2_score(y_val, y_pred))
        
        return {
            'r2_mean': np.mean(r2_scores),
            'r2_std': np.std(r2_scores)
        }
    
    def save(self, filepath: str):
        """Save model to file."""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'n_components': self.n_components,
            'is_fitted': self.is_fitted
        }, filepath)
        print(f"PLS model saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'PLSModel':
        """Load model from file."""
        data = joblib.load(filepath)
        instance = cls(n_components=data['n_components'])
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.is_fitted = data['is_fitted']
        return instance


class SVMModel:
    """
    Support Vector Machine regression model for concentration prediction.
    
    Uses MultiOutputRegressor to handle multi-component predictions.
    
    Attributes:
        kernel (str): SVM kernel type
        C (float): Regularization parameter
        gamma (str): Kernel coefficient
    """
    
    def __init__(
        self, 
        kernel: str = 'rbf',
        C: float = 100.0,
        gamma: str = 'scale',
        epsilon: float = 0.1,
        scale_input: bool = True
    ):
        """
        Initialize SVM model.
        
        Args:
            kernel: Kernel type ('rbf', 'linear', 'poly')
            C: Regularization parameter
            gamma: Kernel coefficient
            epsilon: Epsilon in epsilon-SVR model
            scale_input: Whether to standardize input features
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.epsilon = epsilon
        self.scale_input = scale_input
        
        base_svr = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon)
        self.model = MultiOutputRegressor(base_svr)
        self.scaler = StandardScaler() if scale_input else None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVMModel':
        """
        Fit SVM model to training data.
        
        Args:
            X: Spectra array of shape (n_samples, n_wavenumbers)
            y: Concentration array of shape (n_samples, n_components)
        
        Returns:
            Self for method chaining
        """
        if self.scaler:
            X = self.scaler.fit_transform(X)
        
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict concentrations from spectra.
        
        Args:
            X: Spectra array of shape (n_samples, n_wavenumbers)
        
        Returns:
            Predicted concentrations of shape (n_samples, n_components)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.scaler:
            X = self.scaler.transform(X)
        
        return self.model.predict(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Calculate regression metrics.
        
        Args:
            X: Test spectra
            y: True concentrations
        
        Returns:
            Dictionary of metrics (R², RMSE, MAE)
        """
        y_pred = self.predict(X)
        
        return {
            'r2': r2_score(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred)
        }
    
    def save(self, filepath: str):
        """Save model to file."""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'is_fitted': self.is_fitted
        }, filepath)
        print(f"SVM model saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'SVMModel':
        """Load model from file."""
        data = joblib.load(filepath)
        instance = cls(
            kernel=data['kernel'],
            C=data['C'],
            gamma=data['gamma'],
            epsilon=data['epsilon']
        )
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.is_fitted = data['is_fitted']
        return instance


def compare_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Train and compare PLS and SVM models.
    
    Args:
        X_train: Training spectra
        y_train: Training concentrations
        X_test: Test spectra
        y_test: Test concentrations
    
    Returns:
        Dictionary of model results
    """
    results = {}
    
    # PLS
    print("Training PLS model...")
    pls = PLSModel(n_components=10)
    pls.fit(X_train, y_train)
    results['PLS'] = pls.score(X_test, y_test)
    print(f"  R²: {results['PLS']['r2']:.4f}, RMSE: {results['PLS']['rmse']:.4f}")
    
    # SVM
    print("Training SVM model...")
    svm = SVMModel(kernel='rbf', C=100)
    svm.fit(X_train, y_train)
    results['SVM'] = svm.score(X_test, y_test)
    print(f"  R²: {results['SVM']['r2']:.4f}, RMSE: {results['SVM']['rmse']:.4f}")
    
    return results, pls, svm


if __name__ == "__main__":
    # Test with synthetic data
    import sys
    sys.path.insert(0, '.')
    from data.synthetic_generator import generate_training_dataset
    
    # Generate data
    dataset = generate_training_dataset(
        n_train=500,
        n_test=100,
        n_components=4,
        noise_level=0.02
    )
    
    # Compare models
    results, pls, svm = compare_models(
        dataset['X_train'],
        dataset['y_train'],
        dataset['X_test'],
        dataset['y_test']
    )
    
    print("\n" + "=" * 50)
    print("Model Comparison Results:")
    print("=" * 50)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
