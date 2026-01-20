"""
Training Script for Chemometrics Regression Models.

Trains PLS and SVM models for concentration prediction using
synthetic mixture Raman spectra.

Usage:
    python training/train_regressor.py

Author: Spectroscopic ML Pipeline
Date: 2024-12
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.synthetic_generator import generate_training_dataset
from preprocessing.pipeline import PreprocessingPipeline
from models.chemometrics import PLSModel, SVMModel


def create_parity_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    component_names: list,
    model_name: str,
    save_path: str
):
    """
    Create parity plot (predicted vs actual) for regression evaluation.
    
    Args:
        y_true: True concentration values (n_samples, n_components)
        y_pred: Predicted concentration values
        component_names: Names of each component
        model_name: Name of the model for title
        save_path: Path to save the figure
    """
    n_components = y_true.shape[1]
    fig, axes = plt.subplots(1, n_components, figsize=(4 * n_components, 4))
    
    if n_components == 1:
        axes = [axes]
    
    for i, (ax, name) in enumerate(zip(axes, component_names)):
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        
        # Calculate R² for this component
        ss_res = np.sum((y_true[:, i] - y_pred[:, i]) ** 2)
        ss_tot = np.sum((y_true[:, i] - y_true[:, i].mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        ax.set_xlabel('True Concentration')
        ax.set_ylabel('Predicted Concentration')
        ax.set_title(f'{name}\n$R^2$ = {r2:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name} - Parity Plots', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved parity plot: {save_path}")


def main():
    """Main training script for regression models."""
    print("=" * 60)
    print("CHEMOMETRICS REGRESSION MODEL TRAINING")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    N_TRAIN = 800
    N_TEST = 200
    N_COMPONENTS = 4
    N_WAVENUMBERS = 1000
    NOISE_LEVEL = 0.02
    RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    
    # Create directories
    os.makedirs(os.path.join(RESULTS_DIR, 'models'), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'plots'), exist_ok=True)
    
    # Generate synthetic data
    print("\n[1/5] Generating synthetic mixture spectra...")
    dataset = generate_training_dataset(
        n_train=N_TRAIN,
        n_test=N_TEST,
        n_components=N_COMPONENTS,
        n_wavenumbers=N_WAVENUMBERS,
        wavenumber_range=(400, 1800),
        noise_level=NOISE_LEVEL,
        seed=42
    )
    
    # Preprocess data
    print("\n[2/5] Applying preprocessing pipeline...")
    pipeline = PreprocessingPipeline(
        use_cosmic_ray_removal=True,
        use_denoising=True,
        use_baseline_correction=True,
        use_normalization=True,
        normalization_method='minmax'
    )
    
    X_train = pipeline.process_batch(dataset['X_train'])
    X_test = pipeline.process_batch(dataset['X_test'])
    y_train = dataset['y_train']
    y_test = dataset['y_test']
    
    print(f"  Training set: {X_train.shape}")
    print(f"  Test set: {X_test.shape}")
    
    # Store results
    results = []
    
    # Train PLS Model
    print("\n[3/5] Training PLS regression model...")
    pls_model = PLSModel(n_components=10, scale_input=True)
    pls_model.fit(X_train, y_train)
    
    # PLS Cross-validation
    cv_results = pls_model.cross_validate(X_train, y_train, cv=5)
    print(f"  5-Fold CV R²: {cv_results['r2_mean']:.4f} ± {cv_results['r2_std']:.4f}")
    
    # PLS Test evaluation
    pls_metrics = pls_model.score(X_test, y_test)
    print(f"  Test R²: {pls_metrics['r2']:.4f}")
    print(f"  Test RMSE: {pls_metrics['rmse']:.4f}")
    print(f"  Test MAE: {pls_metrics['mae']:.4f}")
    
    # Save PLS model
    pls_path = os.path.join(RESULTS_DIR, 'models', 'pls_model.joblib')
    pls_model.save(pls_path)
    
    # PLS parity plot
    y_pred_pls = pls_model.predict(X_test)
    create_parity_plot(
        y_test, y_pred_pls,
        dataset['component_names'],
        'PLS Regression',
        os.path.join(RESULTS_DIR, 'plots', 'parity_plot_pls.png')
    )
    
    results.append({
        'Model': 'PLS',
        'CV_R2_Mean': cv_results['r2_mean'],
        'CV_R2_Std': cv_results['r2_std'],
        'Test_R2': pls_metrics['r2'],
        'Test_RMSE': pls_metrics['rmse'],
        'Test_MAE': pls_metrics['mae']
    })
    
    # Train SVM Model
    print("\n[4/5] Training SVM regression model...")
    svm_model = SVMModel(kernel='rbf', C=100, gamma='scale', scale_input=True)
    svm_model.fit(X_train, y_train)
    
    # SVM Test evaluation
    svm_metrics = svm_model.score(X_test, y_test)
    print(f"  Test R²: {svm_metrics['r2']:.4f}")
    print(f"  Test RMSE: {svm_metrics['rmse']:.4f}")
    print(f"  Test MAE: {svm_metrics['mae']:.4f}")
    
    # Save SVM model
    svm_path = os.path.join(RESULTS_DIR, 'models', 'svm_model.joblib')
    svm_model.save(svm_path)
    
    # SVM parity plot
    y_pred_svm = svm_model.predict(X_test)
    create_parity_plot(
        y_test, y_pred_svm,
        dataset['component_names'],
        'SVM Regression',
        os.path.join(RESULTS_DIR, 'plots', 'parity_plot_svm.png')
    )
    
    results.append({
        'Model': 'SVM',
        'CV_R2_Mean': None,  # SVM CV is expensive
        'CV_R2_Std': None,
        'Test_R2': svm_metrics['r2'],
        'Test_RMSE': svm_metrics['rmse'],
        'Test_MAE': svm_metrics['mae']
    })
    
    # Save pure component spectra plot
    print("\n[5/5] Saving visualization of pure components...")
    fig, ax = plt.subplots(figsize=(12, 6))
    wavenumbers = dataset['wavenumbers']
    for i, (spectrum, name) in enumerate(zip(dataset['pure_components'], dataset['component_names'])):
        ax.plot(wavenumbers, spectrum + i * 0.3, linewidth=1, label=name)
    ax.set_xlabel('Wavenumber (cm⁻¹)')
    ax.set_ylabel('Intensity (offset for clarity)')
    ax.set_title('Pure Component Reference Spectra')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, 'plots', 'pure_components.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save summary results
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(RESULTS_DIR, 'regression_results.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nResults saved to: {results_csv_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results_df


if __name__ == "__main__":
    results = main()
