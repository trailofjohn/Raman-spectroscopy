"""
Evaluation Metrics and Visualization Utilities.

This module provides comprehensive evaluation functions for both
classification and regression tasks in spectroscopic analysis.

Author: Spectroscopic ML Pipeline
Date: 2024-12
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import (
    # Classification metrics
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    # Regression metrics
    r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
)


# =============================================================================
# REGRESSION METRICS
# =============================================================================

def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics.
    
    Args:
        y_true: True values (n_samples,) or (n_samples, n_outputs)
        y_pred: Predicted values
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'R2': r2_score(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'Explained_Variance': explained_variance_score(y_true, y_pred)
    }
    
    # Per-output metrics if multivariate
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        for i in range(y_true.shape[1]):
            metrics[f'R2_Component_{i+1}'] = r2_score(y_true[:, i], y_pred[:, i])
            metrics[f'RMSE_Component_{i+1}'] = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
    
    return metrics


def plot_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = 'Parity Plot',
    save_path: Optional[str] = None,
    component_names: Optional[List[str]] = None
):
    """
    Create parity plot (predicted vs actual) for regression evaluation.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save figure
        component_names: Names for each output component
    """
    y_true = np.atleast_2d(y_true)
    y_pred = np.atleast_2d(y_pred)
    
    if y_true.shape[0] < y_true.shape[1]:
        y_true = y_true.T
        y_pred = y_pred.T
    
    n_components = y_true.shape[1]
    
    if component_names is None:
        component_names = [f'Component {i+1}' for i in range(n_components)]
    
    cols = min(4, n_components)
    rows = (n_components + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    
    for i in range(n_components):
        row, col = i // cols, i % cols
        ax = axes[row, col]
        
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5, s=20, c='steelblue')
        
        # Perfect prediction line
        min_val = min(y_true[:, i].min(), y_pred[:, i].min()) * 0.95
        max_val = max(y_true[:, i].max(), y_pred[:, i].max()) * 1.05
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        # Calculate R²
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        
        ax.set_xlabel('True Value')
        ax.set_ylabel('Predicted Value')
        ax.set_title(f'{component_names[i]}\n$R^2$ = {r2:.4f}, RMSE = {rmse:.4f}')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_components, rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].set_visible(False)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Create residual distribution plot.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        save_path: Path to save figure
    """
    residuals = (y_true - y_pred).flatten()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Residual histogram
    axes[0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Residual')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Residual Distribution')
    axes[0].grid(True, alpha=0.3)
    
    # QQ-like residual plot
    axes[1].scatter(y_pred.flatten(), residuals, alpha=0.5, s=20)
    axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Predicted Value')
    axes[1].set_ylabel('Residual')
    axes[1].set_title('Residuals vs Predicted')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


# =============================================================================
# CLASSIFICATION METRICS
# =============================================================================

def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Class probabilities (optional, for AUC)
        average: Averaging method for multi-class
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'Recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'F1_Score': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    if y_proba is not None:
        try:
            n_classes = y_proba.shape[1]
            if n_classes == 2:
                metrics['AUC'] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                metrics['AUC_OvR'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
        except ValueError:
            pass  # Skip if AUC cannot be calculated
    
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    save_path: Optional[str] = None,
    title: str = 'Confusion Matrix'
):
    """
    Create and plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        normalize: Whether to normalize
        save_path: Path to save figure
        title: Plot title
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt=fmt,
        xticklabels=class_names, yticklabels=class_names,
        cmap='Blues', ax=ax, cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()
    
    return cm


def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Plot ROC curves for multi-class classification.
    
    Args:
        y_true: True labels
        y_proba: Class probabilities (n_samples, n_classes)
        class_names: List of class names
        save_path: Path to save figure
    """
    n_classes = y_proba.shape[1]
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for i in range(n_classes):
        y_binary = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_binary, y_proba[:, i])
        auc = roc_auc_score(y_binary, y_proba[:, i])
        ax.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Get classification report as DataFrame.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    
    Returns:
        Classification report as DataFrame
    """
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    return pd.DataFrame(report).transpose()


# =============================================================================
# SUMMARY GENERATION
# =============================================================================

def generate_summary_csv(
    regression_results: Optional[Dict[str, Dict]] = None,
    classification_results: Optional[Dict[str, Dict]] = None,
    save_path: str = 'results/summary_results.csv'
) -> pd.DataFrame:
    """
    Generate comprehensive summary CSV combining all results.
    
    Args:
        regression_results: Dict of {model_name: metrics_dict}
        classification_results: Dict of {model_name: metrics_dict}
        save_path: Path to save CSV
    
    Returns:
        Combined results DataFrame
    """
    rows = []
    
    if regression_results:
        for model_name, metrics in regression_results.items():
            row = {'Task': 'Regression', 'Model': model_name}
            row.update(metrics)
            rows.append(row)
    
    if classification_results:
        for model_name, metrics in classification_results.items():
            row = {'Task': 'Classification', 'Model': model_name}
            row.update(metrics)
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"Summary saved to: {save_path}")
    
    return df


if __name__ == "__main__":
    # Test evaluation functions
    print("Testing evaluation module...")
    
    # Test regression metrics
    y_true_reg = np.random.rand(100, 4)
    y_pred_reg = y_true_reg + np.random.randn(100, 4) * 0.1
    
    reg_metrics = regression_metrics(y_true_reg, y_pred_reg)
    print("\nRegression Metrics:")
    for k, v in reg_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Test classification metrics
    y_true_cls = np.random.randint(0, 5, 100)
    y_pred_cls = np.where(np.random.rand(100) > 0.1, y_true_cls, np.random.randint(0, 5, 100))
    
    cls_metrics = classification_metrics(y_true_cls, y_pred_cls)
    print("\nClassification Metrics:")
    for k, v in cls_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print("\nEvaluation module test completed!")
