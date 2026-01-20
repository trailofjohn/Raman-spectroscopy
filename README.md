# Raman Spectroscopy Analysis Pipeline

A comprehensive Python pipeline for Raman spectrum analysis combining classical chemometrics (PLS, SVM) with deep learning (1D-CNN). The system supports **compound identification** (classification) and **concentration estimation** (regression) with fully reproducible results.

## Key Results

| Task | Model | Accuracy/R² | Dataset |
|------|-------|-------------|---------|
| Wheat Classification | ResidualCNN | **86.47%** | 53,134 spectra, 4 classes |
| Mineral Family Classification | SVM | **58.86%** | 5,784 RRUFF spectra, 11 families |
| Concentration Estimation | PLS/SVM | Synthetic verified | Beer-Lambert mixtures |

> [!NOTE]
> **Key Finding:** Classical methods (SVM, PLS) outperform CNNs on reference library data (RRUFF), while CNNs excel on large homogeneous datasets (wheat). This aligns with spectroscopy physics—Raman peaks occur at fixed wavenumbers, violating CNN's translation-invariance assumption.

---

## Project Structure

```
raman_spectroscopy_pipeline/
├── data/                           # Data acquisition module
│   ├── data_loader.py              # RamanSpy dataset loading (wheat, RRUFF)
│   ├── synthetic_generator.py      # Synthetic mixture spectra generation
│   ├── rruff_local_loader.py       # Local RRUFF .txt file parser
│   ├── cache_manager.py            # Preprocessing cache with invalidation
│   └── exceptions.py               # Custom exception classes
│
├── preprocessing/                  # Spectrum preprocessing
│   └── pipeline.py                 # Pipeline: Whitaker-Hayes → SavGol → ASLS → MinMax
│
├── models/                         # ML models
│   ├── chemometrics.py             # PLS & SVM regression/classification
│   └── cnn_classifier.py           # LightweightCNN, ResidualCNN (PyTorch)
│
├── training/                       # Training scripts
│   ├── train_wheat_rigorous.py     # Wheat classification (full rigor)
│   ├── train_rruff_families.py     # Mineral family classification
│   ├── train_regressor.py          # PLS/SVM concentration estimation
│   ├── train_uncertainty.py        # MC Dropout uncertainty quantification
│   ├── rruff_retrieval.py          # SAD-based spectral library matching
│   ├── spectral_decomposition.py   # NMF unmixing
│   └── [other training scripts]    # Additional variants
│
├── evaluation/                     # Evaluation utilities
│   └── metrics.py                  # R², RMSE, confusion matrix, ROC curves
│
├── utils/                          # Utilities
│   └── reproducibility.py          # Seed locking for PyTorch, NumPy, CUDA
│
├── results/                        # Output directory (generated)
│   ├── models/                     # Saved trained models (.pth, .joblib)
│   ├── plots/                      # Generated visualizations
│   └── [task_name]/                # Per-task results with provenance.json
│
├── rruff/                          # Local RRUFF database (76K+ files)
│
├── gradio_app.py                   # Web interface (http://localhost:7860)
├── cli_tool.py                     # Command-line interface
├── requirements.txt                # Python dependencies
├── setup_and_run.sh                # Setup script
└── PROJECT_REPORT.md              # Scientific documentation
```

---

## Quick Start

### 1. Clone and Setup Environment

```bash
# Clone repository
git clone <repository-url>
cd raman_spectroscopy_pipeline

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate 

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dependencies

```
ramanspy>=0.2.0          # Raman spectroscopy toolkit
torch>=2.0.0             # Deep learning
scikit-learn>=1.3.0      # Classical ML
gradio>=4.0.0            # Web interface
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0
joblib>=1.3.0
tqdm>=4.65.0
```

---

## Training Models

### Wheat Classification (CNN)

```bash
python training/train_wheat_rigorous.py
```
- **Output:** `results/wheat_rigorous/` (model, confusion matrix, provenance)
- **Expected:** ~86% accuracy with calibrated probabilities

### Mineral Family Classification (SVM)

```bash
python training/train_rruff_families.py
```
- **Output:** `results/rruff_families/`
- **Note:** SVM outperforms CNN for spectral data with fixed peak positions

### Concentration Regression (PLS/SVM)

```bash
python training/train_regressor.py
```
- **Output:** `results/` (parity plots, regression metrics)

---

## Using the Pipeline

### Web Interface (Gradio)

```bash
python gradio_app.py
# Visit http://localhost:7860
```

**Features:**
- Upload RRUFF format spectra (.txt)
- Wheat classification / Mineral family classification
- Concentration regression (PLS/SVM)
- Live spectrum visualization

### Command Line Interface

```bash
# Wheat classification
python cli_tool.py --input spectrum.csv --task wheat

# Mineral family classification
python cli_tool.py --input spectrum.csv --task mineral

# Concentration regression
python cli_tool.py --input spectrum.csv --task regress --model pls
python cli_tool.py --input spectrum.csv --task regress --model svm

# Help
python cli_tool.py --help
```

---

## Reproducibility

All training is fully deterministic:

```python
from utils.reproducibility import set_all_seeds
set_all_seeds(42)  # PyTorch, NumPy, Python random, CUDA
```

Every output includes `provenance.json` with:
- Dataset source (real/synthetic)
- Preprocessing config hash
- Train/val/test split indices
- Timestamp and code version

---

## Preprocessing Pipeline

```
Raw Spectrum → Cosmic Ray Removal → Savitzky-Golay Smoothing → ASLS Baseline → MinMax Normalization
```

| Step | Method | Parameters |
|------|--------|------------|
| Spike Removal | Whitaker-Hayes | threshold=6.0 |
| Denoising | Savitzky-Golay | window=9, polyorder=3 |
| Baseline | ASLS | λ=1e5, p=0.01 |
| Normalization | Min-Max | 0-1 range |

---

## Essential vs Non-Essential Files

### Essential Files

| Category | Files | Purpose |
|----------|-------|---------|
| **Core Modules** | `data/`, `preprocessing/`, `models/`, `evaluation/`, `utils/` | Pipeline logic |
| **Training** | `training/*.py` | Model training scripts |
| **Interfaces** | `gradio_app.py`, `cli_tool.py` | User-facing tools |
| **Config** | `requirements.txt` | Dependencies |
| **Docs** | `README.md`, `PROJECT_REPORT.md`, `FILE_NOTES.md` | Documentation |
| **Models** | `results/models/*.pth`, `*.joblib` | Trained weights |
| **RRUFF Data** | `rruff/` directory | Mineral spectra database |

### Non-Essential Files (Can Be Removed)

| File Pattern | Size | Reason |
|--------------|------|--------|
| `Data (1).mat` through `Data (5).mat` | 662 MB each | Duplicate data files |
| `Data.mat*.tmp` | Various | Temporary/incomplete files |
| `ILSdata (1).csv` | 10 MB | Duplicate CSV |
| `.venv/` | Variable | Virtual environment (recreatable) |
| `__pycache__/` directories | Variable | Python cache (regenerated) |
| `data/cache/*.npz` | Variable | Preprocessing cache (regenerated) |

**Total removable space:** ~4 GB (5 duplicate .mat + temp files)

### Cleanup Command

```bash
# Remove duplicates and temp files (Windows PowerShell)
Remove-Item "Data (*.mat"
Remove-Item "Data.mat*.tmp"
Remove-Item "ILSdata (1).csv"

# Remove Python cache
Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
```

---

## Running Tests

The pipeline includes self-test capability in each module:

```bash
# Test data loading
python -m data.data_loader

# Test preprocessing
python -m preprocessing.pipeline

# Test models
python -m models.chemometrics
python -m models.cnn_classifier

# Test evaluation metrics
python -m evaluation.metrics
```

---

## Scientific Notes

### When to Use Each Model

| Condition | Recommended Model |
|-----------|-------------------|
| Large homogeneous dataset (>10K samples) | CNN |
| Reference library with few samples/class | SVM, PLS-DA |
| Fixed-position spectral features | Avoid convolutions |
| Open-set identification | SAD retrieval |

### RRUFF Database Notes

- **Invalid:** 1,415-class mineral identity classification (~3.7 samples/class)
- **Valid:** ~10 chemical family classification
- **Better:** Spectral Angle Distance (SAD) retrieval for identification

---

## References

1. Eilers, P.H.C. & Boelens, H.F.M. (2005). Baseline Correction with Asymmetric Least Squares Smoothing.
2. RRUFF Project: https://rruff.info/
3. RamanSpy Documentation: https://ramanspy.readthedocs.io/

---

## License

[Add your license here]

---

*Last Updated: January 2026*  
*Version: 1.0*
