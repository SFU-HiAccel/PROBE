# SemBench Environment Setup Guide

## Quick Setup

To set up the environment automatically, run:

```bash
bash scripts/setup_environment.sh
```

This will create a conda environment named `sembench` with Python 3.12 and all required dependencies.

**Custom environment name:**
```bash
bash scripts/setup_environment.sh my_custom_env_name
```

---

## Manual Setup

If you prefer to set up manually, follow these steps:

### 1. Create Conda Environment

```bash
conda create -n sembench python=3.12 -y
```

### 2. Install Conda Packages (Base Scientific Stack)

```bash
conda install -n sembench -c conda-forge -y \
    numpy pandas scikit-learn scipy matplotlib seaborn pillow
```

### 3. Install PyTorch with CUDA Support

```bash
conda install -n sembench -c pytorch -c nvidia \
    pytorch torchvision torchaudio pytorch-cuda=12.4 -y
```

### 4. Install Pip Packages

```bash
conda activate sembench
pip install -r requirements.txt
```

---

## Environment Details

**Environment Name:** `sembench`
**Python Version:** 3.12.12
**Installation Strategy:** Conda for base packages + Pip for specialty packages

### Why This Approach?

1. **Conda packages** provide optimized builds for scientific computing (NumPy, SciPy, PyTorch)
2. **CUDA support** is handled automatically through conda's PyTorch installation
3. **System-specific packages** (lotus, palimpzest, thalamusdb) are only available on PyPI

---

## Installed Packages

### Conda-Installed (Optimized Scientific Stack)

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.12.12 | Base interpreter |
| NumPy | 2.3.4 | Numerical computing |
| Pandas | 2.3.3 | Data manipulation |
| PyTorch | 2.6.0 | Deep learning |
| Scikit-learn | 1.7.2 | Machine learning |
| SciPy | 1.16.3 | Scientific computing |
| Matplotlib | 3.10.7 | Plotting |
| Seaborn | 0.13.2 | Statistical visualization |

**Plus:** CUDA 12.4 libraries (libcublas, libcufft, libcurand, etc.)

### Pip-Installed (System-Specific & Dependencies)

#### Core System Packages

| System | Package | Version |
|--------|---------|---------|
| **Lotus** | lotus-ai | 1.1.3 |
| **Palimpzest** | palimpzest | 0.8.2 |
| **ThalamusDB** | thalamusdb | 0.1.15 |
| **BigQuery** | google-cloud-bigquery | 3.38.0 |

#### Google Cloud Packages

- `google-cloud-bigquery` 3.38.0
- `google-cloud-storage` 3.5.0
- `google-cloud-bigquery-storage` 2.34.0
- `db-dtypes` 1.4.3

#### ML & AI Libraries

- `transformers` 4.57.1
- `sentence-transformers` 3.4.1
- `openai` 2.7.1
- `litellm` 1.75.9
- `anthropic` 0.72.0
- `chromadb` 1.3.4
- `faiss-cpu` 1.12.0

#### Supporting Libraries

- `smolagents` 1.22.0 (for palimpzest)
- `cdlib` 0.4.0 (for evaluation)
- `gradio` 5.49.1 (for UI)
- `datasets` 4.4.1 (for data loading)
- `evaluate` 0.4.6 (for metrics)
- `opencv-python` 4.12.0.88 (for image processing)
- And 100+ other dependencies...

---

## Usage

### Activate Environment

```bash
conda activate sembench
```

### Run Benchmarks

```bash
# Test Lotus
python3 src/run.py --systems lotus --use-cases movie --queries 1 --model gemini-2.5-flash --scale-factor 2000

# Test Palimpzest
python3 src/run.py --systems palimpzest --use-cases movie --queries 1 --model gemini-2.5-flash --scale-factor 2000

# Test ThalamusDB
python3 src/run.py --systems thalamusdb --use-cases movie --queries 1 --model gemini-2.5-flash --scale-factor 2000

# Test BigQuery
python3 src/run.py --systems bigquery --use-cases movie --queries 1 --model gemini-2.5-flash --scale-factor 2000
```

### Deactivate Environment

```bash
conda deactivate
```

---

## Key Dependencies Discovered During Setup

The following packages were discovered to be required but were not initially in requirements.txt:

1. **`google-cloud-storage`** - Required for BigQuery storage operations
2. **`google-cloud-bigquery-storage`** - Optimizes BigQuery data fetching
3. **`db-dtypes`** - Required for BigQuery data type handling
4. **`cdlib`** - Required for evaluation metrics
5. **`smolagents`** - Required by palimpzest
6. **`prettytable`** - Palimpzest dependency
7. **`psutil`** - Palimpzest dependency
8. **`PyLD`** - Palimpzest dependency
9. **`tabulate`** - Palimpzest dependency
10. **`together`** - Palimpzest dependency

All of these have been added to the final `requirements.txt`.