# Development Environment Setup

This document describes the software environment configuration for Ubuntu 24.04 LTS (Noble) with dual Python support for Python 3.12 and Python 3.14.

## System Information

- **OS**: Ubuntu 24.04 LTS (Noble Numbat)
- **Kernel**: Linux 6.14 HWE
- **GPU**: NVIDIA (Driver 535.274.02)

## Python Environments

This setup maintains two Python environments to support different package compatibility requirements:

### Python 3.12 Environment
Primary environment for packages requiring mature ecosystem support, particularly TensorFlow.

### Python 3.14 Environment  
Testing environment for upcoming Ubuntu 26.04 LTS compatibility.

## Key Conda Packages

- **Ipopt**: 3.14.19 (Linux x86_64) - Interior Point Optimizer
- **ASL**: 20231111 - AMPL Solver Library

## Ipopt with ASL installation instructions
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh
conda install -c conda-forge ipopt
```

## Python 3.12 Packages

### Machine Learning & Deep Learning
- **TensorFlow**: 2.20.0 (with TensorBoard 2.20.0)
- **PyTorch**: 2.10.0 (with Triton 3.6.0)
- **Keras**: 3.13.1
- **scikit-learn**: 1.8.0
- **ONNX**: 1.20.1 (with ONNXRuntime 1.23.2)

### Scientific Computing
- **NumPy**: 1.26.4
- **SciPy**: 1.16.3
- **pandas**: 2.3.3
- **matplotlib**: 3.10.8
- **seaborn**: 0.13.2
- **tables** (PyTables): 3.9.2
- **numexpr**: 2.9.0
- **Bottleneck**: 1.6.0

### NVIDIA CUDA Libraries
- **CUDA Runtime**: 12.8.90
- **cuDNN**: 9.10.2.21
- **cuBLAS**: 12.8.4.1
- **NCCL**: 2.27.5
- Complete CUDA toolkit including cuFFT, cuRAND, cuSolver, cuSparse

### Optimization & Hyperparameter Tuning
- **Optuna**: 4.7.0 (with OptunaHub 0.4.0)
- **pymoo**: 0.6.1.6 (Multi-objective optimization)
- **scikit-optimize**: 0.10.2
- **Pyomo**: 6.9.5
- **DEAP**: 1.4.3 (Distributed Evolutionary Algorithms)

### Data Processing & Feature Engineering
- **category_encoders**: 2.9.0
- **mrmr-selection**: 0.2.8 (Feature selection)
- **pandas-datareader**: 0.10.0
- **openpyxl**: 3.1.2
- **beautifulsoup4**: 4.12.3
- **lxml**: 5.2.1

### Statistical Analysis
- **statsmodels**: 0.14.6
- **patsy**: 1.0.2

### Development & Testing
- **pytest**: 7.4.4 (with pytest-mock 3.12.0, pytest-forked 1.6.0)
- **GitPython**: 3.1.37
- **PyGithub**: 2.2.0
- **loguru**: 0.7.3
- **rich**: 13.7.1
- **tqdm**: 4.67.1

### Utilities
- **joblib**: 1.3.2
- **psutil**: 5.9.8
- **python-dotenv**: 1.2.1
- **alembic**: 1.18.1 (Database migrations)
- **SQLAlchemy**: 2.0.46

## Python 3.14 Packages

### Key Differences from Python 3.12

The Python 3.14 environment focuses on compatibility testing for the upcoming Ubuntu 26.04 LTS release. Notable differences include:

#### Added Packages
- **aiohttp**: 3.9.1 (with async dependencies: aiosignal, frozenlist, multidict, yarl, propcache)
- **sh**: 2.2.2
- **DateTimeRange**: 2.3.2
- **typepy**: 1.3.4
- **mbstrdecoder**: 1.1.4
- **fonttools**: 4.61.1
- **imagesize**: 1.4.1

#### Version Differences
- **NumPy**: 2.4.1 (vs 1.26.4 in Python 3.12)
- **SciPy**: 1.17.0 (vs 1.16.3)
- **Bottleneck**: 1.3.5 (vs 1.6.0)
- **beautifulsoup4**: 4.13.5 (vs 4.12.3)
- **GitPython**: 3.1.46 (vs 3.1.37)
- **pytest**: 9.0.2 (vs 7.4.4)
- **pytest-mock**: 3.15.1 (vs 3.12.0)
- **pillow**: 12.1.0 (vs 10.2.0)
- **PyQt5**: 5.15.11 (vs 5.15.10)
- **Pygments**: 2.19.2 (vs 2.17.2)
- **packaging**: 25.0 (vs 24.0)

#### Missing Packages (Python 3.14)
The following packages from Python 3.12 are not yet available in Python 3.14:
- TensorFlow ecosystem (tensorflow, tensorboard, tf2onnx)
- absl-py, astunparse, gast, google-pasta, grpcio
- flatbuffers, ml_dtypes
- ONNX ecosystem (onnx, onnxruntime, onnxscript)
- arxiv, feedparser
- Various optimization libraries (doepy, pyDOE, moocore)
- category_encoders, mrmr-selection
- pandas-datareader, jenkspy
- pysubgroup, statsmodels, patsy, seaborn

### Common Packages (Both Environments)
- **PyTorch**: 2.10.0 ✅
- **pandas**: 2.3.3 ✅
- **matplotlib**: 3.10.8 ✅
- **scikit-learn**: 1.8.0 ✅
- **Optuna**: 4.7.0 ✅
- **pymoo**: 0.6.1.6 ✅
- **Pyomo**: 6.9.5 ✅
- All NVIDIA CUDA libraries ✅

## System Packages (Ubuntu)

### Development Tools
- **cmake**: 3.28.3
- **ninja-build**: 1.11.1
- **git**: 2.43.0
- **gh** (GitHub CLI): 2.86.0
- **vim-gtk3**: 9.1.0016

### Python Development
- **python3-dev**: 3.12.3 (system)
- **python3.14-dev**: 3.14.2
- **python3.14-venv**: 3.14.2
- **python3.14-tk**: 3.14.2
- **python3-pip**: 24.0

### Libraries & Dependencies
- **libboost-python-dev**: 1.83.0
- **libhdf5-dev**: 1.10.10
- **libgmp-dev**: 6.3.0
- **libz3-dev**: 4.8.12
- **glpk-utils**: 5.0 (GNU Linear Programming Kit)
- **z3**: 4.8.12 (SMT Solver)

### Desktop Applications
- **Google Chrome**: 144.0.7559.96
- **Claude Desktop**: 1.1.381
- **GitHub Desktop**: 3.4.13
- **OnlyOffice Desktop**: 9.2.1

### Fonts & Localization
Extensive international font support including:
- CJK fonts (Noto CJK, IPAfont, Nanum, Arphic)
- Liberation fonts
- Ubuntu fonts
- Emoji support (Noto Color Emoji)

Language packs installed for:
- English (en)
- Hebrew (he)
- Russian (ru)

## Custom Packages

- **tasepy**: Custom package installed from local repository (`/home/mdmitry/github/tasepy`)

## Notes

### Python 3.14 Compatibility
This environment includes Python 3.14 for forward compatibility testing with Ubuntu 26.04 LTS (expected April 2026). Key considerations:

1. **aiohttp Setup**: Required manual installation of `python3.14-setuptools` to avoid build errors. The default Python 3.12 setuptools caused compilation failures due to header incompatibilities.

2. **TensorFlow**: Not yet available for Python 3.14. Use Python 3.12 environment for TensorFlow-dependent workflows.

3. **NumPy 2.x**: Python 3.14 environment uses NumPy 2.4.1, which may have breaking API changes compared to NumPy 1.x used in Python 3.12.

### NVIDIA CUDA
Full CUDA 12.8/12.9 toolkit installed with comprehensive library support for GPU-accelerated computing.

### Optimization Stack
Robust optimization capabilities with support for:
- Single and multi-objective optimization
- Evolutionary algorithms
- Bayesian optimization
- Linear and nonlinear programming
- Constraint satisfaction

## License

Environment configuration for research and development purposes.
