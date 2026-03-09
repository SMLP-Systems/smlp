# SMLP Installation Guide for Ubuntu 24.04

## 1. Install SMLP package, MathSat and run DORA test

```bash
./run_dora
```

## 2. Run regression 

```bash
source smlp_package_venv/bin/activate
cd smlp_package_venv/smlp/regr_smlp/code
./smlp_regr.py <regression_script_parameters>
```
