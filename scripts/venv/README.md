# SMLP Installation Guide for Ubuntu 24.04

## 1. Install Python 3.11

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-dev python3.11-distutils python3.11-venv python3.11-tk
```

## 2. Install gomp library - this step is needed only if gcc is *NOT* installed

sudo apt-get install libgomp1

## 3. Install SMLP package, MathSat and run DORA test

```bash
./run_dora
```

## 4. Run regression 

```bash
source smlp_package_venv/bin/activate
cd smlp_package_venv/smlp/regr_smlp/code
./smlp_regr.py <regression_script_parameters>
```
