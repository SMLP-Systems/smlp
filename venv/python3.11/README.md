# SMLP installation instructions in virtual environment

## 1.  Dependencies installation instructions for Ubuntu 24.04

```bash
export DEBIAN_FRONTEND=noninteractive
sudo apt update
sudo apt-get install software-properties-common wget git gzip vim xvfb -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get install libgomp1 tcsh python3.11-dev python3.11-tk python3.11-venv python3.11-distutils python3.11 -y
```

## 2. Clean SMLP installation and running Dead-Or-Alive Test

```bash
./run_venv_build
```

## 3. Build Validation in virtual enviroment

```bash
source venv/bin/activate
./run_smlp_dora
```

## 4. Build validation in Docker

```bash
./run_docker_build
./run_venv_container
source venv/bin/activate
./run_smlp_dora
```
## 5. Regression in virtual enviroment

```bash
source venv/bin/activate
\cp -p ../../docker/python3.11/run_mathsat_build ../../..
../../../run_mathsat_build
rm -rf /tmp/mathsat*
cd smlp_regression
./run_smlp_regression
```

## 6. Regression in Docker

```bash
./run_venv_container
cd smlp_regression
./run_smlp_regression
```

