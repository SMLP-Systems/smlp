# SMLP installation instructions in virtual environment

## 1.  Dependencies installation instructions for Ubuntu 24.04

```
export DEBIAN_FRONTEND=noninteractive
sudo apt update
sudo apt-get install software-properties-common wget git gzip gcc g++ make pkg-config vim xvfb -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get install tcsh z3 libz3-dev libgmp-dev ninja-build libboost-python-dev python3.11-dev python3.11-tk python3.11-venv python3.11 -y
```

## 2. Boost installation instructions

```bash
cd /tmp
wget --tries=0 --read-timeout=20 --timeout=15 --waitretry=1 --retry-connrefused https://sourceforge.net/projects/boost/files/boost/1.83.0/boost_1_83_0.tar.gz
tar -xvf boost_1_83_0.tar.gz
cd boost_1_83_0
./bootstrap.sh --with-python=/usr/bin/python3.11 --with-libraries=python
./b2 install --prefix=$HOME/boost_py311 --with-python python=3.11
sudo cp -p $HOME/boost_py311/lib/libboost_python311.a /usr/lib/x86_64-linux-gnu
sudo cp -p $HOME/boost_py311/lib/libboost_python311.so.1.83.0 /usr/lib/x86_64-linux-gnu
sudo ln -s /usr/lib/x86_64-linux-gnu/{libboost_python311.so.1.83.0,libboost_python311.so}
cd $HOME
\rm -rf boost_py311 /tmp/boost*
```

## 3. Clean SMLP installation

```bash
git clone https://github.com/smlp-systems/smlp smlp_venv
cd smlp_venv
git checkout remotes/origin/venv_build venv
cd venv/python3.11
```

## 4. SMLP dependencies installation instructions

```bash
./run_venv_build
```

## 5. Build Validation 

Enter virtual environment

```
bash
source venv/bin/activate
./run_smlp_dora
```

## 6. Validation in Docker

```bash
./run_docker_build
./run_venv_container
source venv/bin/activate
./run_smlp_dora
```
## 7. Regression in Virtual enviroment after successful build validation

```bash
\cp -p ../../docker/python3.11/run_mathsat_build ../../..
../../../run_mathsat_build
rm -rf /tmp/mathsat*
cd smlp_regression
./run_smlp_regression
```

## 7. Regression in Docker environment

```bash
cd smlp_regression
./run_smlp_regression
```

