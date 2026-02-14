# SMLP installation instructions in virtual environment

## 1.  Dependencies installation instructions for Ubuntu 24.04

```
sudo apt update
sudo apt install software-properties-common
sudo apt install add-apt-repository ppa:deadsnakes/ppa
sudo apt install tcsh z3 libz3-dev ninja-build libboost-python-dev python3.11-dev python3.11-tk python3.11-venv python3.11
```

## 2. Boost installation instructions

```bash
cd /tmp
wget https://sourceforge.net/projects/boost/files/boost/1.83.0/boost_1_83_0.tar.gz
tar -xvf boost_1_83_0.tar.gz
cd boost_1_83_0
./bootstrap.sh --with-python=/usr/bin/python3.11 --with-libraries=python
./b2 install --prefix=$HOME/boost_py311 --with-python python=3.11
sudo cp -p $HOME/boost_py311/lib/libboost_python311.a /usr/lib/x86_64-linux-gnu
sudo cp -p $HOME/boost_py311/lib/libboost_python311.so.1.83.0 /usr/lib/x86_64-linux-gnu
sudo ln -s /usr/lib/x86_64-linux-gnu/{libboost_python311.so.1.83.0,libboost_python311.so}
```

## 3. Clean SMLP installation

```bash
git clone https://git.github.com/SMLP-Systems/smlp smlp_venv
cd smlp_venv
```

## 4. SMLP dependencies installation instructions

```bash
run_venv_build
```
