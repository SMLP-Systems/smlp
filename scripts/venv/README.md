# SMLP Installation Guide for Ubuntu 24.04

This guide describes how to install [smlptech](https://pypi.org/project/smlptech/) on Ubuntu 24.04.

---

## Prerequisites

- Ubuntu 24.04
- `sudo` access
- Internet access (for apt, pip, and wget)

---

## Step 1 — Install system dependencies

```bash
sudo apt-get update
sudo apt-get install -y \
    jq \
    libgomp1 \
    tcsh \
    wget \
```

| Dependency | Used by | Mandatory
|---|---|---|
| jq | Quickstart and Tutorial | No
| **libgomp1** | **SMLP** | **Yes**
| tcsh | Tutorial | No
| wget | Mathsat installation | No


---

## Step 2 — Install Python 3.11 with Tk support


```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-tk
```
---

## Step 3 — Install smlptech in virtual environment

Installs smlptech into an isolated virtual environment under `~/.venv`.
No `sudo` required for the installation itself.

```bash
python3.11 -m venv ~/.venv
export PATH=~/.venv/bin:$PATH
source ~/.venv/bin/activate
pip3.11 install smlptech
```

To make the virtual environment available in every new shell session, add the following line to `~/.bashrc`:

```bash
export PATH=~/.venv/bin:$PATH
```

---

## Step 4 (Recommended) Validate the installation

Run the following checks to confirm the installation is working:

```bash
# Confirm smlp is importable and print its version
python3.11 -c "import smlp; from importlib.metadata import version; print('smlp version:', version('smlptech'))"

# Confirm Tk is available (required for GUI components and PNG files generation in non-GUI environment)
python3.11 -c "import tkinter; print('tkinter Tcl/Tk:', tkinter.TclVersion)"
```

Both commands should complete without errors.

---

## Step 5 (Optional) — Install MathSAT

MathSAT is a Satisfiability Modulo Theories (SMT) solver developed as a joint project between Fondazione Bruno Kessler (FBK) and the University of Trento (DISI) in Italy. It is optionally used by SMLP.

⚠️ **Licensing limitations**

Please, read [MathSat5 license terms](https://mathsat.fbk.eu/download.html) before using MathSat

- *MathSAT5 is available for research and evaluation purposes only.* **It can not be used in a commercial environment, particularly as part of a commercial product, without written permission.** *MathSAT5 is provided as is, without any warranty.*

To install MathSat and validate installation:

```bash
wget https://raw.githubusercontent.com/SMLP-Systems/smlp/refs/heads/master/scripts/docker/run_mathsat_build
chmod +x run_mathsat_build
./run_mathsat_build && rm -rf /tmp/mathsat* && external/mathsat-5.6.8-linux-x86_64-reentrant/bin/mathsat -version
```

---

## Summary

| Step | Description | Required |
|------|-------------|----------|
| 1 | System dependencies | Yes |
| 2 | Python 3.11 + Tk via deadsnakes PPA | Yes |
| 3 | Install smlptech | Yes |
| 4 | Validate installation | No |
| 5 | MathSAT SMT solver | Optional |
