# SMLP Installation on Ubuntu 24.04 with Python 3.11

This guide describes how to install [smlptech](https://pypi.org/project/smlptech/) on Ubuntu 24.04.
Python 3.11 is required. Since Ubuntu 24.04 ships Python 3.12 by default, Python 3.11
is installed from the [deadsnakes PPA](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa).

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
    git \
    jq \
    libgomp1 \
    locales \
    tcsh \
    tzdata \
    vim \
    wget \
    x11-xserver-utils \
    xvfb
```

---

## Step 2 — Install Python 3.11 with Tk support

Ubuntu 24.04 does not ship Python 3.11 in its standard repositories.
Add the deadsnakes PPA and install:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-tk
python3.11 -m ensurepip
```
---

## Step 3 — Install smlptech

Two installation modes are supported. Choose one.

### Option A — User mode (virtual environment, recommended)

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

### Option B — System mode

Installs smlptech system-wide using `sudo`. This mirrors a typical end-user installation on their own machine.

```bash

sudo pip3.11 install --ignore-installed smlptech

```

> **Note:** `--ignore-installed` is required because Ubuntu ships some of
> smlptech's dependencies (e.g. `blinker`) as apt-managed packages without
> pip `RECORD` files. Without this flag, pip will fail trying to uninstall them.


---

## Step 4 — Validate the installation

Run the following checks to confirm the installation is working:

```bash
# Confirm smlp is importable and print its version
python3.11 -c "import smlp; from importlib.metadata import version; print('smlp version:', version('smlptech'))"

# Confirm Tk is available (required for GUI components)
python3.11 -c "import tkinter; print('tkinter Tcl/Tk:', tkinter.TclVersion)"

# Confirm sudo works
sudo --non-interactive whoami
```

All three commands should complete without errors.

---

## Step 5 (Optional) — UTF-8 locale

Required if you encounter encoding issues:

```bash
sudo locale-gen en_US.UTF-8
```

Add to `~/.bashrc` or `/etc/environment`:

```bash
export LANG=en_US.UTF-8
export LANGUAGE=en_US:en
export LC_ALL=en_US.UTF-8
```

---

## Step 6 (Optional) — Virtual display

A virtual display is needed to run SMLP tools in non-GUI (headless) environments,
for example on servers or in CI pipelines. `xvfb` was installed in Step 1.

Download and run the `open_virtual_display` helper script before launching
any SMLP tool that opens a GUI window:

```bash
wget https://raw.githubusercontent.com/SMLP-Systems/smlp/refs/heads/master/scripts/docker/open_virtual_display
chmod +x open_virtual_display
```

---

## Step 7 (Optional) — MathSAT

MathSAT is an SMT solver used by some SMLP components. To install it:

```bash
wget https://raw.githubusercontent.com/SMLP-Systems/smlp/refs/heads/master/scripts/docker/run_mathsat_build
chmod +x run_mathsat_build
./run_mathsat_build
```

---

## Step 8 (For SMLP developers) — Clone the smlp repository

```bash
# Replace `master` by development branch name, if needed
GIT_BRANCH=master
git clone --branch $GIT_BRANCH https://github.com/SMLP-Systems/smlp.git
```

---

## Summary

| Step | Description | Required |
|------|-------------|----------|
| 1 | System dependencies | Yes |
| 2 | Python 3.11 + Tk via deadsnakes PPA | Yes |
| 3 | Install smlptech (venv or system mode) | Yes |
| 4 | Validate installation | Yes |
| 5 | UTF-8 locale | Optional |
| 6 | Virtual display (`xvfb`) | Optional |
| 7 | MathSAT SMT solver | Optional |
| 8 | Clone smlp repository | Developers only |
