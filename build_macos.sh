#!/usr/bin/env bash
# =============================================================================
# build_macos.sh  –  macOS equivalent of the manylinux Dockerfile
#
# Installs all prerequisites via Homebrew + pip, clones the smlp repo on the
# requested branch, builds a wheel, and repairs it with delocate so all
# dylib dependencies are bundled (equivalent to auditwheel on Linux).
#
# Usage:
#   ./build_macos.sh
#
# The finished wheel is left in  ./dist/
# =============================================================================
set -euo pipefail

set -x

PYTHON_VERSION="3.11"

# ---------------------------------------------------------------------------
# 1. System packages via Homebrew
# ---------------------------------------------------------------------------
echo "[build] Installing Homebrew packages..."
if ! command -v brew &>/dev/null; then
    echo "ERROR: Homebrew not found. Install it from https://brew.sh" >&2
    exit 1
fi

CC="$(brew --prefix)/bin/gcc-15"
CXX="$(brew --prefix)/bin/g++-15"

brew install wget git make m4 pkg-config gmp

# ---------------------------------------------------------------------------
# 2. Python 3.11
# ---------------------------------------------------------------------------
echo "[build] Checking Python ${PYTHON_VERSION}..."
if ! command -v python3.11 &>/dev/null; then
    echo "[build] Installing Python ${PYTHON_VERSION} via Homebrew..."
    brew install python@3.11
fi

PYTHON="$(command -v python3.11)"
PIP="${PYTHON} -m pip"

echo "[build] Using Python: ${PYTHON}"
echo "[build] Python version: $(${PYTHON} --version)"

# ---------------------------------------------------------------------------
# 3. Python build tools
# ---------------------------------------------------------------------------
echo "[build] Installing Python build tools..."
${PIP} install --upgrade pip
${PIP} install "setuptools >= 71.1.0"
#${PIP} install meson ninja "z3-solver==4.8.12" delocate

${PIP} install meson ninja "z3-solver==4.16.0" delocate

# ---------------------------------------------------------------------------
# 4. Environment
# ---------------------------------------------------------------------------
# On macOS, Homebrew installs binaries into $(brew --prefix)/bin
BREW_PREFIX="$(brew --prefix)"
export PATH="${BREW_PREFIX}/bin:${HOME}/.local/bin:${PATH}"

# Point setup.py at the z3-solver installed for this Python
PY_SITE="$(${PYTHON} -c "import site; print(site.getsitepackages()[0])")"
export Z3_PREFIX="${PY_SITE}/z3"

# GMP is installed by Homebrew – reuse it so setup.py skips the source build
export GMP_ROOT="${BREW_PREFIX}/opt/gmp"

echo "[build] Z3_PREFIX=${Z3_PREFIX}"
echo "[build] GMP_ROOT=${GMP_ROOT}"

# DLD_PATH is a workaround of MacOS purging DYLD_LIBRARY_PATH when calling python processes

export DLD_PATH="${Z3_PREFIX}/lib"

echo "[DLD_PATH] $DLD_PATH"

# ---------------------------------------------------------------------------
# 5. Build wheel
# ---------------------------------------------------------------------------
echo "[build] Building wheel..."
${PYTHON} -m pip wheel . -w dist/

# ---------------------------------------------------------------------------
# 7. Repair wheel with delocate (macOS equivalent of auditwheel)
# ---------------------------------------------------------------------------
echo "[build] Repairing wheel with delocate..."


${PYTHON} repair_wheel.py dist/

#----- Fix libpython3.11.dylib path in the .so inside the whl into @rpath/libpython3.11.dylib

echo "[./repair_path.sh] Fixing path to libpython3.11.dylib in .so"
./repair_path.sh
    

#---------


echo ""
echo "======================================================"
echo " Build complete.  Wheel is in: $PWD/dist/"
echo "======================================================"
