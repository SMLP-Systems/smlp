# SMLP Installation Guide

Complete installation steps for a clean **Ubuntu 20.04** image.

---

## 1. System packages

These require `sudo` and only need to be installed once.

```bash
apt update
apt install -y ca-certificates curl gnupg wget

curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xF23C5A6CF475977595C89F51BA6932366A755776" \
    | gpg --dearmor -o /etc/apt/trusted.gpg.d/deadsnakes.gpg

echo "deb https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu focal main" \
    > /etc/apt/sources.list.d/deadsnakes.list

apt update
apt install -y \
    gcc g++ git make m4 pkg-config xvfb \
    python3.13 python3.13-dev python3.13-venv python3.13-tk
```

---

## 2. Install pip for Python 3.13

```bash
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.13
```

---

## 3. Python user packages

These are installed into `~/.local` and do not require `sudo`.

```bash
python3.13 -m pip install --user --force-reinstall --break-system-packages setuptools
python3.13 -m pip install --user meson ninja z3-solver==4.8.12
```

---

## 4. Add `~/.local/bin` to PATH

Required so that `meson`, `ninja`, and `z3` are found during the build.

```bash
export PATH=$HOME/.local/bin:$PATH
```

Add it permanently to your shell profile:

```bash
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

---

## 5. Clone the smlp repository

```bash
git clone https://github.com/SMLP-Systems/smlp.git
cd smlp
git switch smlp_python313
```

---

## 6. Install smlp

The `setup.py` automatically:
- Downloads and compiles **Boost 1.83** (cached in `~/.local/boost_py313`, ~5 min first time)
- Downloads and compiles **GMP 6.3.0** (cached in `~/.local/gmp`, ~1 min first time)
- Clones the **kay** C++ dependency
- Runs `meson` + `ninja` to build the native extension

```bash
python3.13 -m pip install .
```

Subsequent installs reuse the Boost and GMP caches and are much faster.

---

## 7. Build a wheel (optional)

To save a redistributable `.whl` file instead of installing directly:

```bash
python3.13 -m pip wheel . -w dist/
```

The wheel is saved in `dist/smlp-*.whl` and can be installed without recompiling:

```bash
python3.13 -m pip install dist/smlp-*.whl
```

> **Portability note:** The plain wheel (`linux_x86_64`) embeds RPATH entries pointing to
> `~/.local/boost_py313/lib`, `~/.local/gmp/lib`, and `~/.local/lib/python3.13/site-packages/z3/lib`.
> It will only work on machines with the same library paths.

### Manylinux wheel (fully portable)

A manylinux wheel bundles all `.so` dependencies inside it and works on any Linux x86_64
with a compatible glibc. First install the required tools:

```bash
python3.13 -m pip install --user auditwheel patchelf
```

Then build the plain wheel and repair it:

```bash
python3.13 -m pip wheel . -w dist/
python3.13 repair_wheel.py
```

The repaired `manylinux_*_x86_64` wheel will be saved alongside the original in `dist/`.
Distribute the manylinux one.

---

## 8. Headless display (xvfb)

Required when running SMLP without a `$DISPLAY` (e.g. in Docker or CI).

```bash
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
```

Add to your shell profile to make it permanent:

```bash
echo 'export DISPLAY=:99' >> ~/.bashrc
```

---

## 9. Verify

```bash
python3.13 -c "import smlp; print('smlp imported OK')"
```

---

## Environment variables

All are optional. Set before running `pip install .` to override defaults.

| Variable | Default | Description |
|---|---|---|
| `BOOST_ROOT` | *(build from source)* | Reuse an existing Boost prefix, skips download + compile |
| `BOOST_CACHE_DIR` | `~/.local/boost_py313` | Where to cache the compiled Boost |
| `BOOST_VERSION` | `1.83.0` | Boost version to download |
| `GMP_ROOT` | *(build from source)* | Reuse an existing GMP prefix, skips download + compile |
| `GMP_CACHE_DIR` | `~/.local/gmp` | Where to cache the compiled GMP |
| `GMP_VERSION` | `6.3.0` | GMP version to download |
| `Z3_PREFIX` | `~/.local/lib/python3.13/site-packages/z3` | Reuse an existing Z3 install |
| `Z3_BIN_DIR` | `~/.local/z3/bin` | Directory containing the `z3` binary |
| `Z3_VERSION` | `4.8.12` | Z3 version to download if binary not found |
| `KAY_DIR` | *(cloned into build temp)* | Reuse an existing kay checkout |
| `SMLP_BRANCH` | *(auto-detected)* | Git branch to use in the smlp repo |

---

## Reinstalling

To reinstall after code changes:

```bash
python3.13 -m pip uninstall smlp -y
python3.13 -m pip install .
```

To force a full rebuild including Boost and GMP:

```bash
rm -rf ~/.local/boost_py313 ~/.local/gmp
python3.13 -m pip install .
```

---

## Troubleshooting

**`No module named 'z3'` during build**

Install z3-solver before running pip install:
```bash
python3.13 -m pip install --user z3-solver==4.8.12
```

**`meson: command not found` during build**

Make sure `~/.local/bin` is on `PATH` as described in step 3.

**`sys/pstat.h: No such file or directory` during GMP build**

Delete the GMP cache and rebuild — the `--disable-assembly` flag handles this:
```bash
rm -rf ~/.local/gmp ~/.local/_gmp_build_tmp
python3.13 -m pip install .
```
