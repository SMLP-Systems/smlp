"""
setup.py for the smlp package.

System prerequisites (require sudo, install once)
--------------------------------------------------
  sudo apt install gcc g++ git make m4 pkg-config

User prerequisites (no sudo, install once)
------------------------------------------
  python3.13 -m pip install --user meson ninja z3-solver

Build flow
----------
1.  Boost.Python 1.83 is compiled from source for Python 3.13 and cached in
    ~/.local/boost_py313  (or the path in $BOOST_CACHE_DIR).
    The build is skipped on subsequent runs if the cache directory already
    contains the marker file  .built_for_python313.
    Set $BOOST_ROOT to point at an existing Boost prefix to skip this step
    entirely.

2.  The 'kay' C++ dependency is cloned from GitHub into the pip build-temp
    directory (or reused from $KAY_DIR).

3.  `meson setup` + `ninja install` is run inside  utils/poly/  of the
    repository this setup.py lives in.  No repo cloning is performed;
    setup.py is expected to be in the root of the smlp checkout.

4.  The installed smlp extension package is copied into the wheel.

Environment variables
---------------------
BOOST_ROOT       Reuse an existing Boost prefix – skips download + compile.
                 e.g.  export BOOST_ROOT=~/.local/boost_py313
BOOST_CACHE_DIR  Where to cache the compiled Boost (default: ~/.local/boost_py313).
BOOST_VERSION    Boost version to download (default: 1.83.0).
KAY_DIR          Reuse an existing kay checkout.
GMP_ROOT         Reuse an existing GMP prefix – skips download + compile.
                 e.g.  export GMP_ROOT=~/.local/gmp
GMP_CACHE_DIR    Where to cache compiled GMP (default: ~/.local/gmp).
GMP_VERSION      GMP version to download (default: 6.3.0).
Z3_PREFIX        Reuse an existing Z3 install prefix – skips pip z3-solver.
                 e.g.  export Z3_PREFIX=~/.local/z3
Z3_VERSION       Z3 version to download binary for (default: 4.8.12).
Z3_BIN_DIR       Path to directory containing z3 binary (default: ~/.local/z3/bin).
SMLP_BRANCH      Git branch to switch to in the smlp repo (auto-detected if unset).
"""

import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

from setuptools import setup
from setuptools.command.build_ext import build_ext as _build_ext


# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

BOOST_VERSION   = os.environ.get("BOOST_VERSION", "1.83.0")
BOOST_CACHE_DIR = Path(
    os.environ.get("BOOST_CACHE_DIR", Path.home() / ".local" / "boost_py313")
).expanduser()

# Default Z3_PREFIX: where z3-solver installs its lib/libz3.so
# This is the standard location when installed via:
#   python3.13 -m pip install --user z3-solver
Z3_DEFAULT_PREFIX = (
    Path.home() / ".local" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages" / "z3"
)

GMP_VERSION   = os.environ.get("GMP_VERSION", "6.3.0")
GMP_CACHE_DIR = Path(
    os.environ.get("GMP_CACHE_DIR", Path.home() / ".local" / "gmp")
).expanduser()

Z3_VERSION    = os.environ.get("Z3_VERSION", "4.8.12")
Z3_BIN_DIR    = Path(
    os.environ.get("Z3_BIN_DIR", Path.home() / ".local" / "z3" / "bin")
).expanduser()

# Root of this repository (where setup.py lives)
REPO_ROOT = Path(__file__).parent.resolve()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd, **kwargs):
    print(f"[smlp build] $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run([str(c) for c in cmd], **kwargs)
    if result.returncode != 0:
        # Print captured output if any
        if hasattr(result, "stdout") and result.stdout:
            print(result.stdout)
        if hasattr(result, "stderr") and result.stderr:
            print(result.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd)


def _verify_tarball(path: Path) -> bool:
    """Return True if the tarball can be opened and is not truncated."""
    try:
        with tarfile.open(path) as tf:
            # Read all members to detect truncation
            tf.getmembers()
        return True
    except Exception:
        return False


def _download(url: str, dest: Path, retries: int = 5) -> None:
    """Download a file with retries, verifying integrity after each attempt."""
    import time

    # Remove any existing file — it may be a corrupt partial download
    if dest.exists():
        if _verify_tarball(dest):
            print(f"[smlp build] Using verified cached tarball {dest}")
            return
        else:
            print(f"[smlp build] Removing corrupt cached tarball {dest}")
            dest.unlink()

    for attempt in range(1, retries + 1):
        try:
            print(f"[smlp build] Downloading {url} (attempt {attempt}/{retries}) ...")
            urllib.request.urlretrieve(url, dest)
            if _verify_tarball(dest):
                print(f"[smlp build] Download verified OK.")
                return
            else:
                print(f"[smlp build] Download corrupt, retrying ...")
                dest.unlink()
        except Exception as e:
            print(f"[smlp build] Download failed: {e}")
            if dest.exists():
                dest.unlink()
        if attempt < retries:
            wait = 2 ** attempt
            print(f"[smlp build] Retrying in {wait}s ...")
            time.sleep(wait)
    sys.exit(f"[smlp build] ERROR: failed to download {url} after {retries} attempts.")


def _meson_bin(build_tmp: Path) -> list[str]:
    """
    Write a meson wrapper script and return the command to invoke it.

    The wrapper explicitly adds the meson install location to sys.path,
    so it works in pip's isolated build environment where user site-packages
    is not on sys.path. Meson stores the wrapper path in the build dir and
    reuses it for internal calls like `meson install`, so it must be a real
    executable file — not a -c string.
    """
    # Find where mesonbuild is installed via pip show
    mesonbuild_location = None
    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", "meson"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if line.startswith("Location:"):
                mesonbuild_location = line.split(":", 1)[1].strip()
                break

    # Fallback: check user site-packages directly
    if not mesonbuild_location:
        user_site = (
            Path.home() / ".local" / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        )
        if (user_site / "mesonbuild").exists():
            mesonbuild_location = str(user_site)

    if not mesonbuild_location:
        raise RuntimeError(
            "[smlp build] meson not found. Run:  python3.13 -m pip install --user meson"
        )

    print(f"[smlp build] meson location: {mesonbuild_location}")

    # Write a wrapper script with a proper shebang so Meson can store and
    # reuse its path for internal calls (meson install, meson test, etc.)
    wrapper = build_tmp / "meson"
    wrapper.write_text(
        f"#!/usr/bin/env {sys.executable}\n"
        "import sys\n"
        f"sys.path.insert(0, {mesonbuild_location!r})\n"
        "from mesonbuild.mesonmain import main\n"
        "sys.exit(main())\n"
    )
    wrapper.chmod(0o755)
    print(f"[smlp build] Using meson wrapper: {wrapper}")
    return [str(wrapper)]


def _ninja_bin() -> str:
    """
    Resolve the ninja binary, preferring user-space installs over system ones.

    Search order:
      1. The 'ninja' PyPI package  (pip install ninja → <prefix>/bin/ninja)
      2. ~/.local/bin/ninja        (pip install --user ninja)
      3. PATH                      (last resort — may find /usr/bin/ninja)
    """
    import importlib.util
    from shutil import which

    # ── 1. pip ninja package ─────────────────────────────────────────────
    spec = importlib.util.find_spec("ninja")
    if spec is not None:
        try:
            import ninja as _ninja_pkg  # type: ignore
            candidate = Path(_ninja_pkg.BIN_DIR) / "ninja"
            if candidate.exists():
                print(f"[smlp build] Using pip ninja: {candidate}")
                return str(candidate)
        except Exception:
            pass

    # ── 2. ~/.local/bin (pip install --user) ─────────────────────────────
    user_ninja = Path.home() / ".local" / "bin" / "ninja"
    if user_ninja.exists():
        print(f"[smlp build] Using user ninja: {user_ninja}")
        return str(user_ninja)

    # ── 3. PATH fallback ─────────────────────────────────────────────────
    found = which("ninja")
    if found:
        print(f"[smlp build] Using ninja from PATH: {found}")
        return found

    raise RuntimeError(
        "[smlp build] ninja not found. Run:  pip install ninja"
    )


# ---------------------------------------------------------------------------
# Step 1 – Boost.Python (compiled from source, cached in user-space)
# ---------------------------------------------------------------------------

def _boost_prefix() -> Path:
    """
    Return the Boost install prefix, building from source if necessary.

    Search order:
      1. $BOOST_ROOT env var        → use as-is, no build
      2. BOOST_CACHE_DIR marker     → cache hit, skip build
      3. Download + compile into BOOST_CACHE_DIR
    """
    # ── Option A: caller supplied an existing prefix ──────────────────────
    env_root = os.environ.get("BOOST_ROOT")
    if env_root:
        prefix = Path(env_root).expanduser()
        print(f"[smlp build] Using BOOST_ROOT={prefix}")
        return prefix

    # ── Option B: cached build already present ────────────────────────────
    tag_file = BOOST_CACHE_DIR / ".built_for_python313"
    if tag_file.exists():
        print(f"[smlp build] Boost cache found at {BOOST_CACHE_DIR}, skipping build.")
        return BOOST_CACHE_DIR

    # ── Option C: download + compile into user-space cache ────────────────
    ver_flat     = BOOST_VERSION.replace(".", "_")
    tarball_name = f"boost_{ver_flat}.tar.gz"
    url          = f"https://archives.boost.io/release/{BOOST_VERSION}/source/{tarball_name}"

    # Temporary directory for download + extraction (sibling of cache dir)
    tmp_dir = BOOST_CACHE_DIR.parent / "_boost_build_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    tarball_path = tmp_dir / tarball_name
    if not tarball_path.exists():
        print(f"[smlp build] Downloading Boost {BOOST_VERSION} ...")
        _download(url, tarball_path)
    else:
        print(f"[smlp build] Using cached tarball {tarball_path}")

    print(f"[smlp build] Extracting {tarball_name} ...")
    with tarfile.open(tarball_path) as tf:
        tf.extractall(tmp_dir)

    src = tmp_dir / f"boost_{ver_flat}"

    print(f"[smlp build] Bootstrapping Boost (python={sys.executable}) ...")
    _run(
        ["./bootstrap.sh",
         f"--with-python={sys.executable}",
         "--with-libraries=python"],
        cwd=str(src),
    )

    BOOST_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[smlp build] Compiling Boost → {BOOST_CACHE_DIR}  (this takes a few minutes) ...")
    _run(
        ["./b2", "install",
         f"--prefix={BOOST_CACHE_DIR}",
         "--with-python",
         "python=3.13"],
        cwd=str(src),
    )

    # Leave a marker so we skip the build on the next pip install
    tag_file.touch()

    # Remove the source + tarball; keep only the install
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"[smlp build] Boost built and cached at {BOOST_CACHE_DIR}")
    return BOOST_CACHE_DIR



def _boost_env(prefix: Path) -> dict:
    """
    Environment variables for meson/ninja so the user-space Boost is found
    without touching /usr/lib.
    """
    lib_dir = prefix / "lib"
    inc_dir = prefix / "include"

    env = os.environ.copy()
    env["BOOST_ROOT"]       = str(prefix)
    env["BOOST_INCLUDEDIR"] = str(inc_dir)
    env["BOOST_LIBRARYDIR"] = str(lib_dir)

    # Force Meson to use the same Python that is running this build script,
    # preventing it from falling back to the system Python 3.12.
    env["PYTHON"]           = sys.executable
    env["PYTHON3"]          = sys.executable

    # Tell Meson the exact versioned Boost.Python library name,
    # e.g. Python 3.13 → boost_python313
    py_ver = f"{sys.version_info.major}{sys.version_info.minor}"
    env["BOOST_PYTHON_LIBNAME"] = f"boost_python{py_ver}"

    existing_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"]  = f"{lib_dir}:{existing_ld}" if existing_ld else str(lib_dir)

    pkgconfig    = lib_dir / "pkgconfig"
    existing_pkg = env.get("PKG_CONFIG_PATH", "")
    env["PKG_CONFIG_PATH"]  = f"{pkgconfig}:{existing_pkg}" if existing_pkg else str(pkgconfig)

    return env


def _add_z3_to_env(env: dict, z3_lib: Path) -> dict:
    """Prepend the z3-solver lib/bin directories to the relevant env vars."""
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{z3_lib}:{existing_ld}" if existing_ld else str(z3_lib)

    existing_pkg = env.get("PKG_CONFIG_PATH", "")
    env["PKG_CONFIG_PATH"] = f"{z3_lib}:{existing_pkg}" if existing_pkg else str(z3_lib)

    # Add z3 binary to PATH so meson can find the solver executable
    z3_bin = z3_lib.parent / "bin"
    existing_path = env.get("PATH", os.environ.get("PATH", ""))
    env["PATH"] = f"{z3_bin}:{existing_path}" if existing_path else str(z3_bin)

    return env


def _add_gmp_to_env(env: dict, gmp_prefix: Path) -> dict:
    """Prepend the GMP lib/include directories to the relevant env vars."""
    gmp_lib = gmp_prefix / "lib"
    gmp_inc = gmp_prefix / "include"

    existing_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"]  = f"{gmp_lib}:{existing_ld}" if existing_ld else str(gmp_lib)

    pkgconfig    = gmp_lib / "pkgconfig"
    existing_pkg = env.get("PKG_CONFIG_PATH", "")
    env["PKG_CONFIG_PATH"]  = f"{pkgconfig}:{existing_pkg}" if existing_pkg else str(pkgconfig)

    existing_cpp = env.get("CPPFLAGS", "")
    env["CPPFLAGS"] = f"-I{gmp_inc} {existing_cpp}".strip()

    existing_ld_flags = env.get("LDFLAGS", "")
    env["LDFLAGS"] = f"-L{gmp_lib} {existing_ld_flags}".strip()

    return env


# ---------------------------------------------------------------------------
# Step 2 – kay dependency
# ---------------------------------------------------------------------------

def _ensure_kay(build_tmp: Path) -> Path:
    kay_env = os.environ.get("KAY_DIR")
    if kay_env:
        kay_dir = Path(kay_env).expanduser()
        print(f"[smlp build] Using existing kay at {kay_dir}")
        return kay_dir

    kay_dir = build_tmp.resolve() / "kay"
    if kay_dir.exists():
        print(f"[smlp build] Reusing kay clone at {kay_dir}")
    else:
        _run(["git", "clone", "https://github.com/fbrausse/kay", str(kay_dir)])
    return kay_dir


# ---------------------------------------------------------------------------
# Step 1c – GMP (compiled from source, cached in user-space)
# ---------------------------------------------------------------------------

def _write_gmp_pc(prefix: Path) -> None:
    """
    Write a gmp.pc pkg-config file into <prefix>/lib/pkgconfig/.
    GMP does not generate one by default, so Meson cannot find it
    via pkg-config without this file.
    """
    pkgconfig_dir = prefix / "lib" / "pkgconfig"
    pkgconfig_dir.mkdir(parents=True, exist_ok=True)
    pc_file = pkgconfig_dir / "gmp.pc"
    pc_file.write_text(
        f"prefix={prefix}\n"
        "exec_prefix=${prefix}\n"
        "libdir=${exec_prefix}/lib\n"
        "includedir=${prefix}/include\n"
        "\n"
        "Name: gmp\n"
        "Description: GNU Multiple Precision Arithmetic Library\n"
        f"Version: {GMP_VERSION}\n"
        "Libs: -L${libdir} -lgmp\n"
        "Cflags: -I${includedir}\n"
    )
    print(f"[smlp build] Wrote pkg-config file: {pc_file}")

    # Also write gmpxx.pc for the C++ wrapper library
    pcxx_file = pkgconfig_dir / "gmpxx.pc"
    pcxx_file.write_text(
        f"prefix={prefix}\n"
        "exec_prefix=${prefix}\n"
        "libdir=${exec_prefix}/lib\n"
        "includedir=${prefix}/include\n"
        "\n"
        "Name: gmpxx\n"
        "Description: GNU Multiple Precision Arithmetic Library (C++ bindings)\n"
        f"Version: {GMP_VERSION}\n"
        "Requires: gmp\n"
        "Libs: -L${libdir} -lgmpxx -lgmp\n"
        "Cflags: -I${includedir}\n"
    )
    print(f"[smlp build] Wrote pkg-config file: {pcxx_file}")


def _gmp_prefix() -> Path:
    """
    Return the GMP install prefix, building from source if necessary.

    Search order:
      1. $GMP_ROOT env var       → use as-is, no build
      2. GMP_CACHE_DIR marker    → cache hit, skip build
      3. Download + compile into GMP_CACHE_DIR
    """
    # ── Option A: caller supplied an existing prefix ──────────────────────
    env_root = os.environ.get("GMP_ROOT")
    if env_root:
        prefix = Path(env_root).expanduser()
        print(f"[smlp build] Using GMP_ROOT={prefix}")
        return prefix

    # ── Option B: cached build already present ────────────────────────────
    tag_file = GMP_CACHE_DIR / ".built"
    if tag_file.exists():
        print(f"[smlp build] GMP cache found at {GMP_CACHE_DIR}, skipping build.")
        _write_gmp_pc(GMP_CACHE_DIR)
        return GMP_CACHE_DIR

    # ── Option C: download + compile into user-space cache ────────────────
    tarball_name = f"gmp-{GMP_VERSION}.tar.xz"
    url          = f"https://gmplib.org/download/gmp/{tarball_name}"

    tmp_dir = GMP_CACHE_DIR.parent / "_gmp_build_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    tarball_path = tmp_dir / tarball_name
    if not tarball_path.exists():
        print(f"[smlp build] Downloading GMP {GMP_VERSION} ...")
        _download(url, tarball_path)
    else:
        print(f"[smlp build] Using cached tarball {tarball_path}")

    print(f"[smlp build] Extracting {tarball_name} ...")
    with tarfile.open(tarball_path) as tf:
        tf.extractall(tmp_dir)

    src = tmp_dir / f"gmp-{GMP_VERSION}"

    GMP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[smlp build] Compiling GMP → {GMP_CACHE_DIR}  (this takes a minute) ...")
    import platform as _platform
    machine = _platform.machine()  # e.g. x86_64, aarch64
    system  = _platform.system().lower()  # linux
    host    = f"{machine}-pc-{system}-gnu"

    _run(
        ["./configure",
         f"--prefix={GMP_CACHE_DIR}",
         f"--host={host}",
         "--enable-shared",
         "--enable-static",
         "--disable-assembly",
         "--enable-cxx"],   # avoids platform-specific asm issues
        cwd=str(src),
    )
    _run(["make", f"-j{os.cpu_count() or 1}"], cwd=str(src))
    _run(["make", "install"], cwd=str(src))

    # Generate a gmp.pc pkg-config file — GMP doesn't ship one by default
    _write_gmp_pc(GMP_CACHE_DIR)

    # Leave a marker so we skip the build on the next pip install
    tag_file.touch()

    # Remove the source + tarball; keep only the install
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"[smlp build] GMP built and cached at {GMP_CACHE_DIR}")
    return GMP_CACHE_DIR



# ---------------------------------------------------------------------------
# Step 1b – Z3 (via pip z3-solver, no sudo)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Step 1d – Z3 binary (downloaded from GitHub releases)
# ---------------------------------------------------------------------------

def _z3_binary() -> Path:
    """
    Return the path to the z3 executable.

    Search order:
      1. Z3_BIN_DIR env var / constant   (~/.local/z3/bin/z3)
      2. ~/.local/bin/z3                 (pip install --user z3-solver installs it here)
      3. System z3 on PATH               (sudo apt install z3)
      4. Download pre-built from GitHub  (no sudo fallback)
    """
    from shutil import which

    # ── 1. Explicit Z3_BIN_DIR ────────────────────────────────────────────
    if Z3_BIN_DIR.exists() and (Z3_BIN_DIR / "z3").exists():
        print(f"[smlp build] Using z3 binary from Z3_BIN_DIR: {Z3_BIN_DIR / 'z3'}")
        return Z3_BIN_DIR / "z3"

    # ── 2. ~/.local/bin/z3 ───────────────────────────────────────────────
    user_z3 = Path.home() / ".local" / "bin" / "z3"
    if user_z3.exists():
        print(f"[smlp build] Using user z3 binary: {user_z3}")
        return user_z3

    # ── 3. PATH ───────────────────────────────────────────────────────────
    system_z3 = which("z3")
    if system_z3:
        print(f"[smlp build] Using system z3: {system_z3}")
        return Path(system_z3)

    # ── 4. Download pre-built binary from GitHub releases ─────────────────
    import platform as _platform
    machine = _platform.machine()
    arch_map = {"x86_64": "x64", "aarch64": "arm64"}
    arch = arch_map.get(machine, machine)
    z3_release = f"z3-{Z3_VERSION}-{arch}-glibc-2.31"
    url = (
        f"https://github.com/Z3Prover/z3/releases/download/z3-{Z3_VERSION}/"
        f"{z3_release}.zip"
    )

    tmp_dir = Z3_BIN_DIR.parent.parent / "_z3_build_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    zip_path = tmp_dir / f"{z3_release}.zip"

    print(f"[smlp build] Downloading z3 binary {Z3_VERSION} ...")
    _download(url, zip_path)

    import zipfile, shutil as _shutil
    print(f"[smlp build] Extracting z3 binary ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp_dir)

    Z3_BIN_DIR.mkdir(parents=True, exist_ok=True)
    src_bin = tmp_dir / z3_release / "bin" / "z3"
    _shutil.copy2(src_bin, Z3_BIN_DIR / "z3")
    (Z3_BIN_DIR / "z3").chmod(0o755)
    _shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"[smlp build] z3 binary installed at {Z3_BIN_DIR / 'z3'}")
    return Z3_BIN_DIR / "z3"


def _write_z3_pc(z3_lib: Path) -> None:
    """
    Write a z3.pc pkg-config file into <z3_lib>/pkgconfig/.
    z3-solver does not ship one, so Meson cannot find it via pkg-config
    without this file.
    """
    import re
    # Detect z3 version from libz3.so filename e.g. libz3.so.4.8.12
    version = "4.8.12"  # fallback
    for f in z3_lib.glob("libz3.so.*"):
        m = re.search(r"libz3\.so\.([\d.]+)", f.name)
        if m:
            version = m.group(1)
            break

    prefix    = z3_lib.parent  # <site-packages>/z3
    inc_dir   = prefix / "include"

    pkgconfig_dir = z3_lib / "pkgconfig"
    pkgconfig_dir.mkdir(parents=True, exist_ok=True)
    pc_file = pkgconfig_dir / "z3.pc"
    pc_file.write_text(
        f"prefix={prefix}\n"
        f"libdir={z3_lib}\n"
        f"includedir={inc_dir}\n"
        "\n"
        "Name: z3\n"
        "Description: Z3 Theorem Prover\n"
        f"Version: {version}\n"
        "Libs: -L${libdir} -lz3\n"
        "Cflags: -I${includedir}\n"
    )
    print(f"[smlp build] Wrote pkg-config file: {pc_file}")


def _z3_prefix() -> Path:
    """
    Return the z3-solver lib directory containing libz3.so.

    Search order:
      1. $Z3_PREFIX env var         → use <Z3_PREFIX>/lib
      2. Z3_DEFAULT_PREFIX constant → ~/.local/lib/python3.13/site-packages/z3/lib
         (standard location for: pip install --user z3-solver)
    """
    env_prefix = os.environ.get("Z3_PREFIX")
    prefix = Path(env_prefix).expanduser() if env_prefix else Z3_DEFAULT_PREFIX
    lib_dir = prefix / "lib"

    print(f"[smlp build] Looking for libz3.so in: {lib_dir}")

    found = list(lib_dir.rglob("libz3.so")) if lib_dir.exists() else []
    if found:
        print(f"[smlp build] Using z3 lib dir: {lib_dir}")
        _write_z3_pc(lib_dir)
        return lib_dir

    sys.exit(
        f"[smlp build] ERROR: libz3.so not found at {lib_dir}.\n"
        "Install z3-solver with: python3.13 -m pip install --user z3-solver\n"
        "Or set Z3_PREFIX to your z3 package directory, e.g.:\n"
        "  export Z3_PREFIX=~/.local/lib/python3.13/site-packages/z3"
    )


def _write_native_file(boost_prefix: Path, gmp_prefix: Path, z3_lib: Path, z3_bin: Path, build_tmp: Path) -> Path:
    """
    Write a Meson native file that points to the user-space Boost install.
    This is the most reliable way to pass non-standard library paths to Meson —
    more reliable than environment variables, which Meson may ignore depending
    on version and platform.
    """
    boost_lib = boost_prefix / "lib"
    boost_inc = boost_prefix / "include"
    gmp_lib   = gmp_prefix / "lib"
    gmp_inc   = gmp_prefix / "include"
    z3_pc_dir = z3_lib / "pkgconfig"

    native_file = build_tmp / "native.ini"
    native_file.write_text(
        "[properties]\n"
        f"boost_root = '{boost_prefix}'\n"
        f"boost_includedir = '{boost_inc}'\n"
        f"boost_librarydir = '{boost_lib}'\n"
        f"gmp_includedir = '{gmp_inc}'\n"
        f"gmp_librarydir = '{gmp_lib}'\n"
        f"gmpxx_includedir = '{gmp_inc}'\n"
        f"gmpxx_librarydir = '{gmp_lib}'\n"
        "\n"
        "[binaries]\n"
        f"python = '{sys.executable}'\n"
        f"python3 = '{sys.executable}'\n"
        f"pkg-config = 'pkg-config'\n"
        f"z3 = '{z3_bin}'\n"
        "\n"
        "[built-in options]\n"
        f"pkg_config_path = ['{gmp_lib / 'pkgconfig'}', '{boost_lib / 'pkgconfig'}', '{z3_pc_dir}']\n"
        f"c_args = ['-I{gmp_inc}', '-I{boost_inc}']\n"
        f"cpp_args = ['-I{gmp_inc}', '-I{boost_inc}']\n"
        f"c_link_args = ['-L{gmp_lib}', '-L{boost_lib}', '-Wl,-rpath,{gmp_lib}', '-Wl,-rpath,{boost_lib}']\n"
        f"cpp_link_args = ['-L{gmp_lib}', '-L{boost_lib}', '-Wl,-rpath,{gmp_lib}', '-Wl,-rpath,{boost_lib}', '-Wl,-rpath,{z3_lib}']\n"
    )
    print(f"[smlp build] Wrote Meson native file: {native_file}")
    return native_file


def _meson_build(poly_dir: Path, kay_dir: Path,
                 boost_prefix: Path, build_tmp: Path) -> Path:
    """
    Run meson setup + ninja install.
    Returns the path to the installed smlp package directory.
    """
    meson_build_dir = poly_dir / "build"
    install_prefix  = build_tmp.resolve() / "smlp_install"

    if meson_build_dir.exists():
        shutil.rmtree(meson_build_dir)

    z3_lib     = _z3_prefix()
    z3_bin     = _z3_binary()
    gmp_prefix = _gmp_prefix()
    env = _boost_env(boost_prefix)
    env = _add_z3_to_env(env, z3_lib)
    env = _add_gmp_to_env(env, gmp_prefix)

    # Embed RPATH into the built .so so it finds user-space libs at runtime
    # without needing LD_LIBRARY_PATH to be set.
    rpath_dirs = [
        str(boost_prefix / "lib"),
        str(gmp_prefix / "lib"),
        str(z3_lib),
    ]
    rpath_flags = ":".join(f"-Wl,-rpath,{d}" for d in rpath_dirs)
    existing_ldflags = env.get("LDFLAGS", "")
    env["LDFLAGS"] = f"{rpath_flags} {existing_ldflags}".strip()
    native_file = _write_native_file(boost_prefix, gmp_prefix, z3_lib, z3_bin, build_tmp)

    meson_flags = [
        "--wipe",
        f"--native-file={native_file}",
        f"-Dkay-prefix={kay_dir}",
        "-Dz3=enabled",
        "--prefix", str(install_prefix),
        # Explicitly pass both source dir and build dir as absolute paths
        # so Meson works correctly regardless of cwd
        str(poly_dir),
        str(poly_dir / "build"),
    ]

    print(f"[smlp build] PKG_CONFIG_PATH = {env.get('PKG_CONFIG_PATH', '(not set)')}")
    print(f"[smlp build] LD_LIBRARY_PATH  = {env.get('LD_LIBRARY_PATH', '(not set)')}")
    _run(
        _meson_bin(build_tmp) + ["setup"] + meson_flags,
        env=env,
    )

    _run([_ninja_bin(), "-C", str(poly_dir / "build"), "install"],
         cwd=str(poly_dir), env=env)

    # Locate the installed smlp package (Meson may use a versioned python path)
    candidates = (list(install_prefix.glob("lib/python*/dist-packages/smlp")) +
                  list(install_prefix.glob("lib/python3/dist-packages/smlp")))
    if not candidates:
        sys.exit(
            f"[smlp build] ERROR: could not find installed smlp package under "
            f"{install_prefix}. Check the Meson/Ninja output above."
        )
    return candidates[0]


# ---------------------------------------------------------------------------
# Custom build_ext
# ---------------------------------------------------------------------------

class MesonBuildExt(_build_ext):

    def run(self):
        build_tmp = Path(self.build_temp).resolve()
        build_tmp.mkdir(parents=True, exist_ok=True)

        # 1. Boost (compiled from source, cached in ~/.local/boost_py313)
        boost_prefix = _boost_prefix()

        # 2. kay
        kay_dir = _ensure_kay(build_tmp)

        # 3. Meson build – run from within the repo
        poly_dir = REPO_ROOT / "utils" / "poly"
        if not poly_dir.is_dir():
            sys.exit(
                f"[smlp build] ERROR: expected utils/poly/ at {poly_dir}.\n"
                "Make sure setup.py is run from the root of the smlp repository."
            )

        # Optionally switch branch (useful in CI)
        branch = os.environ.get("SMLP_BRANCH")
        if branch:
            _run(["git", "switch", branch], cwd=str(REPO_ROOT))
        else:
            result = subprocess.run(
                ["git", "branch", "-r", "--list", "origin/smlp_python313"],
                capture_output=True, text=True, cwd=str(REPO_ROOT)
            )
            if result.stdout.strip():
                _run(["git", "switch", "smlp_python313"], cwd=str(REPO_ROOT))

        installed_pkg = _meson_build(poly_dir, kay_dir, boost_prefix, build_tmp)

        # 4. Copy into the wheel's lib tree
        dest = Path(self.build_lib) / "smlp"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(str(installed_pkg), str(dest))
        print(f"[smlp build] smlp extension copied to wheel at {dest}")


# ---------------------------------------------------------------------------
# setup()
# ---------------------------------------------------------------------------

setup(
    cmdclass={"build_ext": MesonBuildExt},
    # Dummy extension so setuptools produces a platform-specific wheel
    # and actually invokes build_ext.
    ext_modules=[
        __import__("setuptools").Extension(name="smlp._dummy", sources=[]),
    ],
)
