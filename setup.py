#!/usr/bin/env python3
"""
setup.py for the smlp package  –  macOS edition.

System prerequisites (install once)
-------------------------------------
  brew install gcc git make m4 pkg-config gmp
  # Python 3.11 via Homebrew:
  brew install python@3.11

User prerequisites (no sudo, install once)
------------------------------------------
  python3.11 -m pip install meson ninja z3-solver delocate

Build flow
----------
1.  Boost.Python 1.83 is compiled from source for Python 3.11 and cached in
    ~/.local/boost_py311  (or the path in $BOOST_CACHE_DIR).
    The build is skipped on subsequent runs if the cache directory already
    contains the marker file  .built_for_python<major><minor>.
    Set $BOOST_ROOT to point at an existing Boost prefix to skip this step.
    On macOS the bootstrap uses clang (the default compiler).

2.  The 'kay' C++ dependency is cloned from GitHub into the pip build-temp
    directory (or reused from $KAY_DIR).

3.  `meson setup` + `ninja install` is run inside  utils/poly/  of the
    repository this setup.py lives in.

4.  The installed smlp extension package is copied into the wheel.

Environment variables
---------------------
BOOST_ROOT       Reuse an existing Boost prefix – skips download + compile.
BOOST_CACHE_DIR  Where to cache the compiled Boost (default: ~/.local/boost_py311).
BOOST_VERSION    Boost version to download (default: 1.83.0).
KAY_DIR          Reuse an existing kay checkout.
GMP_ROOT         Reuse an existing GMP prefix (default: $(brew --prefix)/opt/gmp).
GMP_CACHE_DIR    Where to cache compiled GMP (default: ~/.local/gmp).
GMP_VERSION      GMP version to download (default: 6.3.0).
Z3_PREFIX        Reuse an existing Z3 install prefix – skips pip z3-solver.
#Z3_VERSION       Z3 version (default: 4.8.12).
Z3_VERSION       Z3 version (default: 4.16.0).
Z3_BIN_DIR       Path to directory containing z3 binary.
SMLP_BRANCH      Git branch to switch to in the smlp repo.
DLD_PATH         workaround of MacOS purging DYLD_LIBRARY_PATH when calling python processes
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

print(os.environ)


# ---------------------------------------------------------------------------
# Platform guard
# ---------------------------------------------------------------------------
IS_MACOS = platform.system() == "Darwin"
if not IS_MACOS:
    raise RuntimeError(
        "This setup.py is the macOS variant. "
        "Use the manylinux_2_28/setup.py on Linux."
    )

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

BOOST_VERSION   = os.environ.get("BOOST_VERSION", "1.83.0")
BOOST_CACHE_DIR = Path(
    os.environ.get("BOOST_CACHE_DIR", Path.home() / ".local" / "boost_py311")
).expanduser()


# On macOS, z3-solver installs into the active Python's site-packages
Z3_DEFAULT_PREFIX = Path(
    subprocess.check_output(
        [sys.executable, "-c", "import site; print(site.getsitepackages()[0])"],
        text=True
    ).strip()
) / "z3"

GMP_VERSION   = os.environ.get("GMP_VERSION", "6.3.0")

# ---------------------------------------------------------------------------
# setting DYLD_LIBRARY_PATH
# ---------------------------------------------------------------------------

os.environ['DYLD_LIBRARY_PATH']=os.environ['DLD_PATH']
env = os.environ
print(env)

 
# Prefer Homebrew GMP so we can skip the source build
def _homebrew_gmp() -> str | None:
    try:
        prefix = subprocess.check_output(
            ["brew", "--prefix", "gmp"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        if prefix:
            return prefix
    except Exception:
        pass
    return None

_brew_gmp = _homebrew_gmp()
GMP_CACHE_DIR = Path(
    os.environ.get(
        "GMP_CACHE_DIR",
        _brew_gmp if _brew_gmp else str(Path.home() / ".local" / "gmp")
    )
).expanduser()

#Z3_VERSION    = os.environ.get("Z3_VERSION", "4.8.12")
Z3_VERSION    = os.environ.get("Z3_VERSION", "4.16.0")

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
        if hasattr(result, "stdout") and result.stdout:
            print(result.stdout)
        if hasattr(result, "stderr") and result.stderr:
            print(result.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd)


def _verify_tarball(path: Path) -> bool:
    try:
        with tarfile.open(path) as tf:
            tf.getmembers()
        return True
    except Exception:
        return False


def _download(url: str, dest: Path, retries: int = 5) -> None:
    import time
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
    """Write a meson wrapper script and return the command to invoke it."""
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

    if not mesonbuild_location:
        user_site = (
            Path.home() / ".local" / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        )
        if (user_site / "mesonbuild").exists():
            mesonbuild_location = str(user_site)

    if not mesonbuild_location:
        import importlib.util
        spec = importlib.util.find_spec("mesonbuild")
        if spec and spec.submodule_search_locations:
            mesonbuild_location = str(Path(list(spec.submodule_search_locations)[0]).parent)

    if not mesonbuild_location:
        from shutil import which
        meson_bin = which("meson")
        if meson_bin:
            print(f"[smlp build] Using meson from PATH: {meson_bin}")
            return [meson_bin]

    if not mesonbuild_location:
        raise RuntimeError(
            f"[smlp build] meson not found. Run:  python{sys.version_info.major}.{sys.version_info.minor} -m pip install meson"
        )

    print(f"[smlp build] meson location: {mesonbuild_location}")

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
    import importlib.util
    from shutil import which

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

    user_ninja = Path.home() / ".local" / "bin" / "ninja"
    if user_ninja.exists():
        print(f"[smlp build] Using user ninja: {user_ninja}")
        return str(user_ninja)

    found = which("ninja")
    if found:
        print(f"[smlp build] Using ninja from PATH: {found}")
        return found

    raise RuntimeError("[smlp build] ninja not found. Run:  pip install ninja")


# ---------------------------------------------------------------------------
# Step 1 – Boost.Python (compiled from source, cached in user-space)
# ---------------------------------------------------------------------------

def _boost_prefix() -> Path:
    print(f"[KK _boost_prefix]")
    env_root = os.environ.get("BOOST_ROOT")
    if env_root:
        prefix = Path(env_root).expanduser()
        print(f"[smlp build] Using BOOST_ROOT={prefix}")
        return prefix

    py_tag = f"python{sys.version_info.major}{sys.version_info.minor}"
    tag_file = BOOST_CACHE_DIR / f".built_for_{py_tag}"
    if tag_file.exists():
        print(f"[smlp build] Boost cache found at {BOOST_CACHE_DIR}, skipping build.")
        return BOOST_CACHE_DIR

    ver_flat     = BOOST_VERSION.replace(".", "_")
    tarball_name = f"boost_{ver_flat}.tar.gz"
    url          = f"https://archives.boost.io/release/{BOOST_VERSION}/source/{tarball_name}"

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
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    py_inc = subprocess.check_output(
        [sys.executable, "-c",
         "import sysconfig; print(sysconfig.get_path('include'))"],
        text=True
    ).strip()

    # On macOS we do NOT disable libpython linking – the dylib exists and is
    # required.  We also omit the manylinux-specific 'without-<python>' flag.
    user_config = src / "user-config.jam"
    user_config.write_text(
        f"using python : {py_ver} : {sys.executable} : {py_inc} : ;\n"
    )

    _run(
        ["./b2", "install",
         f"--prefix={BOOST_CACHE_DIR}",
         "--with-python",
         f"python={py_ver}"],
        cwd=str(src),
    )

    tag_file.touch()
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"[smlp build] Boost built and cached at {BOOST_CACHE_DIR}")
    return BOOST_CACHE_DIR


def _boost_env(prefix: Path) -> dict:
    print(f"[KK _boost_env]")
    lib_dir = prefix / "lib"
    inc_dir = prefix / "include"

    env = os.environ.copy()
    env["BOOST_ROOT"]       = str(prefix)
    env["BOOST_INCLUDEDIR"] = str(inc_dir)
    env["BOOST_LIBRARYDIR"] = str(lib_dir)

    env["PYTHON"]  = sys.executable
    env["PYTHON3"] = sys.executable

    py_ver = f"{sys.version_info.major}{sys.version_info.minor}"
    env["BOOST_PYTHON_LIBNAME"] = f"boost_python{py_ver}"

    # macOS uses DYLD_LIBRARY_PATH instead of LD_LIBRARY_PATH
    existing_dyld = env.get("DYLD_LIBRARY_PATH", "")
    env["DYLD_LIBRARY_PATH"] = f"{lib_dir}:{existing_dyld}" if existing_dyld else str(lib_dir)

    pkgconfig    = lib_dir / "pkgconfig"
    existing_pkg = env.get("PKG_CONFIG_PATH", "")
    env["PKG_CONFIG_PATH"] = f"{pkgconfig}:{existing_pkg}" if existing_pkg else str(pkgconfig)

    return env


def _add_z3_to_env(env: dict, z3_lib: Path) -> dict:
    print(f"[KK _add_z3_to_env]")
    """Prepend the z3-solver lib/bin directories to the relevant env vars."""
    # macOS: DYLD_LIBRARY_PATH
    existing_dyld = env.get("DYLD_LIBRARY_PATH", "")
    env["DYLD_LIBRARY_PATH"] = f"{z3_lib}:{existing_dyld}" if existing_dyld else str(z3_lib)

    pkgconfig    = z3_lib / "pkgconfig"
    existing_pkg = env.get("PKG_CONFIG_PATH", "")
#    env["PKG_CONFIG_PATH"] = f"{z3_lib}:{existing_pkg}" if existing_pkg else str(z3_lib)
    env["PKG_CONFIG_PATH"] = f"{pkgconfig}:{existing_pkg}" if existing_pkg else str(z3_lib)
    
    z3_bin = z3_lib.parent / "bin"
    existing_path = env.get("PATH", os.environ.get("PATH", ""))
    env["PATH"] = f"{z3_bin}:{existing_path}" if existing_path else str(z3_bin)
    
    print(f"[KK _add_z3_to_env: PATH]={env['PATH']}")
    return env


def _add_gmp_to_env(env: dict, gmp_prefix: Path) -> dict:
    print(f"[KK _add_gmp_to_env]")
    gmp_lib = gmp_prefix / "lib"
    gmp_inc = gmp_prefix / "include"

    existing_dyld = env.get("DYLD_LIBRARY_PATH", "")
    env["DYLD_LIBRARY_PATH"] = f"{gmp_lib}:{existing_dyld}" if existing_dyld else str(gmp_lib)

    pkgconfig    = gmp_lib / "pkgconfig"
    existing_pkg = env.get("PKG_CONFIG_PATH", "")
    env["PKG_CONFIG_PATH"] = f"{pkgconfig}:{existing_pkg}" if existing_pkg else str(pkgconfig)

    existing_cpp = env.get("CPPFLAGS", "")
    env["CPPFLAGS"] = f"-I{gmp_inc} {existing_cpp}".strip()

    existing_ldflags = env.get("LDFLAGS", "")
    env["LDFLAGS"] = f"-L{gmp_lib} {existing_ldflags}".strip()

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
# Step 1c – GMP
# ---------------------------------------------------------------------------

def _write_gmp_pc(prefix: Path) -> None:
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
    env_root = os.environ.get("GMP_ROOT")
    if env_root:
        prefix = Path(env_root).expanduser()
        print(f"[smlp build] Using GMP_ROOT={prefix}")
        _write_gmp_pc(prefix)
        return prefix

    # On macOS, prefer Homebrew GMP (already detected above)
    if _brew_gmp:
        prefix = Path(_brew_gmp)
        print(f"[smlp build] Using Homebrew GMP at {prefix}")
        _write_gmp_pc(prefix)
        return prefix

    tag_file = GMP_CACHE_DIR / ".built"
    if tag_file.exists():
        print(f"[smlp build] GMP cache found at {GMP_CACHE_DIR}, skipping build.")
        _write_gmp_pc(GMP_CACHE_DIR)
        return GMP_CACHE_DIR

    # Fallback: build from source
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

    machine = platform.machine()   # arm64 or x86_64 on macOS
    system  = platform.system().lower()  # darwin
    host    = f"{machine}-apple-{system}"

    _run(
        ["./configure",
         f"--prefix={GMP_CACHE_DIR}",
         f"--host={host}",
         "--enable-shared",
         "--enable-static",
         "--disable-assembly",
         "--enable-cxx"],
        cwd=str(src),
    )
    _run(["make", f"-j{os.cpu_count() or 1}"], cwd=str(src))
    _run(["make", "install"], cwd=str(src))

    _write_gmp_pc(GMP_CACHE_DIR)
    tag_file.touch()
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"[smlp build] GMP built and cached at {GMP_CACHE_DIR}")
    return GMP_CACHE_DIR


# ---------------------------------------------------------------------------
# Step 1d – Z3
# ---------------------------------------------------------------------------

def _z3_binary() -> Path:
    from shutil import which

    if Z3_BIN_DIR.exists() and (Z3_BIN_DIR / "z3").exists():
        print(f"[smlp build] Using z3 binary from Z3_BIN_DIR: {Z3_BIN_DIR / 'z3'}")
        return Z3_BIN_DIR / "z3"

    user_z3 = Path.home() / ".local" / "bin" / "z3"
    if user_z3.exists():
        print(f"[smlp build] Using user z3 binary: {user_z3}")
        return user_z3

    system_z3 = which("z3")
    if system_z3:
        print(f"[smlp build] Using system z3: {system_z3}")
        return Path(system_z3)

    # Download pre-built macOS binary from GitHub releases
    machine = platform.machine()
    arch_map = {"x86_64": "x64", "arm64": "arm64"}
    arch = arch_map.get(machine, machine)
    z3_release = f"z3-{Z3_VERSION}-{arch}-osx-10.16"
    url = (
        f"https://github.com/Z3Prover/z3/releases/download/z3-{Z3_VERSION}/"
        f"{z3_release}.zip"
    )

    tmp_dir = Z3_BIN_DIR.parent.parent / "_z3_build_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    zip_path = tmp_dir / f"{z3_release}.zip"

    print(f"[smlp build] Downloading z3 binary {Z3_VERSION} ...")
    _download(url, zip_path)

    import zipfile
    print(f"[smlp build] Extracting z3 binary ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp_dir)

    Z3_BIN_DIR.mkdir(parents=True, exist_ok=True)
    src_bin = tmp_dir / z3_release / "bin" / "z3"
    shutil.copy2(src_bin, Z3_BIN_DIR / "z3")
    (Z3_BIN_DIR / "z3").chmod(0o755)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"[smlp build] z3 binary installed at {Z3_BIN_DIR / 'z3'}")
    return Z3_BIN_DIR / "z3"


def _write_z3_pc(z3_lib: Path) -> None:
    import re
    print(f"[KK _write_z3_pc]")
    # On macOS, dylibs are named libz3.dylib (no version in the name)
    version = Z3_VERSION  # use known version directly
    for f in list(z3_lib.glob("libz3.*.dylib")) + list(z3_lib.glob("libz3.so.*")):
        m = re.search(r"libz3[._]([\d.]+)", f.name)
        if m:
            version = m.group(1)
            break

    prefix  = z3_lib.parent
    inc_dir = prefix / "include"

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
    print(f"[KK _z3_prefix]")
    env_prefix = os.environ.get("Z3_PREFIX")
    prefix = Path(env_prefix).expanduser() if env_prefix else Z3_DEFAULT_PREFIX
    lib_dir = prefix / "lib"

    print(f"[smlp build] Looking for libz3 in: {lib_dir}")

    # macOS dylibs
    found = (list(lib_dir.rglob("libz3.dylib")) +
             list(lib_dir.rglob("libz3.so"))) if lib_dir.exists() else []
    if found:
        print(f"[smlp build] Using z3 lib dir: {lib_dir}")
        _write_z3_pc(lib_dir)
        return lib_dir

    sys.exit(
        f"[smlp build] ERROR: libz3 not found at {lib_dir}.\n"
        "Install z3-solver with: python3.11 -m pip install z3-solver\n"
        "Or set Z3_PREFIX to your z3 package directory."
    )


def _write_native_file(boost_prefix: Path, gmp_prefix: Path, z3_lib: Path,
                       z3_bin: Path, build_tmp: Path) -> Path:
    print(f"[KK _write_native_file]")
    boost_lib = boost_prefix / "lib"
    boost_inc = boost_prefix / "include"
    gmp_lib   = gmp_prefix / "lib"
    gmp_inc   = gmp_prefix / "include"
    z3_pc_dir = z3_lib / "pkgconfig"

    # macOS rpath flag uses @rpath or absolute paths
    rpath_flags_c  = [f"-Wl,-rpath,{boost_lib}", f"-Wl,-rpath,{gmp_lib}"]
    rpath_flags_cpp = rpath_flags_c + [f"-Wl,-rpath,{z3_lib}"]

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
        #KK
        f"z3_librarydir = '{z3_lib}'\n"
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
        f"c_link_args = ['-L{gmp_lib}', '-L{boost_lib}', '-L{build_tmp}', '-L{z3_lib}',"
        + ", ".join(f"'{f}'" for f in rpath_flags_c) + "]\n"
        f"cpp_link_args = ['-L{gmp_lib}', '-L{boost_lib}', '-L{build_tmp}', '-L{z3_lib}',"
        + ", ".join(f"'{f}'" for f in rpath_flags_cpp) + "]\n"
    )
    print(f"[smlp build] Wrote Meson native file: {native_file}")
    return native_file


def _meson_build(poly_dir: Path, kay_dir: Path,
                 boost_prefix: Path, build_tmp: Path) -> Path:

    print(f"[KK _meson_build]")
    meson_build_dir = poly_dir / "build"
    install_prefix  = build_tmp.resolve() / "smlp_install"

    
    if meson_build_dir.exists():
        shutil.rmtree(meson_build_dir)

    z3_lib     = _z3_prefix()
    z3_bin     = _z3_binary()
    gmp_prefix = _gmp_prefix()

    # macOS: no stub libpython needed – the dylib is real and present
    env = _boost_env(boost_prefix)
    env = _add_z3_to_env(env, z3_lib)
    env = _add_gmp_to_env(env, gmp_prefix)

    rpath_dirs = [
        str(boost_prefix / "lib"),
        str(gmp_prefix / "lib"),
        str(z3_lib),
    ]

    print(f"[KK rpath_dirs] rpath_dirs={rpath_dirs}")
    
    rpath_flags = " ".join(f"-Wl,-rpath,{d}" for d in rpath_dirs)
    existing_ldflags = env.get("LDFLAGS", "")
    env["LDFLAGS"] = f"{rpath_flags} {existing_ldflags}".strip()

    native_file = _write_native_file(boost_prefix, gmp_prefix, z3_lib, z3_bin, build_tmp)

    meson_flags = [
        "--wipe",
        f"--native-file={native_file}",
        f"-Dkay-prefix={kay_dir}",
        "-Dz3=enabled",
        "--prefix", str(install_prefix),
        str(poly_dir),
        str(poly_dir / "build"),
    ]

    print(f"[smlp build] PKG_CONFIG_PATH   = {env.get('PKG_CONFIG_PATH', '(not set)')}")
    print(f"[smlp build] DYLD_LIBRARY_PATH = {env.get('DYLD_LIBRARY_PATH', '(not set)')}")
    with open('/tmp/meson.env', 'w') as f:
        print('\n'.join(f"{k}={v}" for k,v in env.items()), file=f)
    _run(
        _meson_bin(build_tmp) + ["setup"] + meson_flags,
        env=env,
    )

    _run([_ninja_bin(), "-C", str(poly_dir / "build"), "install"],
         cwd=str(poly_dir), env=env)

    candidates = (list(install_prefix.glob("lib/python*/dist-packages/smlp")) +
                  list(install_prefix.glob("lib/python3/dist-packages/smlp")) +
                  list(install_prefix.glob("lib/python*/site-packages/smlp")) +
                  list(install_prefix.glob("lib/python3/site-packages/smlp")))
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
        print(f"[KK MesonBuildExt]")
        
        build_tmp = Path(self.build_temp).resolve()
        build_tmp.mkdir(parents=True, exist_ok=True)

        boost_prefix = _boost_prefix()
        kay_dir      = _ensure_kay(build_tmp)

        poly_dir = REPO_ROOT / "utils" / "poly"
        if not poly_dir.is_dir():
            sys.exit(
                f"[smlp build] ERROR: expected utils/poly/ at {poly_dir}.\n"
                "Make sure setup.py is run from the root of the smlp repository."
            )

        branch = os.environ.get("SMLP_BRANCH")
        if branch:
            _run(["git", "switch", branch], cwd=str(REPO_ROOT))
        else:
            py_ver = f"{sys.version_info.major}{sys.version_info.minor}"
            auto_branch = f"smlp_python{py_ver}_mac"
            result = subprocess.run(
                ["git", "branch", "-r", "--list", f"origin/{auto_branch}"],
                capture_output=True, text=True, cwd=str(REPO_ROOT)
            )
            if result.stdout.strip():
                _run(["git", "switch", auto_branch], cwd=str(REPO_ROOT))

        installed_pkg = _meson_build(poly_dir, kay_dir, boost_prefix, build_tmp)

        dest = Path(self.build_lib) / "smlp"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(str(installed_pkg), str(dest))
        print(f"[smlp build] smlp extension copied to wheel at {dest}")

        smlp_py_src = REPO_ROOT / "src" / "smlp_py"
        if smlp_py_src.is_dir():
            smlp_py_dest = dest / "smlp_py"
            if smlp_py_dest.exists():
                shutil.rmtree(smlp_py_dest)
            shutil.copytree(str(smlp_py_src), str(smlp_py_dest))
            print(f"[smlp build] smlp_py source copied to wheel at {smlp_py_dest}")
        else:
            print(f"[smlp build] WARNING: src/smlp_py not found at {smlp_py_src}, skipping.")

        run_smlp_src = REPO_ROOT / "src" / "run_smlp.py"
        if run_smlp_src.is_file():
            shutil.copy2(str(run_smlp_src), str(dest / "run_smlp.py"))
            print(f"[smlp build] run_smlp.py copied to wheel at {dest / 'run_smlp.py'}")
        else:
            print(f"[smlp build] WARNING: src/run_smlp.py not found at {run_smlp_src}, skipping.")


# ---------------------------------------------------------------------------
# setup()
# ---------------------------------------------------------------------------

setup(
    cmdclass={"build_ext": MesonBuildExt},
    ext_modules=[
        __import__("setuptools").Extension(name="smlp._dummy", sources=[]),
    ],
)
