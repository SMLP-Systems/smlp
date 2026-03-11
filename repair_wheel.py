#!/usr/bin/env python3
"""
Run delocate-wheel on the smlp wheel in dist/ to produce a
self-contained macOS wheel with all dylib dependencies bundled.

delocate is the macOS equivalent of auditwheel on Linux.

Usage:
    python3 repair_wheel.py [dist_dir] [--plat PLATFORM]

dist_dir defaults to 'dist/'.
--plat defaults to the current macOS deployment target
       (e.g. 'macosx_12_0_arm64' or 'macosx_12_0_x86_64').

Install delocate with:
    pip install delocate
"""
import sys
import os
import subprocess
import platform
import argparse
from pathlib import Path



#----------------
# setting DYLD_LIBRARY_PATH
#-----------------
# DLD_PATH is a workaround of MacOS purging DYLD_LIBRARY_PATH when calling python processes

os.environ['DYLD_LIBRARY_PATH']=os.environ['DLD_PATH']
env = os.environ
print(env)    


def _default_platform_tag() -> str:
    """
    Build a platform tag that matches the current macOS host, e.g.:
      macosx_14_0_arm64   (Apple Silicon, macOS 14)
      macosx_12_0_x86_64  (Intel, macOS 12)

    The tag is derived from MACOSX_DEPLOYMENT_TARGET if set, otherwise
    from the running OS version.
    """
    import os

    # Prefer the deployment target set in the environment (mirrors pip's logic)
    deployment_target = os.environ.get("MACOSX_DEPLOYMENT_TARGET", "")
    if deployment_target:
        major, *rest = deployment_target.split(".")
        minor = rest[0] if rest else "0"
    else:
        mac_ver = platform.mac_ver()[0]  # e.g. "14.3.1"
        parts = mac_ver.split(".")
        major = parts[0] if len(parts) > 0 else "12"
        minor = parts[1] if len(parts) > 1 else "0"

    machine = platform.machine()  # arm64 or x86_64
    return f"macosx_{major}_{minor}_{machine}"


def _find_delocate_location() -> str | None:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", "delocate"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if line.startswith("Location:"):
                return line.split(":", 1)[1].strip()

    user_site = (
        Path.home() / "Library" / "Python"
        / f"{sys.version_info.major}.{sys.version_info.minor}"
        / "lib" / "python" / "site-packages"
    )
    if (user_site / "delocate").exists():
        return str(user_site)

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dist_dir", nargs="?", default="dist")
    parser.add_argument(
        "--plat",
        default=None,
        help=(
            "Target platform tag "
            "(default: auto-detected from current macOS version and arch, "
            "e.g. macosx_14_0_arm64)"
        ),
    )
    args = parser.parse_args()

    dist_dir = Path(args.dist_dir)
    plat = args.plat or _default_platform_tag()

    location = _find_delocate_location()
    if not location:
        print("ERROR: delocate not found.")
        print("Install with: python3 -m pip install delocate")
        sys.exit(1)

    # Match any unrepaired macOS wheel (name contains 'macosx' or 'darwin',
    # but NOT 'delocated' – avoid re-repairing an already-fixed wheel)
    wheels = sorted(
        [
            w for w in dist_dir.glob("smlp-*.whl")
            if ("macosx" in w.name or "darwin" in w.name)
            and "delocated" not in w.name
        ],
        key=lambda p: p.stat().st_mtime,
    )

    # Fallback: grab whatever wheel is newest (handles pure-python or unknown tags)
    if not wheels:
        wheels = sorted(dist_dir.glob("smlp-*.whl"), key=lambda p: p.stat().st_mtime)

    if not wheels:
        print(f"ERROR: No smlp wheel found in {dist_dir}/")
        sys.exit(1)

    wheel = wheels[-1]
    print(f"Repairing {wheel} for platform {plat} ...")

     
    # delocate-wheel bundles all non-system dylibs into the wheel and
    # rewrites their install names, equivalent to auditwheel repair.
    subprocess.check_call([
        sys.executable, "-c",
        f"import sys; sys.path.insert(0, {location!r}); "
        f"from delocate.cmd.delocate_wheel import main; sys.exit(main())",
        "--wheel-dir", str(dist_dir),
        "--require-archs", platform.machine(),
        "--exclude", "libpython3.11",
        str(wheel),
    ], env=env)

    print(f"Done. Repaired wheel saved to {dist_dir}/")


if __name__ == "__main__":
    main()
