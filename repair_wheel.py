#!/usr/bin/env python3.13
"""
Run auditwheel repair on the smlp wheel in dist/ to produce a
self-contained manylinux wheel with all .so dependencies bundled.

Usage:
    python3.13 repair_wheel.py [dist_dir]

dist_dir defaults to 'dist/'.
"""
import sys
import subprocess
from pathlib import Path


def _find_auditwheel_location() -> str | None:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", "auditwheel"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if line.startswith("Location:"):
                return line.split(":", 1)[1].strip()

    user_site = (
        Path.home() / ".local" / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    if (user_site / "auditwheel").exists():
        return str(user_site)

    return None


def main():
    dist_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "dist")

    location = _find_auditwheel_location()
    if not location:
        print("ERROR: auditwheel not found.")
        print("Install with: python3.13 -m pip install --user auditwheel patchelf")
        sys.exit(1)

    wheels = sorted(dist_dir.glob("smlp-*linux_x86_64.whl"), key=lambda p: p.stat().st_mtime)
    if not wheels:
        print(f"ERROR: No linux_x86_64 wheel found in {dist_dir}/")
        sys.exit(1)

    wheel = wheels[-1]
    print(f"Repairing {wheel} ...")
    subprocess.check_call([
        sys.executable, "-c",
        f"import sys; sys.path.insert(0, {location!r}); "
        f"from auditwheel.main import main; sys.exit(main())",
        "repair", str(wheel), "-w", str(dist_dir)
    ])
    print(f"Done. manylinux wheel saved to {dist_dir}/")


if __name__ == "__main__":
    main()
