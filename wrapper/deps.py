"""Fetch what a wrapper/ model or post-processing stage needs, on first use.

``ensure_deps(name)`` (``name`` is a ``WRAPPERS`` key, ``"any2full"`` or
``"resplat"``) runs before the stage's module is imported:

1. its git submodules under ``third_party/`` are checked out if empty (a clone
   without ``--recursive``);
2. the packages of pyproject.toml's ``[<name>]`` extra that are missing or too
   old are pip-installed into the running interpreter (pyproject.toml is the
   single list; nothing already satisfied is upgraded);
3. ReSplat only: if its CUDA extensions (gsplat, pointops) or many-views patch
   are missing, ``scripts/install_resplat.sh`` builds them (a few minutes, once).

Weights are not handled here: each wrapper downloads its own on first load.
"""

import importlib
import importlib.metadata
import importlib.util
import os
import re
import subprocess
import sys

from packaging.requirements import Requirement

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# third_party/ submodules each stage imports from.
SUBMODULES = {
    "vggt": ("vggt",),
    "vggt_omega": ("vggt-omega", "vggt"),  # reuses vggt's image/geometry utils
    "dvlt": ("dvlt",),
    "da3": ("depth_anything_3",),
    "mapanything": ("mapanything",),
    "pi3x": ("pi3",),
    "any2full": ("Any2Full",),
    "resplat": ("resplat",),
}


def ensure_deps(name: str) -> None:
    """Make ``name``'s submodules and pip packages available."""
    ensure_submodules(*SUBMODULES[name])
    ensure_extra(name)
    if name == "resplat" and not _resplat_ready():
        print(
            "⏳ First use of ReSplat: building its CUDA extensions (a few minutes)",
            flush=True,
        )
        script = os.path.join(_ROOT, "scripts", "install_resplat.sh")
        env = {**os.environ, "PYTHON": sys.executable}
        if subprocess.run(["bash", script], env=env).returncode:
            raise RuntimeError(
                "Building ReSplat failed (see the message above). Fix that, then "
                "rerun or run `bash scripts/install_resplat.sh` yourself."
            )
        importlib.invalidate_caches()


def ensure_submodules(*names: str) -> None:
    """Check out each empty ``third_party/<name>`` submodule."""
    for name in names:
        path = os.path.join(_ROOT, "third_party", name)
        if os.path.isdir(path) and os.listdir(path):
            continue
        print(f"⏳ First use: fetching submodule third_party/{name} ...", flush=True)
        cmd = ["git", "-C", _ROOT, "submodule", "update", "--init"]
        if subprocess.run([*cmd, f"third_party/{name}"]).returncode != 0:
            raise RuntimeError(
                f"Could not fetch third_party/{name} (see git's message above). "
                "EPO needs a git clone for this: git clone --recursive "
                "https://github.com/mattiadurso/epo.git"
            )


def ensure_extra(extra: str) -> None:
    """Pip-install the packages of pyproject's ``[extra]`` that are not satisfied."""
    missing = [req for req in _extra_requirements(extra) if not _satisfied(req)]
    if not missing:
        return
    print(f"⏳ First use of '{extra}': installing {' '.join(missing)} ...", flush=True)
    if subprocess.run([sys.executable, "-m", "pip", "install", *missing]).returncode:
        raise RuntimeError(
            f"pip could not install the '{extra}' packages (see its message above); "
            f'install them yourself with: pip install -e "{_ROOT}[{extra}]"'
        )
    importlib.invalidate_caches()


def _normalize(name: str) -> str:
    """PEP 685 extra-name normalization (``vggt_omega`` == ``vggt-omega``)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _extra_requirements(extra: str) -> list[str]:
    """Requirement strings of ``extra`` in pyproject.toml ([] if it has none)."""
    with open(os.path.join(_ROOT, "pyproject.toml"), "rb") as f:
        extras = tomllib.load(f)["project"].get("optional-dependencies", {})
    for key, reqs in extras.items():
        if _normalize(key) == _normalize(extra):
            return reqs
    return []


def _satisfied(requirement: str) -> bool:
    """Whether an installed distribution meets ``requirement``."""
    req = Requirement(requirement)
    if req.marker is not None and not req.marker.evaluate({"extra": ""}):
        return True
    try:
        version = importlib.metadata.version(req.name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return req.specifier.contains(version, prereleases=True)


def _resplat_ready() -> bool:
    """ReSplat's CUDA extensions are built and its many-views patch applied."""
    if any(importlib.util.find_spec(m) is None for m in ("gsplat", "pointops")):
        return False
    patch = os.path.join(_ROOT, "third_party", "patches", "resplat_many_views.patch")
    resplat = os.path.join(_ROOT, "third_party", "resplat")
    check = ["git", "-C", resplat, "apply", "--reverse", "--check", patch]
    return subprocess.run(check, capture_output=True).returncode == 0
