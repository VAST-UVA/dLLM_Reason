"""Console script entry points for dllm-reason.

These thin wrappers delegate to the scripts/ directory.  They work correctly
whether the package is installed in editable mode (``pip install -e .``) or as
a regular wheel (``pip install git+https://...``).

Resolution order for the scripts directory:
  1. The sibling ``scripts/`` folder of the repository root — present when
     running from a git checkout or ``pip install -e .``.
  2. The ``scripts/`` folder bundled inside the installed package (populated
     via ``[tool.setuptools.package-data]`` when building a wheel).
"""

import importlib.resources
import subprocess
import sys
from pathlib import Path


def _find_scripts_dir() -> Path:
    """Return the path to the scripts/ directory, however the package was installed."""
    # Strategy 1: repo checkout / editable install
    # __file__ = src/dllm_reason/cli.py  → three parents up → repo root
    candidate = Path(__file__).resolve().parent.parent.parent / "scripts"
    if candidate.is_dir():
        return candidate

    # Strategy 2: wheel install — scripts are copied into the package data dir
    try:
        # Python 3.9+ path; falls back gracefully
        with importlib.resources.path("dllm_reason", "scripts") as p:
            if Path(p).is_dir():
                return Path(p)
    except (FileNotFoundError, TypeError):
        pass

    raise RuntimeError(
        "Cannot find the dllm-reason scripts directory. "
        "Please install in editable mode: pip install -e ."
    )


def _run(script: str) -> None:
    scripts_dir = _find_scripts_dir()
    sys.exit(subprocess.call([sys.executable, str(scripts_dir / script)] + sys.argv[1:]))


def train():
    _run("train.py")


def evaluate():
    _run("evaluate.py")


def eval_dags():
    _run("eval_dags.py")


def search_dag():
    _run("search_dag.py")


def visualize_dag():
    _run("visualize_dag.py")
