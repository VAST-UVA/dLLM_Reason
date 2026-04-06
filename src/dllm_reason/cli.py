"""Console script entry points.

Thin wrappers that delegate to the scripts/ modules.
"""

import subprocess
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"


def _run(script: str) -> None:
    sys.exit(subprocess.call([sys.executable, str(_SCRIPTS_DIR / script)] + sys.argv[1:]))


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
