"""dLLM-Reason: DAG-guided discrete diffusion language models for reasoning.

Core public API
---------------
.. code-block:: python

    from dllm_reason.graph.dag import TokenDAG
    from dllm_reason.scheduler.dag_scheduler import DAGScheduler
    from dllm_reason.models.llada import LLaDAWrapper

Version
-------
Use ``dllm_reason.__version__`` to check the installed version.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("dllm-reason")
except PackageNotFoundError:  # running from source without install
    __version__ = "0.0.0.dev"

__all__ = ["__version__"]
