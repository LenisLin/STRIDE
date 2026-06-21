"""Package version resolution."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

_FALLBACK_VERSION = "0.1.0"

try:
    __version__ = version("stride")
except PackageNotFoundError:
    __version__ = _FALLBACK_VERSION

__all__ = ("__version__",)
