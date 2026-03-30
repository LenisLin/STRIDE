"""Migration shim for the legacy `slotar.exceptions` import path."""
from __future__ import annotations

from .compat.exceptions import *  # noqa: F401,F403
