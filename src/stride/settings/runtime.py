"""Runtime settings surface for STRIDE execution controls."""
from __future__ import annotations

from dataclasses import dataclass

from ..errors import ContractError

_ALLOWED_UOT_BACKENDS: tuple[str, ...] = ("numpy", "torch")


@dataclass(frozen=True)
class RuntimeSettings:
    """Light runtime controls for UOT execution, chunking, and worker limits."""

    uot_backend: str = "numpy"
    device: str = "cpu"
    max_calibration_workers: int = 2
    plan_chunk_elements: int | None = None

    def __post_init__(self) -> None:
        backend = str(self.uot_backend).strip().lower()
        if backend not in _ALLOWED_UOT_BACKENDS:
            raise ContractError(
                f"RuntimeSettings.uot_backend must be one of {_ALLOWED_UOT_BACKENDS}, got {self.uot_backend!r}"
            )
        object.__setattr__(self, "uot_backend", backend)

        device = str(self.device).strip()
        if device == "":
            raise ContractError("RuntimeSettings.device must be a non-empty string")
        object.__setattr__(self, "device", device)

        if int(self.max_calibration_workers) <= 0:
            raise ContractError("RuntimeSettings.max_calibration_workers must be a positive integer")
        object.__setattr__(self, "max_calibration_workers", int(self.max_calibration_workers))

        if self.plan_chunk_elements is not None:
            if int(self.plan_chunk_elements) <= 0:
                raise ContractError("RuntimeSettings.plan_chunk_elements must be positive when provided")
            object.__setattr__(self, "plan_chunk_elements", int(self.plan_chunk_elements))

    def resolved_max_calibration_workers(self, n_candidates: int) -> int:
        """Return a worker count capped by the candidate-grid size."""
        candidate_count = int(n_candidates)
        if candidate_count <= 0:
            raise ContractError("n_candidates must be positive")
        return max(1, min(int(self.max_calibration_workers), candidate_count))

    def resolved_execution(self) -> tuple[str, str]:
        """Return the actual backend/device pair that execution will use."""
        requested_device = str(self.device).strip()
        if self.uot_backend != "torch":
            return "numpy", "cpu"

        try:
            import torch
        except ImportError:
            return "numpy", "cpu"

        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            return "torch", "cpu"
        return "torch", requested_device

    def resolved_execution_hardware(self) -> str:
        """Return whether execution will use CPU or GPU hardware."""
        _backend, resolved_device = self.resolved_execution()
        return "gpu" if str(resolved_device).startswith(("cuda", "mps")) else "cpu"

    def execution_metadata(self) -> dict[str, str]:
        """Return requested/resolved execution metadata for diagnostics."""
        resolved_backend, resolved_device = self.resolved_execution()
        return {
            "requested_uot_backend": self.uot_backend,
            "requested_device": self.device,
            "uot_backend": resolved_backend,
            "device": resolved_device,
            "execution_hardware": self.resolved_execution_hardware(),
        }

    def resolved_plan_chunk_elements(self, *, fallback: int) -> int:
        """Return the configured plan-chunk budget, falling back to legacy defaults."""
        if int(fallback) <= 0:
            raise ContractError("fallback plan_chunk_elements must be positive")
        if self.plan_chunk_elements is None:
            return int(fallback)
        return int(self.plan_chunk_elements)

    def resolved_plan_chunk_rows(
        self,
        n_proto: int,
        *,
        fallback_plan_chunk_elements: int,
    ) -> int:
        """Return the maximum number of pair rows per dense [K, K] plan chunk."""
        n_states = int(n_proto)
        if n_states <= 0:
            raise ContractError("n_proto must be a positive integer")
        plan_elements = self.resolved_plan_chunk_elements(fallback=fallback_plan_chunk_elements)
        return max(1, int(plan_elements // (n_states * n_states)))


__all__ = ["RuntimeSettings"]
