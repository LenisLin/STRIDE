"""Cohort-block recurrence loss helpers for STRIDE.

Task: expose cohort common-structure recurrence over ``T_p = [A_p | d_p]`` and
``e_p``. Reference: ``docs/stride_design_freeze.md`` defines ``L_T``,
``L_e_rec``, ``L_recurrence_raw = L_T + L_e_rec``, and
``L_cohort = L_recurrence_raw / s_cohort``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ._constants import S_COHORT
from ._parameters import _require_torch, _validate_parameters, _validate_raw_loss
from ._totals import CohortLossLedger

if TYPE_CHECKING:  # pragma: no cover
    from ._parameters import ADEState


def compute_recurrence_raw(params: "ADEState") -> CohortLossLedger:
    """Return recurrence loss over ``T=[A|d]`` and ``e`` consensus structure."""
    torch_module = _require_torch()
    A, d, e, patient_ids = _validate_parameters(params)
    T = torch_module.cat([A, d.unsqueeze(2)], dim=2)
    T_bar = T.mean(dim=0)
    A_bar = A.mean(dim=0)
    d_bar = d.mean(dim=0)
    e_bar = e.mean(dim=0)
    per_patient_T = ((T - T_bar.unsqueeze(0)) ** 2).sum(dim=2).mean(dim=1)
    per_patient_e = ((e - e_bar.unsqueeze(0)) ** 2).mean(dim=1)
    L_T = per_patient_T.mean()
    L_e_rec = per_patient_e.mean()
    raw = L_T + L_e_rec
    cohort_scaled = raw / S_COHORT
    _validate_raw_loss(raw, name="L_recurrence_raw")
    _validate_raw_loss(cohort_scaled, name="L_cohort")
    return CohortLossLedger(
        raw=raw,
        L_T=L_T,
        L_e_rec=L_e_rec,
        cohort_scaled=cohort_scaled,
        dispersion=raw,
        support_n_patients=len(patient_ids),
        T_bar=T_bar,
        A_bar=A_bar,
        d_bar=d_bar,
        e_bar=e_bar,
        per_patient_dispersion=per_patient_T + per_patient_e,
    )


__all__ = ["S_COHORT", "CohortLossLedger", "compute_recurrence_raw"]
