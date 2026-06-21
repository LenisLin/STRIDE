from __future__ import annotations

import stride
import stride.tl as tl
from stride.errors import ContractError


def test_root_api_exports_current_formal_tl_surface() -> None:
    assert stride.fit is tl.fit
    assert not hasattr(stride, "fit_stride")
    assert stride.FitResult is tl.FitResult
    assert stride.RelationResult is tl.RelationResult
    assert stride.CohortResult is tl.CohortResult
    assert stride.ContractError is ContractError
    assert isinstance(stride.__version__, str)
    assert stride.__all__ == (
        "__version__",
        "ContractError",
        "fit",
        "FitResult",
        "RelationResult",
        "CohortResult",
    )
