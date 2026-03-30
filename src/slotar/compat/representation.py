"""Compatibility wrappers for the older representation-module path."""
from __future__ import annotations

from ..state_space import StateAxis, build_local_state_features, learn_shared_state_axis


def build_community_features(adata: object, k: int = 20) -> None:
    """Compatibility wrapper over `build_local_state_features(...)`."""
    build_local_state_features(adata, k=k, write_compat_aliases=True)  # type: ignore[arg-type]
    return None


def learn_global_prototypes(
    adata: object,
    n_bal: int,
    K: int,
    random_state: int = 42,
) -> None:
    """Compatibility wrapper over `learn_shared_state_axis(...)`."""
    learn_shared_state_axis(  # type: ignore[arg-type]
        adata,
        n_bal=n_bal,
        K=K,
        random_state=random_state,
        write_compat_aliases=True,
    )
    return None


__all__ = [
    "StateAxis",
    "build_community_features",
    "build_local_state_features",
    "learn_global_prototypes",
    "learn_shared_state_axis",
]
