"""Wave 7 — generalized HCR with type-specific orthonormal bases."""

from .discrete_basis import audit_basis, fit_discrete_orthonormal_basis, transform_discrete_values

__all__ = [
    "audit_basis",
    "fit_and_attach_wave7",
    "fit_discrete_orthonormal_basis",
    "transform_discrete_values",
    "wave7_enabled",
]


def __getattr__(name: str):
    if name in {"fit_and_attach_wave7", "wave7_enabled"}:
        from .attach import fit_and_attach_wave7, wave7_enabled

        return {"fit_and_attach_wave7": fit_and_attach_wave7, "wave7_enabled": wave7_enabled}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
