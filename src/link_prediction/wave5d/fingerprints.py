"""Stable data fingerprints for architecture-audit leakage control."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_audit_fingerprint_paths(root: Path) -> dict[str, Path]:
    """Return canonical artifact paths used by the L2 architecture audit."""
    candidates = root / "outputs/wave5d_link/candidates/frozen_candidates.csv"
    # Patient splits live under GSN project; also accept local mirror.
    local_splits = root / "outputs/splits/patient_splits.csv"
    hcr_cache = root / "outputs/hcr/structural_latent_pairwise_aby.csv"
    return {
        "candidate_registry": candidates,
        "patient_split_local": local_splits,
        "hcr_cache": hcr_cache,
    }


def log_audit_fingerprints(root: Path, patient_split_path: Path | None = None) -> dict[str, str]:
    paths = resolve_audit_fingerprint_paths(root)
    out: dict[str, str] = {}
    cand = paths["candidate_registry"]
    if cand.exists():
        out["candidate_registry_sha256"] = sha256_file(cand)
        print(f"candidate_registry_sha256= {out['candidate_registry_sha256']}")
    else:
        print(f"candidate_registry_sha256= MISSING ({cand})")
        out["candidate_registry_sha256"] = "MISSING"

    split = patient_split_path or paths["patient_split_local"]
    if split is not None and Path(split).exists():
        out["patient_split_sha256"] = sha256_file(split)
        print(f"patient_split_sha256= {out['patient_split_sha256']}")
    else:
        print(f"patient_split_sha256= MISSING ({split})")
        out["patient_split_sha256"] = "MISSING"

    hcr = paths["hcr_cache"]
    if hcr.exists():
        out["hcr_cache_sha256"] = sha256_file(hcr)
        print(f"hcr_cache_sha256= {out['hcr_cache_sha256']}")
    else:
        # HCR may be recomputed in-memory; fingerprint the frozen candidate table instead.
        print(f"hcr_cache_sha256= MISSING ({hcr}) — will fingerprint in-memory attach")
        out["hcr_cache_sha256"] = "MISSING"
    return out
