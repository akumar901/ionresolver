"""
ionresolver.validate
--------------------
Decoy-based false discovery control for propagated annotations.

Any mass-difference search will return hits by chance: at a few ppm across a
thousand features, some pairs will line up with a biotransformation shift for
no reason at all. Without a null model there is no way to tell a method from a
coincidence.

The approach here mirrors target/decoy in proteomics. Real transformation mass
shifts are replaced with implausible ones drawn from the same magnitude range,
and the search is rerun. Hits surviving with decoy shifts are false by
construction, so their rate estimates the FDR of the real search.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .network import propagate
from .transformations import Transformation, get_transformations


@dataclass
class ValidationResult:
    """Outcome of a decoy validation run."""

    n_real: int
    decoy_mean: float
    decoy_std: float
    fdr: float
    fdr_lo: float
    fdr_hi: float
    enrichment: float
    z_score: float
    n_decoy_runs: int

    def summary(self) -> str:
        return (
            f"real hits         : {self.n_real}\n"
            f"decoy hits        : {self.decoy_mean:.1f} +/- {self.decoy_std:.1f} "
            f"(n={self.n_decoy_runs})\n"
            f"estimated FDR     : {self.fdr * 100:.1f}%  "
            f"(95% CI {self.fdr_lo * 100:.1f}-{self.fdr_hi * 100:.1f}%)\n"
            f"enrichment        : {self.enrichment:.1f}x\n"
            f"z-score vs null   : {self.z_score:.1f}"
        )

    def as_dict(self) -> dict:
        return {
            "n_real": self.n_real,
            "decoy_mean": round(self.decoy_mean, 2),
            "decoy_std": round(self.decoy_std, 2),
            "fdr": round(self.fdr, 4),
            "fdr_lo": round(self.fdr_lo, 4),
            "fdr_hi": round(self.fdr_hi, 4),
            "enrichment": round(self.enrichment, 2),
            "z_score": round(self.z_score, 2),
            "n_decoy_runs": self.n_decoy_runs,
        }


def make_decoy_transformations(
    real: list[Transformation],
    rng: np.random.Generator,
    offset: float = 0.5137,
) -> list[Transformation]:
    """Generate decoy shifts spanning the same range as the real ones.

    The offset is an irrational-looking constant chosen so decoy shifts do not
    coincide with any real elemental composition — a decoy that happens to
    equal a true biotransformation would contaminate the null.
    """
    shifts = np.array([t.mass_shift for t in real])
    lo, hi = shifts.min(), shifts.max()
    draws = rng.uniform(lo, hi, len(real)) + offset

    out = []
    for real_tr, d in zip(real, draws):
        # Name the decoy after the transformation it replaces so it inherits the
        # same plausibility rule. Without this the decoys pass a filter the real
        # hits must clear, and the FDR estimate is biased upward.
        out.append(
            _FixedShift(
                name=f"decoy of {real_tr.name}",
                shift=float(d),
                category="decoy",
            )
        )
    return out


class _FixedShift(Transformation):
    """A Transformation carrying an arbitrary shift rather than a formula."""

    def __init__(self, name: str, shift: float, category: str = "decoy"):
        super().__init__(
            name=name,
            formula_delta="",
            formula_loss="",
            category=category,
            reversible=False,
            note="decoy",
        )
        object.__setattr__(self, "_shift", shift)

    @property
    def mass_shift(self) -> float:  # type: ignore[override]
        return self._shift  # type: ignore[attr-defined]


def validate_propagation(
    X: np.ndarray,
    mzs: np.ndarray,
    seeds: dict[int, str],
    tol_ppm: float = 5.0,
    coloc_min: float = 0.60,
    n_decoy: int = 50,
    categories: list[str] | None = None,
    seed: int = 0,
) -> ValidationResult:
    """Estimate the FDR of a propagation run using decoy mass shifts.

    Parameters
    ----------
    X, mzs, seeds, tol_ppm, coloc_min, categories
        Passed through to :func:`ionresolver.network.propagate`.
    n_decoy : int
        Number of independent decoy runs. The FDR point estimate varies by
        roughly ten percentage points between random seeds at n=10, so the
        default is deliberately generous.
    seed : int
        Random seed, for reproducibility.

    Returns
    -------
    ValidationResult
    """
    real_trs = get_transformations(categories=categories)

    real_hits = len(
        propagate(
            X, mzs, seeds,
            transformations=real_trs,
            tol_ppm=tol_ppm,
            coloc_min=coloc_min,
        )
    )

    rng = np.random.default_rng(seed)
    counts = []
    for _ in range(n_decoy):
        decoys = make_decoy_transformations(real_trs, rng)
        counts.append(
            len(
                propagate(
                    X, mzs, seeds,
                    transformations=decoys,
                    tol_ppm=tol_ppm,
                    coloc_min=coloc_min,
                )
            )
        )
    counts = np.asarray(counts, dtype=float)

    mean, std = float(counts.mean()), float(counts.std())
    fdr = mean / real_hits if real_hits > 0 else float("nan")

    # Percentile interval over the decoy draws. The point estimate moves a lot
    # between random seeds, so a single number overstates what is known.
    if real_hits > 0 and len(counts):
        lo_c, hi_c = np.percentile(counts, [2.5, 97.5])
        fdr_lo, fdr_hi = lo_c / real_hits, hi_c / real_hits
    else:
        fdr_lo = fdr_hi = float("nan")
    enrich = real_hits / max(mean, 0.5)
    z = (real_hits - mean) / std if std > 0 else float("inf")

    return ValidationResult(
        n_real=real_hits,
        decoy_mean=mean,
        decoy_std=std,
        fdr=fdr,
        fdr_lo=fdr_lo,
        fdr_hi=fdr_hi,
        enrichment=enrich,
        z_score=z,
        n_decoy_runs=n_decoy,
    )


def threshold_sweep(
    X: np.ndarray,
    mzs: np.ndarray,
    seeds: dict[int, str],
    coloc_values=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    tol_ppm: float = 5.0,
    n_decoy: int = 10,
    categories: list[str] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Sweep the colocalisation threshold and report hits and FDR at each.

    Use this to choose an operating point: lower thresholds recover more
    annotations at higher FDR, and the right trade-off depends on whether the
    output feeds hypothesis generation or a final result table.
    """
    rows = []
    for cm in coloc_values:
        v = validate_propagation(
            X, mzs, seeds,
            tol_ppm=tol_ppm,
            coloc_min=cm,
            n_decoy=n_decoy,
            categories=categories,
            seed=seed,
        )
        rows.append(
            {
                "coloc_min": cm,
                "n_real": v.n_real,
                "decoy_mean": round(v.decoy_mean, 1),
                "fdr_percent": round(v.fdr * 100, 1) if v.n_real else float("nan"),
                "fdr_lo": round(v.fdr_lo * 100, 1) if v.n_real else float("nan"),
                "fdr_hi": round(v.fdr_hi * 100, 1) if v.n_real else float("nan"),
                "enrichment": round(v.enrichment, 1),
            }
        )
    return pd.DataFrame(rows)
