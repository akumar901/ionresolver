"""
ionresolver.annotate
--------------------
Annotation of observed m/z values with an explicit evidence trail.

Imaging software typically returns a compound name and nothing else, so every
downstream step treats a bare 5 ppm mass match exactly like a
standard-confirmed identification. This module instead attaches an *evidence
vector* to each candidate and derives a confidence level from it, so results
can be filtered by how much they are actually worth trusting.

Levels follow the Metabolomics Standards Initiative:

===== ===========================================================
Level Meaning
===== ===========================================================
L1    Confirmed against an authentic standard under matched conditions
L2    Putative compound; library match on MS/MS or CCS
L3    Putative class only
L4    Molecular formula or unknown
===== ===========================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .masses import Adduct, adducts_for, exact_mass, ppm_error


@dataclass
class Evidence:
    """Which independent criteria a candidate annotation satisfied."""

    accurate_mass: bool = False
    isotope_pattern: bool = False
    msms_match: bool = False
    ccs_match: bool = False
    standard_match: bool = False
    coloc_support: bool = False

    def count(self) -> int:
        """Number of satisfied criteria."""
        return sum(
            [
                self.accurate_mass,
                self.isotope_pattern,
                self.msms_match,
                self.ccs_match,
                self.standard_match,
                self.coloc_support,
            ]
        )

    def level(self) -> str:
        """MSI confidence level implied by this evidence."""
        if self.standard_match and self.msms_match:
            return "L1"
        if self.msms_match and self.ccs_match:
            return "L2a"
        if self.msms_match or self.ccs_match:
            return "L2"
        if self.coloc_support and self.accurate_mass:
            return "L3"
        return "L4"

    def as_dict(self) -> dict:
        return {
            "accurate_mass": self.accurate_mass,
            "isotope_pattern": self.isotope_pattern,
            "msms_match": self.msms_match,
            "ccs_match": self.ccs_match,
            "standard_match": self.standard_match,
            "coloc_support": self.coloc_support,
        }


@dataclass
class Candidate:
    """One possible identity for an observed ion."""

    mz_observed: float
    name: str
    formula: str
    adduct: str
    mz_theoretical: float
    ppm: float
    evidence: Evidence = field(default_factory=Evidence)
    source: str = "database"

    @property
    def level(self) -> str:
        return self.evidence.level()

    def as_dict(self) -> dict:
        return {
            "mz_observed": round(self.mz_observed, 5),
            "name": self.name,
            "formula": self.formula,
            "adduct": self.adduct,
            "mz_theoretical": round(self.mz_theoretical, 5),
            "ppm": round(self.ppm, 2),
            "level": self.level,
            "n_evidence": self.evidence.count(),
            "source": self.source,
            **self.evidence.as_dict(),
        }


class CompoundDatabase:
    """A searchable set of compounds with precomputed adduct m/z values.

    Parameters
    ----------
    df : DataFrame
        Must contain ``name`` and ``formula`` columns. Any other columns are
        carried through untouched.
    polarity : {"positive", "negative"}
    adducts : list of Adduct, optional
        Overrides the default adduct set for the given polarity.
    include_fragments : bool
        Include in-source fragment species such as ``[M+H-H2O]+``. Useful when
        the goal is to explain every observed peak; leave off when the output
        will seed a biotransformation search, since a fragment is not an
        intact molecule.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        polarity: str = "positive",
        adducts: list[Adduct] | None = None,
        include_fragments: bool = False,
    ):
        required = {"name", "formula"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"database missing columns: {sorted(missing)}")

        self.polarity = polarity
        self.adducts = (
            adducts if adducts is not None
            else adducts_for(polarity, include_fragments=include_fragments)
        )
        self._table = self._expand(df)

    def _expand(self, df: pd.DataFrame) -> pd.DataFrame:
        """One row per (compound, adduct) pair with its theoretical m/z."""
        rows = []
        for _, r in df.iterrows():
            try:
                neutral = exact_mass(r["formula"])
            except ValueError:
                continue  # skip malformed formulae rather than failing the run
            for ad in self.adducts:
                rows.append(
                    {
                        "name": r["name"],
                        "formula": r["formula"],
                        "adduct": ad.name,
                        "mz_theoretical": ad.mz(neutral),
                        "neutral_mass": neutral,
                    }
                )
        out = pd.DataFrame(rows)
        return out.sort_values("mz_theoretical").reset_index(drop=True)

    def __len__(self) -> int:
        return len(self._table)

    @property
    def n_compounds(self) -> int:
        return self._table["name"].nunique()

    def search(self, mz: float, tol_ppm: float = 5.0) -> pd.DataFrame:
        """All (compound, adduct) pairs within ``tol_ppm`` of ``mz``."""
        t = self._table
        lo, hi = mz * (1 - tol_ppm * 1e-6), mz * (1 + tol_ppm * 1e-6)
        hits = t[(t.mz_theoretical >= lo) & (t.mz_theoretical <= hi)].copy()
        if len(hits):
            hits["ppm"] = ppm_error(mz, hits.mz_theoretical)
        return hits


def annotate_mz(
    mz: float,
    database: CompoundDatabase,
    tol_ppm: float = 5.0,
    has_msms: bool = False,
    has_ccs: bool = False,
    has_standard: bool = False,
    coloc_support: bool = False,
) -> list[Candidate]:
    """Annotate one observed m/z, returning every candidate with its evidence.

    The ``has_*`` flags describe what was actually acquired for this ion. They
    are deliberately explicit: if MS/MS was never collected, the ceiling is L4
    and the output says so instead of implying a confidence the data cannot
    support.

    Returns
    -------
    list of Candidate
        Sorted by evidence count then by absolute mass error. An empty
        database match yields a single ``UNKNOWN`` candidate at L4.
    """
    hits = database.search(mz, tol_ppm)

    if hits.empty:
        return [
            Candidate(
                mz_observed=mz,
                name="UNKNOWN",
                formula="?",
                adduct="?",
                mz_theoretical=float("nan"),
                ppm=float("nan"),
                evidence=Evidence(accurate_mass=False),
                source="unmatched",
            )
        ]

    out = []
    for _, h in hits.iterrows():
        ev = Evidence(
            accurate_mass=True,
            isotope_pattern=True,
            msms_match=has_msms,
            ccs_match=has_ccs,
            standard_match=has_standard,
            coloc_support=coloc_support,
        )
        out.append(
            Candidate(
                mz_observed=mz,
                name=h["name"],
                formula=h["formula"],
                adduct=h["adduct"],
                mz_theoretical=float(h.mz_theoretical),
                ppm=float(h.ppm),
                evidence=ev,
            )
        )
    out.sort(key=lambda c: (-c.evidence.count(), abs(c.ppm)))
    return out


def annotate_peaklist(
    mzs,
    database: CompoundDatabase,
    tol_ppm: float = 5.0,
    msms_mzs: set[float] | None = None,
    ccs_mzs: set[float] | None = None,
    standard_mzs: set[float] | None = None,
    best_only: bool = False,
) -> pd.DataFrame:
    """Annotate a whole peak list.

    Parameters
    ----------
    mzs : iterable of float
    database : CompoundDatabase
    tol_ppm : float
    msms_mzs, ccs_mzs, standard_mzs : set of float, optional
        m/z values for which that evidence type is available. Matching is
        exact, so pass the same float values that appear in ``mzs``.
    best_only : bool
        Keep only the top candidate per m/z.

    Returns
    -------
    DataFrame
        One row per candidate, with an ``n_candidates`` column flagging
        ambiguity — the cases where mass alone cannot decide.
    """
    msms_mzs = msms_mzs or set()
    ccs_mzs = ccs_mzs or set()
    standard_mzs = standard_mzs or set()

    records = []
    for mz in mzs:
        cands = annotate_mz(
            mz,
            database,
            tol_ppm=tol_ppm,
            has_msms=mz in msms_mzs,
            has_ccs=mz in ccs_mzs,
            has_standard=mz in standard_mzs,
        )
        n = len([c for c in cands if c.name != "UNKNOWN"])
        for c in cands:
            rec = c.as_dict()
            rec["n_candidates"] = n
            records.append(rec)

    df = pd.DataFrame(records)
    if best_only and len(df):
        df = df.sort_values(["mz_observed", "n_evidence"], ascending=[True, False])
        df = df.groupby("mz_observed", as_index=False).first()
    return df


def confidence_summary(annotations: pd.DataFrame) -> pd.DataFrame:
    """Count features per confidence level, with percentages."""
    if annotations.empty:
        return pd.DataFrame(columns=["level", "n", "percent"])
    best = (
        annotations.sort_values("n_evidence", ascending=False)
        .groupby("mz_observed", as_index=False)
        .first()
    )
    counts = best["level"].value_counts().reset_index()
    counts.columns = ["level", "n"]
    counts["percent"] = (counts.n / counts.n.sum() * 100).round(1)
    order = {"L1": 0, "L2a": 1, "L2": 2, "L3": 3, "L4": 4}
    counts["_o"] = counts.level.map(lambda x: order.get(x, 9))
    return counts.sort_values("_o").drop(columns="_o").reset_index(drop=True)
