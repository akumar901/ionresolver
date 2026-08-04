"""
ionresolver.network
-------------------
Biotransformation network propagation — the core method.

Most unannotated features in an imaging experiment are not novel chemistry.
They are known metabolites that have been modified in vivo, and every
modification carries a fixed exact mass. Two independent conditions therefore
constrain a proposed identity:

1. the unknown sits at a confidently annotated parent's m/z plus a valid
   biotransformation mass shift, within tolerance;
2. the unknown and its proposed parent colocalise in tissue.

Mass alone throws up coincidences at any realistic tolerance. Requiring
colocalisation as well suppresses them, because an arbitrary pair of ions has
no reason to share a spatial distribution. Neither criterion is sufficient on
its own; together they support a Level 3 putative assignment that MS/MS can
then confirm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .coloc import cosine_coloc
from .groups import infer_groups, is_artefact_shift, is_plausible
from .transformations import Transformation, get_transformations


@dataclass
class PropagatedAnnotation:
    """A putative identity derived from an annotated parent."""

    mz_unknown: float
    feature_index: int
    parent_name: str
    parent_mz: float
    parent_index: int
    transformation: str
    category: str
    mass_shift: float
    ppm: float
    coloc: float
    proposed_name: str
    level: str = "L3"
    parent_adduct: str = "?"
    plausible: bool = True
    plausibility_note: str = ""
    alternatives: str = ""
    artefact_risk: str = ""

    def as_dict(self) -> dict:
        return {
            "mz_unknown": round(self.mz_unknown, 5),
            "feature_index": self.feature_index,
            "proposed_name": self.proposed_name,
            "parent_name": self.parent_name,
            "parent_adduct": self.parent_adduct,
            "parent_mz": round(self.parent_mz, 5),
            "transformation": self.transformation,
            "category": self.category,
            "mass_shift": round(self.mass_shift, 5),
            "ppm": round(self.ppm, 2),
            "coloc": round(self.coloc, 3),
            "level": self.level,
            "plausible": self.plausible,
            "plausibility_note": self.plausibility_note,
            "alternatives": self.alternatives,
            "artefact_risk": self.artefact_risk,
        }


def propagate(
    X: np.ndarray,
    mzs: np.ndarray,
    seeds: dict[int, str],
    transformations: list[Transformation] | None = None,
    seed_adducts: dict[int, str] | None = None,
    tol_ppm: float = 5.0,
    coloc_min: float = 0.60,
    categories: list[str] | None = None,
    sample_context: str | None = None,
    max_per_unknown: int = 1,
    check_plausibility: bool = True,
    strict_plausibility: bool = False,
    exclude_artefacts: bool = True,
    compound_groups: dict[str, set[str]] | None = None,
    dedupe_by_mass: bool = True,
) -> pd.DataFrame:
    """Propagate identities from annotated seeds to unknown features.

    Parameters
    ----------
    X : ndarray, shape (n_pixels, n_features)
        Intensity matrix. TIC-normalise before calling.
    mzs : ndarray, shape (n_features,)
        m/z for each column of ``X``.
    seeds : dict
        Maps feature index to compound name for confidently annotated ions.
    transformations : list of Transformation, optional
        Defaults to the full catalogue including reverses.
    seed_adducts : dict, optional
        Maps feature index to the adduct that seeded it. Without this a row
        reading "PC(32:0) + hydration" cannot be checked by hand, because the
        arithmetic only closes for one of the compound's several adducts.
    tol_ppm : float
        Mass tolerance for matching parent + shift.
    coloc_min : float
        Minimum colocalisation with the parent. This is the parameter that
        controls the false-discovery rate; see :mod:`ionresolver.validate`.
    categories : list of str, optional
        Restrict to transformation categories, e.g. ``["phase_ii"]``.
    sample_context : str, optional
        Biological context of the sample, used to exclude chemistry that cannot
        occur in it. ``"cell_culture"`` drops microbial transformations, since
        an axenic culture has no gut flora. ``"tissue"`` keeps everything.
        Accepted values: ``"cell_culture"``, ``"tissue"``, ``"plasma"``.
    max_per_unknown : int
        Keep at most this many proposals per unknown feature, best first.
    check_plausibility : bool
        Reject assignments whose chemistry is impossible for the parent — a
        phosphatidylcholine cannot be deaminated, for instance. Adds a third
        constraint independent of both mass and colocalisation.
    strict_plausibility : bool
        Reject transformations that have no plausibility rule defined, rather
        than accepting them.
    exclude_artefacts : bool
        Drop mass shifts that coincide with common in-source losses or adduct
        differences. A feature at parent - 18.0106 is usually water loss from
        the parent ion, and it colocalises perfectly because it is the same
        molecule — the strongest-looking hits are often the least real.
    compound_groups : dict, optional
        Explicit functional groups per compound name. Overrides inference.
    dedupe_by_mass : bool
        Several transformations share a mass shift (dehydrogenation,
        desaturation and reverse reduction are all -2.0157). When True these
        collapse to one row, with the alternatives listed in a column, so a
        single mass relationship is not counted repeatedly.

    Returns
    -------
    DataFrame
        One row per proposed annotation, sorted by colocalisation.
    """
    if X.shape[1] != len(mzs):
        raise ValueError(
            f"X has {X.shape[1]} features but {len(mzs)} m/z values were given"
        )

    if transformations is None:
        transformations = get_transformations(categories=categories)

    if sample_context:
        excluded = _CONTEXT_EXCLUSIONS.get(sample_context.strip().lower())
        if excluded is None:
            raise ValueError(
                f"unknown sample_context {sample_context!r}; expected one of "
                f"{sorted(_CONTEXT_EXCLUSIONS)}"
            )
        transformations = [t for t in transformations if t.category not in excluded]

    results: list[PropagatedAnnotation] = []
    compound_groups = compound_groups or {}
    seed_adducts = seed_adducts or {}

    for parent_idx, parent_name in seeds.items():
        pgroups = compound_groups.get(parent_name) or infer_groups(parent_name)
        parent_mz = float(mzs[parent_idx])
        parent_img = X[:, parent_idx]

        for tr in transformations:
            target = parent_mz + tr.mass_shift
            if target <= 0:
                continue

            lo = target * (1 - tol_ppm * 1e-6)
            hi = target * (1 + tol_ppm * 1e-6)
            for j in np.where((mzs >= lo) & (mzs <= hi))[0]:
                j = int(j)
                if j in seeds:
                    continue  # already annotated; not an unknown

                c = cosine_coloc(parent_img, X[:, j])
                if c < coloc_min:
                    continue

                art, art_label = is_artefact_shift(tr.mass_shift)
                if art and exclude_artefacts:
                    continue

                ok, why = (True, "not checked")
                if check_plausibility:
                    ok, why = is_plausible(tr.name, pgroups, strict=strict_plausibility)
                    if not ok:
                        continue

                results.append(
                    PropagatedAnnotation(
                        mz_unknown=float(mzs[j]),
                        feature_index=j,
                        parent_name=parent_name,
                        parent_mz=parent_mz,
                        parent_index=parent_idx,
                        parent_adduct=seed_adducts.get(parent_idx, "?"),
                        transformation=tr.name,
                        category=tr.category,
                        mass_shift=tr.mass_shift,
                        ppm=(float(mzs[j]) - target) / target * 1e6,
                        coloc=c,
                        proposed_name=f"{parent_name} + {tr.name}",
                        plausible=ok,
                        plausibility_note=why,
                        artefact_risk=art_label,
                    )
                )

    if not results:
        return pd.DataFrame(
            columns=[
                "mz_unknown", "feature_index", "proposed_name", "parent_name",
                "parent_adduct", "parent_mz", "transformation", "category",
                "mass_shift",
                "ppm", "coloc", "level", "plausible", "plausibility_note",
                "alternatives", "artefact_risk",
            ]
        )

    df = pd.DataFrame([r.as_dict() for r in results])
    df = df.sort_values("coloc", ascending=False)

    if dedupe_by_mass:
        df = _collapse_equal_mass_shifts(df)

    if max_per_unknown:
        df = df.groupby("mz_unknown", as_index=False, group_keys=False).head(
            max_per_unknown
        )
    return df.sort_values("coloc", ascending=False).reset_index(drop=True)


def seeds_from_annotations(
    annotations: pd.DataFrame,
    mzs: np.ndarray,
    min_level: str = "L2",
    tol_ppm: float = 5.0,
) -> dict[int, str]:
    """Build a seed dict from an annotation table, keeping confident rows only.

    Parameters
    ----------
    annotations : DataFrame
        Output of :func:`ionresolver.annotate.annotate_peaklist`.
    mzs : ndarray
        Feature m/z axis, used to map annotations back to column indices.
    min_level : {"L1", "L2a", "L2", "L3", "L4"}
        Lowest confidence level accepted as a seed. Seeding from weak
        annotations propagates their errors, so the default is deliberately
        strict.
    tol_ppm : float
        Tolerance for matching an annotation's m/z to a feature index.
    """
    rank = {"L1": 0, "L2a": 1, "L2": 2, "L3": 3, "L4": 4}
    cutoff = rank.get(min_level, 2)

    keep = annotations[annotations.level.map(lambda x: rank.get(x, 9)) <= cutoff]
    keep = keep[keep.name != "UNKNOWN"]

    seeds: dict[int, str] = {}
    for _, r in keep.iterrows():
        d = np.abs(mzs - r.mz_observed) / r.mz_observed * 1e6
        j = int(d.argmin())
        if d[j] <= tol_ppm:
            seeds[j] = r["name"]
    return seeds


def seed_adducts_from_annotations(
    annotations: pd.DataFrame,
    mzs: np.ndarray,
    min_level: str = "L2",
    tol_ppm: float = 5.0,
) -> dict[int, str]:
    """Map feature index to the adduct that seeded it.

    Pass the result to :func:`propagate` so each row records which ion of the
    parent the mass arithmetic was done against. Without it, "PC(32:0) +
    hydration" cannot be verified by hand — the sum only closes for one of the
    compound's adducts.
    """
    rank = {"L1": 0, "L2a": 1, "L2": 2, "L3": 3, "L4": 4}
    cutoff = rank.get(min_level, 2)

    keep = annotations[annotations.level.map(lambda x: rank.get(x, 9)) <= cutoff]
    keep = keep[keep.name != "UNKNOWN"]

    out: dict[int, str] = {}
    for _, r in keep.iterrows():
        d = np.abs(mzs - r.mz_observed) / r.mz_observed * 1e6
        j = int(d.argmin())
        if d[j] <= tol_ppm:
            out[j] = r.get("adduct", "?")
    return out


def network_edges(propagated: pd.DataFrame) -> pd.DataFrame:
    """Reshape propagation results into an edge list for graph export.

    Suitable for writing to Cytoscape or loading with networkx.
    """
    if propagated.empty:
        return pd.DataFrame(columns=["source", "target", "transformation", "weight"])
    return pd.DataFrame(
        {
            "source": propagated.parent_name,
            "target": propagated.proposed_name,
            "transformation": propagated.transformation,
            "weight": propagated.coloc,
        }
    )


def _collapse_equal_mass_shifts(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse rows that describe the same mass relationship.

    Several transformations are mass-degenerate: dehydrogenation, desaturation
    and reverse reduction all shift by -2.0157. Reporting them as separate hits
    counts one physical relationship up to three times. This keeps the best
    row per (unknown, parent, mass shift) and records the discarded names in
    an ``alternatives`` column, so the ambiguity stays visible without
    inflating the count.
    """
    if df.empty:
        return df

    df = df.copy()
    df["_shift_key"] = df.mass_shift.round(4)
    keys = ["mz_unknown", "parent_mz", "_shift_key"]

    keep = []
    for _, grp in df.groupby(keys, sort=False):
        grp = grp.sort_values("coloc", ascending=False)
        best = grp.iloc[0].copy()
        others = [n for n in grp.transformation.tolist()[1:]]
        best["alternatives"] = "; ".join(dict.fromkeys(others))
        keep.append(best)

    out = pd.DataFrame(keep).drop(columns="_shift_key")
    return out.sort_values("coloc", ascending=False).reset_index(drop=True)


def split_by_category(propagated: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split results into lipid-series and true biotransformation relationships.

    Within a lipid class, chain extension and desaturation relate members of a
    homologous series that share membranes and therefore colocalise almost by
    construction. Those relationships are real chemistry but weak evidence, and
    reporting them alongside conjugation reactions buries the interesting
    results. This separates the two.

    Returns
    -------
    dict with keys ``"lipid_series"``, ``"biotransformation"``, and ``"all"``.
    """
    if propagated.empty:
        return {"lipid_series": propagated, "biotransformation": propagated,
                "all": propagated}

    lipid_parent = propagated.parent_name.str.match(
        r"^(LPC|LPE|PC|PE|PS|PI|PG|PA|SM|TG|DG|MG|CE|Cer)\b", case=False, na=False
    )
    # Mass shifts that simply step along a homologous lipid series.
    series_shifts = {"chain extension (C2H4)", "desaturation", "dehydrogenation",
                     "reduction", "reverse reduction", "reverse dehydrogenation",
                     "acetyl-CoA unit"}
    is_series = propagated.transformation.isin(series_shifts)

    is_lipid_series = (propagated.category == "lipid") | (lipid_parent & is_series)
    return {
        "lipid_series": propagated[is_lipid_series].reset_index(drop=True),
        "biotransformation": propagated[~is_lipid_series].reset_index(drop=True),
        "all": propagated,
    }


# Transformation categories that cannot occur in a given sample type.
# A human cell culture has no gut flora, so microbial chemistry is impossible
# there regardless of how well the mass and colocalisation agree.
_CONTEXT_EXCLUSIONS: dict[str, set[str]] = {
    "cell_culture": {"microbial"},
    "tissue": set(),
    "plasma": set(),
}


def merge_split_features(
    mzs: np.ndarray,
    X: np.ndarray,
    tol_ppm: float = 6.0,
) -> tuple[np.ndarray, np.ndarray, dict[int, list[int]]]:
    """Merge features that are within one bin width of each other.

    Binning a processed imzML at N ppm frequently splits a single peak across
    two adjacent bins. Both survive the occupancy filter, both get propagated,
    and one relationship is reported twice under different names. Merging on a
    tolerance slightly wider than the bin width collapses them.

    Parameters
    ----------
    mzs : ndarray, shape (n_features,)
    X : ndarray, shape (n_pixels, n_features)
    tol_ppm : float
        Merge features closer than this. Set a little above the binning width.

    Returns
    -------
    (merged_mzs, merged_X, mapping)
        ``mapping`` records which original indices went into each merged one.
    """
    order = np.argsort(mzs)
    mzs_sorted = mzs[order]

    groups: list[list[int]] = []
    current = [0]
    for i in range(1, len(mzs_sorted)):
        sep = (mzs_sorted[i] - mzs_sorted[i - 1]) / mzs_sorted[i - 1] * 1e6
        if sep <= tol_ppm:
            current.append(i)
        else:
            groups.append(current)
            current = [i]
    groups.append(current)

    new_mz = np.empty(len(groups))
    new_X = np.empty((X.shape[0], len(groups)), dtype=X.dtype)
    mapping: dict[int, list[int]] = {}

    for g_i, g in enumerate(groups):
        orig = [int(order[k]) for k in g]
        block = X[:, orig]
        weights = block.sum(axis=0)
        total = weights.sum()
        # intensity-weighted centroid keeps the merged m/z on the true peak
        new_mz[g_i] = (
            float(np.average(mzs[orig], weights=weights)) if total > 0
            else float(np.mean(mzs[orig]))
        )
        new_X[:, g_i] = block.sum(axis=1)
        mapping[g_i] = orig

    return new_mz, new_X, mapping
