"""
ionresolver — annotation of imaging mass spectrometry data with explicit
confidence.

Companion to `chemresolver <https://github.com/akumar901/chemresolver->`_,
which resolves compound *names* to structures. ionresolver resolves observed
*ions* to identities, and reports how much each identity is worth trusting.
"""

__version__ = "0.4.0"

from .masses import (
    Adduct, exact_mass, parse_formula, ppm_error, within_ppm,
    POSITIVE_ADDUCTS, NEGATIVE_ADDUCTS, POSITIVE_FRAGMENTS,
    NEGATIVE_FRAGMENTS, adducts_for,
)
from .transformations import (
    Transformation, ALL_TRANSFORMATIONS, get_transformations,
    transformation_table, PHASE_I, PHASE_II, LIPID, MICROBIAL,
)
from .coloc import (
    cosine_coloc, pearson_coloc, coloc_matrix, tic_normalize, on_tissue_mask,
)
from .annotate import (
    Evidence, Candidate, CompoundDatabase,
    annotate_mz, annotate_peaklist, confidence_summary,
)
from .network import (propagate, seeds_from_annotations, seed_adducts_from_annotations,
                      network_edges, split_by_category, merge_split_features)
from .groups import (infer_groups, is_plausible, is_artefact_shift,
                     TRANSFORMATION_RULES, GROUPS, ARTEFACT_SHIFTS)
from .validate import validate_propagation, threshold_sweep, ValidationResult

__all__ = [
    "Adduct", "exact_mass", "parse_formula", "ppm_error", "within_ppm",
    "POSITIVE_ADDUCTS", "NEGATIVE_ADDUCTS", "POSITIVE_FRAGMENTS",
    "NEGATIVE_FRAGMENTS", "adducts_for",
    "Transformation", "ALL_TRANSFORMATIONS", "get_transformations",
    "transformation_table", "PHASE_I", "PHASE_II", "LIPID", "MICROBIAL",
    "cosine_coloc", "pearson_coloc", "coloc_matrix", "tic_normalize",
    "on_tissue_mask", "Evidence", "Candidate", "CompoundDatabase",
    "annotate_mz", "annotate_peaklist", "confidence_summary",
    "propagate", "seeds_from_annotations", "seed_adducts_from_annotations",
    "network_edges", "split_by_category", "merge_split_features",
    "infer_groups", "is_plausible", "is_artefact_shift",
    "TRANSFORMATION_RULES", "GROUPS", "ARTEFACT_SHIFTS",
    "validate_propagation", "threshold_sweep", "ValidationResult",
]
