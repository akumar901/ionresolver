"""
ionresolver.groups
------------------
Functional-group reasoning for chemical plausibility.

Mass and colocalisation together still admit assignments that are chemically
impossible. A phosphatidylcholine cannot be deaminated — its nitrogen is a
quaternary ammonium with no hydrogens to lose. A compound with no carboxyl
group cannot undergo glycine conjugation. Filtering on functional groups adds
a third, independent constraint that neither mass nor spatial distribution
provides.

Groups are assigned in two ways: explicitly, when the compound database
supplies a ``groups`` column, or inferred from compound class and formula.
Inference is deliberately conservative — an uncertain group is treated as
present, so plausible assignments are not discarded on a guess.
"""

from __future__ import annotations

import re

from .masses import parse_formula

# Functional groups the transformation rules refer to.
GROUPS = {
    "hydroxyl",             # -OH
    "carboxyl",             # -COOH
    "primary_amine",        # -NH2
    "secondary_amine",      # -NH-
    "quaternary_ammonium",  # -N+(CH3)3, cannot be deaminated or N-conjugated
    "phosphate",            # -PO4
    "thiol",                # -SH
    "aromatic",             # aromatic ring
    "carbonyl",             # C=O
    "double_bond",          # C=C, available for saturation/desaturation
    "ester",                # -COO-
    "amide",                # -CONH-
}

# Lipid classes recognised by name, with the groups they carry.
# Phosphatidylcholines and sphingomyelins are the important cases: their
# nitrogen is quaternary, which rules out a whole family of reactions.
_LIPID_CLASSES = {
    "PC":  {"quaternary_ammonium", "phosphate", "ester", "double_bond"},
    "LPC": {"quaternary_ammonium", "phosphate", "ester", "hydroxyl", "double_bond"},
    "SM":  {"quaternary_ammonium", "phosphate", "amide", "hydroxyl", "double_bond"},
    "PE":  {"primary_amine", "phosphate", "ester", "double_bond"},
    "LPE": {"primary_amine", "phosphate", "ester", "hydroxyl", "double_bond"},
    "PS":  {"primary_amine", "carboxyl", "phosphate", "ester", "double_bond"},
    "PI":  {"phosphate", "hydroxyl", "ester", "double_bond"},
    "PG":  {"phosphate", "hydroxyl", "ester", "double_bond"},
    "PA":  {"phosphate", "ester", "double_bond"},
    "TG":  {"ester", "double_bond"},
    "DG":  {"ester", "hydroxyl", "double_bond"},
    "MG":  {"ester", "hydroxyl", "double_bond"},
    "CE":  {"ester", "double_bond"},
    "Cer": {"amide", "hydroxyl", "double_bond"},
}

# Compounds whose groups are worth stating outright rather than inferring.
_KNOWN_COMPOUNDS = {
    "choline":            {"quaternary_ammonium", "hydroxyl"},
    "betaine":            {"quaternary_ammonium", "carboxyl"},
    "carnitine":          {"quaternary_ammonium", "hydroxyl", "carboxyl"},
    "acetylcarnitine":    {"quaternary_ammonium", "ester", "carboxyl"},
    "palmitoylcarnitine": {"quaternary_ammonium", "ester", "carboxyl"},
    "creatine":           {"primary_amine", "carboxyl", "amide"},
    "glucose":            {"hydroxyl", "carbonyl"},
    "adenine":            {"primary_amine", "aromatic"},
    "adenosine":          {"primary_amine", "hydroxyl", "aromatic"},
    "glutathione":        {"thiol", "carboxyl", "primary_amine", "amide"},
    "spermidine":         {"primary_amine", "secondary_amine"},
    "phenylalanine":      {"primary_amine", "carboxyl", "aromatic"},
    "tryptophan":         {"primary_amine", "carboxyl", "aromatic"},
    "tyrosine":           {"primary_amine", "carboxyl", "aromatic", "hydroxyl"},
    "benzoate":           {"carboxyl", "aromatic"},
    "hippurate":          {"carboxyl", "aromatic", "amide"},
    "p-cresol":           {"hydroxyl", "aromatic"},
    "phenylacetate":      {"carboxyl", "aromatic"},
}

_LIPID_RE = re.compile(r"^(LPC|LPE|PC|PE|PS|PI|PG|PA|SM|TG|DG|MG|CE|Cer)\b", re.I)


def infer_groups(name: str, formula: str | None = None) -> set[str]:
    """Infer functional groups from a compound name and, optionally, formula.

    Resolution order: exact known compound, then lipid class prefix, then
    formula-based heuristics. When nothing matches, a permissive set is
    returned so that unknown compounds are not silently excluded.
    """
    key = name.strip().lower()
    if key in _KNOWN_COMPOUNDS:
        return set(_KNOWN_COMPOUNDS[key])

    m = _LIPID_RE.match(name.strip())
    if m:
        cls = m.group(1)
        for known, groups in _LIPID_CLASSES.items():
            if known.lower() == cls.lower():
                out = set(groups)
                # "PC(O-34:1)" and similar denote ether lipids, not esters
                if "O-" in name or "P-" in name:
                    out.discard("ester")
                return out

    if formula:
        return _groups_from_formula(formula)

    return {"hydroxyl", "carboxyl", "double_bond"}  # permissive fallback


def _groups_from_formula(formula: str) -> set[str]:
    """Coarse group inference from elemental composition alone.

    This can only rule things in, not out, with any confidence — a formula
    says nothing about connectivity. It is used as a last resort.
    """
    try:
        counts = parse_formula(formula)
    except ValueError:
        return {"hydroxyl", "carboxyl", "double_bond"}

    groups: set[str] = set()
    c = counts.get("C", 0)
    h = counts.get("H", 0)
    n = counts.get("N", 0)
    o = counts.get("O", 0)
    p = counts.get("P", 0)
    s = counts.get("S", 0)

    if o >= 1:
        groups |= {"hydroxyl", "carbonyl"}
    if o >= 2:
        groups.add("carboxyl")
    if n >= 1:
        groups |= {"primary_amine", "secondary_amine"}
    if p >= 1:
        groups.add("phosphate")
    if s >= 1:
        groups.add("thiol")

    # degree of unsaturation; >= 4 usually implies a ring system
    if c and h:
        dou = c - h / 2 + n / 2 + 1
        if dou >= 1:
            groups.add("double_bond")
        if dou >= 4:
            groups.add("aromatic")
    return groups


# --- what each transformation demands of its substrate ---------------------
# "requires_any": the parent must carry at least one of these groups.
# "forbidden":    the parent must carry none of these.
TRANSFORMATION_RULES: dict[str, dict[str, set[str]]] = {
    # Phase II conjugation
    "sulfation": {
        "requires_any": {"hydroxyl", "primary_amine", "secondary_amine", "aromatic"},
        "forbidden": set(),
    },
    "glucuronidation": {
        "requires_any": {"hydroxyl", "carboxyl", "primary_amine", "secondary_amine", "thiol"},
        "forbidden": set(),
    },
    "glycine conjugation": {
        "requires_any": {"carboxyl"},          # acyl-CoA intermediate needs a carboxyl
        "forbidden": set(),
    },
    "glutamine conjugation": {
        "requires_any": {"carboxyl"},
        "forbidden": set(),
    },
    "taurine conjugation": {
        "requires_any": {"carboxyl"},
        "forbidden": set(),
    },
    "glutathione conjugation": {
        "requires_any": {"aromatic", "double_bond", "carbonyl"},   # electrophilic centre
        "forbidden": set(),
    },
    "acetylation": {
        "requires_any": {"hydroxyl", "primary_amine", "secondary_amine", "thiol"},
        "forbidden": {"quaternary_ammonium"},   # unless another site exists
    },
    "methylation": {
        "requires_any": {"hydroxyl", "primary_amine", "secondary_amine", "thiol", "carboxyl"},
        "forbidden": set(),
    },
    # Phase I
    "hydroxylation": {
        "requires_any": {"aromatic", "double_bond"},
        "forbidden": set(),
    },
    "epoxidation": {
        "requires_any": {"double_bond", "aromatic"},
        "forbidden": set(),
    },
    "oxidation to ketone": {
        "requires_any": {"hydroxyl"},
        "forbidden": set(),
    },
    "reduction": {
        "requires_any": {"carbonyl", "double_bond", "ester"},
        "forbidden": set(),
    },
    "dehydrogenation": {
        "requires_any": {"hydroxyl", "double_bond", "ester"},
        "forbidden": set(),
    },
    "decarboxylation": {
        "requires_any": {"carboxyl"},
        "forbidden": set(),
    },
    "deamination": {
        "requires_any": {"primary_amine", "secondary_amine"},
        "forbidden": {"quaternary_ammonium"},   # no N-H to remove
    },
    "hydration": {
        "requires_any": {"double_bond", "ester", "amide"},
        "forbidden": set(),
    },
    # Lipid series
    "chain extension (C2H4)": {
        "requires_any": {"ester", "amide", "carboxyl"},
        "forbidden": set(),
    },
    "desaturation": {
        "requires_any": {"ester", "amide", "double_bond"},
        "forbidden": set(),
    },
    "phosphorylation": {
        "requires_any": {"hydroxyl"},
        "forbidden": set(),
    },
    "hexose addition": {
        "requires_any": {"hydroxyl", "amide"},
        "forbidden": set(),
    },
    "acetyl-CoA unit": {
        "requires_any": {"ester", "carboxyl"},
        "forbidden": set(),
    },
    # Microbial
    "dehydroxylation": {
        "requires_any": {"hydroxyl"},
        "forbidden": set(),
    },
    "sulfate removal": {
        "requires_any": {"phosphate", "hydroxyl"},   # needs an existing sulfate
        "forbidden": set(),
    },
    "glucuronide cleavage": {
        "requires_any": {"hydroxyl"},                # needs an existing glucuronide
        "forbidden": set(),
    },
    "indole formation": {
        "requires_any": {"aromatic"},
        "forbidden": set(),
    },
}


def is_plausible(
    transformation_name: str,
    parent_groups: set[str],
    strict: bool = False,
) -> tuple[bool, str]:
    """Check whether a transformation can act on a parent with these groups.

    Parameters
    ----------
    transformation_name : str
        Name as it appears in the catalogue. Reverse transformations
        ("reverse methylation") are checked against their forward rule.
    parent_groups : set of str
    strict : bool
        When True, a transformation with no rule defined is rejected. The
        default accepts it, so an incomplete rule set does not silently
        discard valid chemistry.

    Returns
    -------
    (bool, str)
        Whether the assignment is plausible, and a short reason.
    """
    base = transformation_name
    for prefix in ("decoy of ", "reverse "):
        base = base.replace(prefix, "")
    base = base.strip()
    rule = TRANSFORMATION_RULES.get(base)

    if rule is None:
        return (not strict, "no rule defined" if strict else "no rule; accepted")

    forbidden = rule["forbidden"] & parent_groups
    if forbidden:
        return False, f"parent has {sorted(forbidden)[0]}, incompatible"

    required = rule["requires_any"]
    if required and not (required & parent_groups):
        return False, f"parent lacks any of {sorted(required)}"

    return True, "plausible"


# --- in-source artefacts and adduct differences ----------------------------
# Mass relationships that look like biotransformations but usually are not.
# A feature at parent - 18.0106 on a MALDI instrument is far more likely to be
# in-source water loss from the parent ion than a distinct dehydrated molecule,
# and it will colocalise perfectly because it *is* the same molecule. Adduct
# differences behave the same way.
ARTEFACT_SHIFTS: dict[float, str] = {
    -18.010565: "in-source water loss [M+H-H2O]+",
    -17.026549: "in-source ammonia loss",
    +21.981944: "Na/H adduct difference ([M+Na]+ vs [M+H]+)",
    +37.955882: "K/H adduct difference ([M+K]+ vs [M+H]+)",
    +15.973898: "K/Na adduct difference",
    -1.007825:  "hydrogen loss / radical species",
    +1.007825:  "hydrogen gain / isotope tail",
    -27.994915: "in-source CO loss",
    -43.989829: "in-source CO2 loss",
    -46.005479: "in-source formic acid loss",
    +17.026549: "ammonia adduct difference",
}


def is_artefact_shift(mass_shift: float, tol: float = 0.005) -> tuple[bool, str]:
    """Flag mass shifts that are probably instrumental rather than biological.

    Parameters
    ----------
    mass_shift : float
    tol : float
        Absolute tolerance in Da.

    Returns
    -------
    (bool, str)
        Whether the shift matches a known artefact, and its description.
    """
    for shift, label in ARTEFACT_SHIFTS.items():
        if abs(mass_shift - shift) <= tol:
            return True, label
    return False, ""
