"""
chemresolver.msi.transformations
--------------------------------
A catalogue of biotransformations — the exact mass shifts the body applies to
metabolites.

Most "unknown" features in an imaging experiment are not novel chemistry. They
are known metabolites that have been conjugated, oxidised, methylated, or
otherwise modified. Each modification adds a fixed, exact mass. Searching for
those offsets from a confidently annotated parent is a way to propose an
identity for a feature that accurate mass alone leaves anonymous.

The Phase II conjugations here are the same chemistry that produces hippurate
(benzoate + glycine), p-cresol sulfate (p-cresol + sulfate), and
phenylacetylglutamine (phenylacetate + glutamine).
"""

from __future__ import annotations

from dataclasses import dataclass

from .masses import exact_mass


@dataclass(frozen=True)
class Transformation:
    """A single biotransformation.

    Parameters
    ----------
    name : str
        Human-readable label, e.g. ``"sulfation"``.
    formula_delta : str
        Atoms gained, as a formula. Empty when the change is a loss only.
    formula_loss : str
        Atoms lost, as a formula. Empty when the change is a gain only.
    category : str
        ``"phase_i"``, ``"phase_ii"``, ``"lipid"``, or ``"microbial"``.
    reversible : bool
        Whether the reaction commonly runs in both directions in vivo.
    note : str
        Short comment on biological context.
    """

    name: str
    formula_delta: str
    formula_loss: str
    category: str
    reversible: bool = False
    note: str = ""

    @property
    def mass_shift(self) -> float:
        """Net exact mass change, gains minus losses."""
        gain = exact_mass(self.formula_delta) if self.formula_delta else 0.0
        loss = exact_mass(self.formula_loss) if self.formula_loss else 0.0
        return gain - loss


# --- Phase II conjugation: the body tagging a compound for clearance -------
PHASE_II: list[Transformation] = [
    Transformation("sulfation", "SO3", "", "phase_ii",
                   note="SULT enzymes; gives p-cresol sulfate"),
    Transformation("glucuronidation", "C6H8O6", "", "phase_ii",
                   note="UGT enzymes; dominant clearance route"),
    Transformation("glycine conjugation", "C2H3NO", "", "phase_ii",
                   note="GLYAT; benzoate to hippurate"),
    Transformation("glutamine conjugation", "C5H8N2O2", "", "phase_ii",
                   note="phenylacetate to phenylacetylglutamine"),
    Transformation("taurine conjugation", "C2H5NO2S", "", "phase_ii",
                   note="bile acid conjugation"),
    Transformation("glutathione conjugation", "C10H17N3O6S", "", "phase_ii",
                   note="GST; electrophile detoxification"),
    Transformation("acetylation", "C2H2O", "", "phase_ii",
                   note="NAT enzymes"),
    Transformation("methylation", "CH2", "", "phase_ii", reversible=True,
                   note="COMT, methyltransferases"),
]

# --- Phase I: oxidation, reduction, hydrolysis ----------------------------
PHASE_I: list[Transformation] = [
    Transformation("hydroxylation", "O", "", "phase_i",
                   note="cytochrome P450, e.g. CYP2E1"),
    Transformation("oxidation to ketone", "O", "H2", "phase_i"),
    Transformation("reduction", "H2", "", "phase_i", reversible=True),
    Transformation("dehydrogenation", "", "H2", "phase_i", reversible=True),
    Transformation("decarboxylation", "", "CO2", "phase_i"),
    Transformation("deamination", "O", "NH3", "phase_i"),
    Transformation("hydration", "H2O", "", "phase_i", reversible=True),
    Transformation("epoxidation", "O", "", "phase_i"),
]

# --- Lipid series: how lipids vary within a class --------------------------
LIPID: list[Transformation] = [
    Transformation("chain extension (C2H4)", "C2H4", "", "lipid",
                   note="two-carbon elongation"),
    Transformation("desaturation", "", "H2", "lipid",
                   note="one extra double bond"),
    Transformation("phosphorylation", "HPO3", "", "lipid", reversible=True),
    Transformation("hexose addition", "C6H10O5", "", "lipid",
                   note="glycosylation"),
    Transformation("acetyl-CoA unit", "C2H2O", "", "lipid"),
]

# --- Microbial: gut flora chemistry ---------------------------------------
MICROBIAL: list[Transformation] = [
    Transformation("dehydroxylation", "", "O", "microbial",
                   note="gut bacterial bile acid metabolism"),
    Transformation("sulfate removal", "", "SO3", "microbial",
                   note="bacterial sulfatases"),
    Transformation("glucuronide cleavage", "", "C6H8O6", "microbial",
                   note="bacterial beta-glucuronidase; enterohepatic recycling"),
    Transformation("indole formation", "", "C2H4O2", "microbial",
                   note="tryptophan catabolism by gut flora"),
]

ALL_TRANSFORMATIONS: list[Transformation] = PHASE_I + PHASE_II + LIPID + MICROBIAL


def get_transformations(
    categories: list[str] | None = None,
    include_reverse: bool = True,
) -> list[Transformation]:
    """Select transformations, optionally adding the reverse of reversible ones.

    Parameters
    ----------
    categories : list of str, optional
        Restrict to these categories. ``None`` returns everything.
    include_reverse : bool
        When True, reversible transformations also appear with a negative mass
        shift (e.g. demethylation alongside methylation).
    """
    picked = ALL_TRANSFORMATIONS
    if categories is not None:
        wanted = {c.lower() for c in categories}
        picked = [t for t in picked if t.category in wanted]

    if not include_reverse:
        return list(picked)

    out = list(picked)
    for t in picked:
        if t.reversible:
            out.append(
                Transformation(
                    name=f"reverse {t.name}",
                    formula_delta=t.formula_loss,
                    formula_loss=t.formula_delta,
                    category=t.category,
                    reversible=False,
                    note=f"reverse of: {t.note}" if t.note else "",
                )
            )
    return out


def transformation_table() -> list[dict]:
    """Flatten the catalogue into dicts, sorted by mass shift."""
    rows = [
        {
            "name": t.name,
            "category": t.category,
            "mass_shift": round(t.mass_shift, 6),
            "gains": t.formula_delta or "-",
            "loses": t.formula_loss or "-",
            "reversible": t.reversible,
            "note": t.note,
        }
        for t in ALL_TRANSFORMATIONS
    ]
    return sorted(rows, key=lambda r: r["mass_shift"])
