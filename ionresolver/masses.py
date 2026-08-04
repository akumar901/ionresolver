"""
chemresolver.msi.masses
-----------------------
Exact-mass arithmetic for mass spectrometry imaging.

Turns a molecular formula into a monoisotopic mass, applies adducts, and
converts between neutral mass and observed m/z. This is the arithmetic layer
that everything else in ``chemresolver.msi`` is built on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Monoisotopic masses of the most abundant isotope (CODATA / AME2020).
ELEMENTS: dict[str, float] = {
    "H": 1.0078250319, "D": 2.0141017780, "C": 12.0000000000,
    "N": 14.0030740052, "O": 15.9949146221, "F": 18.9984032000,
    "Na": 22.9897692809, "Mg": 23.9850417000, "Si": 27.9769265327,
    "P": 30.9737615120, "S": 31.9720707000, "Cl": 34.9688527100,
    "K": 38.9637069000, "Ca": 39.9625912000, "Fe": 55.9349421000,
    "Cu": 62.9295989000, "Zn": 63.9291466000, "Br": 78.9183376000,
    "Se": 79.9165218000, "I": 126.9044730000,
}

ELECTRON_MASS = 0.00054858
PROTON_MASS = ELEMENTS["H"] - ELECTRON_MASS  # 1.00727646


@dataclass(frozen=True)
class Adduct:
    """An ionisation adduct.

    Parameters
    ----------
    name : str
        Display name, e.g. ``"[M+H]+"``.
    delta : float
        Mass added to (or removed from) the neutral molecule, electron
        mass already accounted for.
    charge : int
        Signed charge of the resulting ion.
    """

    name: str
    delta: float
    charge: int

    def mz(self, neutral_mass: float) -> float:
        """Observed m/z for a neutral molecule of ``neutral_mass``."""
        return (neutral_mass + self.delta) / abs(self.charge)

    def neutral(self, mz: float) -> float:
        """Inverse of :meth:`mz` — neutral mass implied by an observed m/z."""
        return mz * abs(self.charge) - self.delta


# True adducts: an intact molecule plus a charge carrier.
POSITIVE_ADDUCTS: list[Adduct] = [
    Adduct("[M+H]+", PROTON_MASS, +1),
    Adduct("[M+Na]+", ELEMENTS["Na"] - ELECTRON_MASS, +1),
    Adduct("[M+K]+", ELEMENTS["K"] - ELECTRON_MASS, +1),
    Adduct("[M+NH4]+", ELEMENTS["N"] + 4 * ELEMENTS["H"] - ELECTRON_MASS, +1),
]

NEGATIVE_ADDUCTS: list[Adduct] = [
    Adduct("[M-H]-", -PROTON_MASS, -1),
    Adduct("[M+Cl]-", ELEMENTS["Cl"] + ELECTRON_MASS, -1),
]

# In-source fragments. These are useful for *explaining* an observed peak, but
# they are not intact molecules, so seeding a biotransformation search from one
# builds the network on top of an artefact. Excluded by default.
POSITIVE_FRAGMENTS: list[Adduct] = [
    Adduct("[M+H-H2O]+", PROTON_MASS - (2 * ELEMENTS["H"] + ELEMENTS["O"]), +1),
    Adduct("[M+H-NH3]+", PROTON_MASS - (ELEMENTS["N"] + 3 * ELEMENTS["H"]), +1),
]

NEGATIVE_FRAGMENTS: list[Adduct] = [
    Adduct("[M-H2O-H]-", -PROTON_MASS - (2 * ELEMENTS["H"] + ELEMENTS["O"]), -1),
]

_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_formula(formula: str) -> dict[str, int]:
    """Parse a flat molecular formula into element counts.

    Handles simple formulae such as ``"C6H12O6"``. Bracketed or nested
    formulae are not supported.

    Raises
    ------
    ValueError
        If the string contains no recognisable elements, or names an element
        missing from :data:`ELEMENTS`.
    """
    formula = formula.strip().replace(" ", "")
    if not formula:
        raise ValueError("empty formula")

    counts: dict[str, int] = {}
    consumed = 0
    for element, digits in _FORMULA_TOKEN.findall(formula):
        if element not in ELEMENTS:
            raise ValueError(f"unknown element {element!r} in {formula!r}")
        counts[element] = counts.get(element, 0) + (int(digits) if digits else 1)
        consumed += len(element) + len(digits)

    if consumed != len(formula):
        raise ValueError(f"could not fully parse formula {formula!r}")
    return counts


def exact_mass(formula: str) -> float:
    """Monoisotopic mass of a neutral molecule given its formula."""
    return sum(ELEMENTS[el] * n for el, n in parse_formula(formula).items())


def formula_mass_defect(formula: str) -> float:
    """Mass defect — the fractional part of the exact mass.

    Useful as a coarse sanity filter: lipids sit high, sugars sit low, and a
    candidate whose defect is wildly out of line with the observed ion is
    usually wrong regardless of how well the integer mass matches.
    """
    m = exact_mass(formula)
    return m - int(m)


def ppm_error(observed: float, theoretical: float) -> float:
    """Signed mass error in parts per million."""
    return (observed - theoretical) / theoretical * 1e6


def within_ppm(observed: float, theoretical: float, tol: float) -> bool:
    """True when ``observed`` sits within ``tol`` ppm of ``theoretical``."""
    return abs(ppm_error(observed, theoretical)) <= tol


def mz_window(mz: float, tol_ppm: float) -> tuple[float, float]:
    """Inclusive (low, high) m/z bounds for a ppm tolerance."""
    d = mz * tol_ppm * 1e-6
    return mz - d, mz + d


def adducts_for(polarity: str, include_fragments: bool = False) -> list[Adduct]:
    """Return the adduct list for a polarity.

    Parameters
    ----------
    polarity : {"positive", "negative"}
    include_fragments : bool
        Add in-source fragment species such as ``[M+H-H2O]+``. These help
        explain observed peaks, but they are fragments rather than intact
        molecules, so they make poor seeds for a biotransformation search —
        any network built from them inherits the artefact. Off by default.
    """
    p = polarity.strip().lower()
    if p.startswith("pos"):
        base, frag = POSITIVE_ADDUCTS, POSITIVE_FRAGMENTS
    elif p.startswith("neg"):
        base, frag = NEGATIVE_ADDUCTS, NEGATIVE_FRAGMENTS
    else:
        raise ValueError(
            f"polarity must be 'positive' or 'negative', got {polarity!r}"
        )
    return list(base) + list(frag) if include_fragments else list(base)
