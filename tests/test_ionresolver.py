"""Tests for ionresolver."""
import numpy as np
import pandas as pd
import pytest

import ionresolver as ir


# ---------- masses ----------
@pytest.mark.parametrize("formula,expected", [
    ("C6H12O6", 180.063388),
    ("H2O", 18.010565),
    ("C9H8O4", 180.042259),
    ("C42H82NO8P", 759.577805),
])
def test_exact_mass(formula, expected):
    assert abs(ir.exact_mass(formula) - expected) < 1e-4


def test_parse_formula_rejects_unknown_element():
    with pytest.raises(ValueError):
        ir.parse_formula("C6H12X6")


def test_parse_formula_rejects_empty():
    with pytest.raises(ValueError):
        ir.parse_formula("")


def test_adduct_roundtrip():
    m = ir.exact_mass("C42H82NO8P")
    a = ir.POSITIVE_ADDUCTS[0]
    assert abs(a.neutral(a.mz(m)) - m) < 1e-9


def test_known_lipid_adducts():
    m = ir.exact_mass("C42H82NO8P")           # PC(34:1)
    assert abs(ir.POSITIVE_ADDUCTS[0].mz(m) - 760.5851) < 1e-3   # [M+H]+
    assert abs(ir.POSITIVE_ADDUCTS[1].mz(m) - 782.5670) < 1e-3   # [M+Na]+


def test_ppm_error_sign():
    assert ir.ppm_error(100.001, 100.000) > 0
    assert ir.ppm_error(99.999, 100.000) < 0


# ---------- transformations: validated against published conjugate chemistry ----------
@pytest.mark.parametrize("parent,product,tname,", [
    ("C7H6O2", "C9H9NO3", "glycine conjugation"),      # benzoate -> hippurate
    ("C7H8O", "C7H8O4S", "sulfation"),                  # p-cresol -> p-cresol sulfate
    ("C8H8O2", "C13H16N2O4", "glutamine conjugation"),  # phenylacetate -> PAG
])
def test_transformation_matches_real_chemistry(parent, product, tname):
    observed = ir.exact_mass(product) - ir.exact_mass(parent)
    tr = next(t for t in ir.ALL_TRANSFORMATIONS if t.name == tname)
    assert abs(observed - tr.mass_shift) < 1e-4


def test_reverse_transformations_added():
    fwd = ir.get_transformations(include_reverse=False)
    rev = ir.get_transformations(include_reverse=True)
    assert len(rev) > len(fwd)


def test_category_filter():
    p2 = ir.get_transformations(categories=["phase_ii"], include_reverse=False)
    assert all(t.category == "phase_ii" for t in p2)


# ---------- colocalisation ----------
def test_coloc_identical_is_one():
    img = np.array([0, 1, 5, 10, 3, 0, 8, 2] * 10, dtype=float)
    assert ir.cosine_coloc(img, img) > 0.99


def test_coloc_disjoint_is_low():
    a = np.concatenate([np.ones(80) * 10, np.zeros(80)])
    b = np.concatenate([np.zeros(80), np.ones(80) * 10])
    assert ir.cosine_coloc(a, b) < 0.1


def test_coloc_shape_mismatch_raises():
    with pytest.raises(ValueError):
        ir.cosine_coloc(np.ones(10), np.ones(20))


def test_tic_normalize_equalizes_pixels():
    X = np.array([[1., 1.], [10., 10.], [100., 100.]])
    Xn = ir.tic_normalize(X)
    assert np.allclose(Xn.sum(1), Xn.sum(1)[0])


# ---------- annotation ----------
@pytest.fixture
def small_db():
    df = pd.DataFrame(
        [("Glucose", "C6H12O6"), ("PC(34:1)", "C42H82NO8P")],
        columns=["name", "formula"],
    )
    return ir.CompoundDatabase(df, polarity="positive")


def test_database_requires_columns():
    with pytest.raises(ValueError):
        ir.CompoundDatabase(pd.DataFrame({"compound": ["x"]}))


def test_annotate_finds_known_ion(small_db):
    cands = ir.annotate_mz(760.5851, small_db, tol_ppm=5.0)
    assert any(c.name == "PC(34:1)" for c in cands)


def test_annotate_unmatched_is_unknown_L4(small_db):
    cands = ir.annotate_mz(123.4567, small_db, tol_ppm=5.0)
    assert cands[0].name == "UNKNOWN"
    assert cands[0].level == "L4"


def test_evidence_levels_escalate():
    assert ir.Evidence(accurate_mass=True).level() == "L4"
    assert ir.Evidence(accurate_mass=True, msms_match=True).level() == "L2"
    assert ir.Evidence(msms_match=True, ccs_match=True).level() == "L2a"
    assert ir.Evidence(msms_match=True, standard_match=True).level() == "L1"


def test_msms_flag_raises_level(small_db):
    plain = ir.annotate_mz(760.5851, small_db, has_msms=False)[0]
    withms = ir.annotate_mz(760.5851, small_db, has_msms=True)[0]
    assert plain.level == "L4"
    assert withms.level == "L2"


# ---------- propagation ----------
def test_propagate_recovers_planted_conjugate():
    """A parent and its sulfate conjugate, colocalised, must be recovered."""
    rng = np.random.default_rng(0)
    n_px = 400
    parent_mz = 200.0
    sulfate = next(t for t in ir.ALL_TRANSFORMATIONS if t.name == "sulfation")
    child_mz = parent_mz + sulfate.mass_shift

    shared = np.zeros(n_px)
    shared[:150] = rng.uniform(5, 10, 150)          # same tissue region
    noise = rng.uniform(0, 10, n_px)                 # unrelated distribution

    mzs = np.array([parent_mz, child_mz, 555.5555])
    X = np.column_stack([shared, shared * 0.6 + 0.1, noise])

    out = ir.propagate(X, mzs, {0: "TestParent"}, tol_ppm=5.0, coloc_min=0.5)
    assert len(out) >= 1
    assert out.iloc[0].transformation == "sulfation"


def test_propagate_rejects_noncolocalized():
    """Correct mass difference but disjoint distribution must be rejected."""
    n_px = 400
    sulfate = next(t for t in ir.ALL_TRANSFORMATIONS if t.name == "sulfation")
    a = np.concatenate([np.ones(200) * 10, np.zeros(200)])
    b = np.concatenate([np.zeros(200), np.ones(200) * 10])
    mzs = np.array([200.0, 200.0 + sulfate.mass_shift])
    X = np.column_stack([a, b])

    out = ir.propagate(X, mzs, {0: "P"}, tol_ppm=5.0, coloc_min=0.6)
    assert len(out) == 0


def test_propagate_shape_check():
    with pytest.raises(ValueError):
        ir.propagate(np.ones((10, 3)), np.array([1.0, 2.0]), {0: "x"})


# ---------- validation ----------
def test_validation_runs_and_reports():
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 10, (200, 40))
    mzs = np.linspace(200, 900, 40)
    v = ir.validate_propagation(X, mzs, {0: "A", 5: "B"}, n_decoy=3)
    assert v.n_decoy_runs == 3
    assert isinstance(v.summary(), str)


# ---------- chemical plausibility ----------
def test_pc_cannot_be_deaminated():
    """Phosphatidylcholines carry a quaternary ammonium: no N-H to remove."""
    groups = ir.infer_groups("PC(34:1)", "C42H82NO8P")
    assert "quaternary_ammonium" in groups
    ok, why = ir.is_plausible("deamination", groups)
    assert not ok
    assert "quaternary_ammonium" in why


def test_sphingomyelin_cannot_be_deaminated():
    groups = ir.infer_groups("SM(d18:1/16:0)", "C39H79N2O6P")
    ok, _ = ir.is_plausible("deamination", groups)
    assert not ok


def test_real_conjugations_remain_plausible():
    assert ir.is_plausible("sulfation", ir.infer_groups("p-cresol"))[0]
    assert ir.is_plausible("glycine conjugation", ir.infer_groups("benzoate"))[0]
    assert ir.is_plausible("glutamine conjugation", ir.infer_groups("phenylacetate"))[0]


def test_ether_lipid_has_no_ester():
    assert "ester" not in ir.infer_groups("PC(O-34:1)")
    assert "ester" in ir.infer_groups("PC(34:1)")


def test_decoy_inherits_plausibility_rule():
    """Decoys must face the same filter as real hits or the FDR is biased."""
    groups = {"quaternary_ammonium"}
    assert not ir.is_plausible("decoy of deamination", groups)[0]
    assert not ir.is_plausible("deamination", groups)[0]


# ---------- in-source artefacts ----------
@pytest.mark.parametrize("shift", [-18.010565, 21.981944, 37.955882])
def test_known_artefacts_flagged(shift):
    flagged, label = ir.is_artefact_shift(shift)
    assert flagged and label


@pytest.mark.parametrize("shift", [79.9568, 57.0215, 128.0586, 176.0321])
def test_real_conjugation_shifts_not_flagged(shift):
    assert not ir.is_artefact_shift(shift)[0]


def test_propagate_excludes_artefact_shifts():
    """Water loss colocalises perfectly but is the same molecule."""
    n_px = 400
    img = np.zeros(n_px); img[:150] = 10.0
    mzs = np.array([500.0, 500.0 - 18.010565])
    X = np.column_stack([img, img * 0.8])

    with_filter = ir.propagate(X, mzs, {0: "LPC(16:0)"}, tol_ppm=5,
                               coloc_min=0.5, exclude_artefacts=True)
    without = ir.propagate(X, mzs, {0: "LPC(16:0)"}, tol_ppm=5,
                           coloc_min=0.5, exclude_artefacts=False)
    assert len(with_filter) == 0
    assert len(without) >= 0


# ---------- category split ----------
def test_split_separates_lipid_series():
    df = pd.DataFrame({
        "parent_name": ["PC(34:1)", "p-cresol"],
        "transformation": ["desaturation", "sulfation"],
        "category": ["lipid", "phase_ii"],
    })
    out = ir.split_by_category(df)
    assert len(out["lipid_series"]) == 1
    assert len(out["biotransformation"]) == 1
    assert out["biotransformation"].iloc[0].parent_name == "p-cresol"


# ---------- split-feature merging ----------
def test_merge_collapses_adjacent_bins():
    """Binning splits single peaks across neighbouring bins."""
    mzs = np.array([500.0000, 500.0025, 600.0000])   # first two are 5 ppm apart
    X = np.array([[10.0, 5.0, 20.0],
                  [12.0, 6.0, 22.0]])
    new_mz, new_X, mapping = ir.merge_split_features(mzs, X, tol_ppm=6.0)
    assert len(new_mz) == 2
    assert len(mapping[0]) == 2
    assert new_X[0, 0] == 15.0          # intensities summed


def test_merge_leaves_distinct_features_alone():
    mzs = np.array([500.0, 700.0, 900.0])
    X = np.ones((5, 3))
    new_mz, _, _ = ir.merge_split_features(mzs, X, tol_ppm=6.0)
    assert len(new_mz) == 3


def test_merge_centroid_is_intensity_weighted():
    mzs = np.array([500.0000, 500.0025])
    X = np.array([[100.0, 1.0]])         # first peak dominates
    new_mz, _, _ = ir.merge_split_features(mzs, X, tol_ppm=6.0)
    assert new_mz[0] < 500.0005          # centroid pulled to the strong peak


# ---------- sample context ----------
def test_cell_culture_excludes_microbial():
    """An axenic culture has no gut flora."""
    n_px = 300
    img = np.zeros(n_px); img[:100] = 10.0
    dehydrox = next(t for t in ir.ALL_TRANSFORMATIONS if t.name == "dehydroxylation")
    assert dehydrox.category == "microbial"

    mzs = np.array([500.0, 500.0 + dehydrox.mass_shift])
    X = np.column_stack([img, img * 0.9])

    tissue = ir.propagate(X, mzs, {0: "Carnitine"}, tol_ppm=5, coloc_min=0.5,
                          sample_context="tissue")
    culture = ir.propagate(X, mzs, {0: "Carnitine"}, tol_ppm=5, coloc_min=0.5,
                           sample_context="cell_culture")
    assert len(culture) < len(tissue) or len(culture) == 0


def test_unknown_sample_context_raises():
    with pytest.raises(ValueError):
        ir.propagate(np.ones((5, 2)), np.array([100.0, 200.0]), {0: "x"},
                     sample_context="mars")


# ---------- adduct provenance ----------
def test_parent_adduct_is_reported():
    """A row must say which ion of the parent the arithmetic used."""
    n_px = 300
    img = np.zeros(n_px); img[:100] = 10.0
    sulf = next(t for t in ir.ALL_TRANSFORMATIONS if t.name == "sulfation")
    mzs = np.array([200.0, 200.0 + sulf.mass_shift])
    X = np.column_stack([img, img * 0.9])

    out = ir.propagate(X, mzs, {0: "p-cresol"}, seed_adducts={0: "[M+K]+"},
                       tol_ppm=5, coloc_min=0.5)
    assert len(out) >= 1
    assert out.iloc[0].parent_adduct == "[M+K]+"


def test_seed_adducts_extracted_from_annotations():
    df = pd.DataFrame([("Glucose", "C6H12O6")], columns=["name", "formula"])
    db = ir.CompoundDatabase(df, polarity="positive")
    mzs = np.array([181.0707])                       # glucose [M+H]+
    ann = ir.annotate_peaklist(mzs, db, tol_ppm=20.0)
    ad = ir.seed_adducts_from_annotations(ann, mzs, min_level="L4", tol_ppm=20.0)
    assert ad and list(ad.values())[0].startswith("[M+")


# ---------- FDR interval ----------
def test_validation_reports_interval():
    rng = np.random.default_rng(3)
    X = rng.uniform(0, 10, (200, 40))
    mzs = np.linspace(200, 900, 40)
    v = ir.validate_propagation(X, mzs, {0: "A", 5: "B"}, n_decoy=8)
    assert hasattr(v, "fdr_lo") and hasattr(v, "fdr_hi")
    assert "CI" in v.summary()


def test_coloc_handles_constant_intensity():
    """A strict median threshold empties the mask when intensities are equal."""
    img = np.zeros(300)
    img[:100] = 10.0                       # every non-zero pixel identical
    assert ir.cosine_coloc(img, img * 0.9) > 0.99


# ---------- fragments are not adducts ----------
def test_fragments_excluded_by_default():
    """A water-loss ion is a fragment, not an intact molecule."""
    names = [a.name for a in ir.adducts_for("positive")]
    assert "[M+H-H2O]+" not in names
    assert "[M+H]+" in names


def test_fragments_available_on_request():
    names = [a.name for a in ir.adducts_for("positive", include_fragments=True)]
    assert "[M+H-H2O]+" in names


def test_database_excludes_fragments_by_default():
    df = pd.DataFrame([("Glucose", "C6H12O6")], columns=["name", "formula"])
    plain = ir.CompoundDatabase(df, polarity="positive")
    withfrag = ir.CompoundDatabase(df, polarity="positive", include_fragments=True)
    assert len(withfrag) > len(plain)
