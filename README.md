# ionresolver

Annotation of imaging mass spectrometry data with explicit confidence levels.

Companion package to [chemresolver](https://github.com/akumar901/chemresolver-).
`chemresolver` resolves compound **names** to structures. `ionresolver` resolves
observed **ions** to identities — and reports how much each identity is worth
trusting.

---

## The problem

Imaging mass spectrometry gives you an m/z and an intensity for every pixel.
It does not give you compound identities. Two things make that hard:

1. **Chromatography is gone.** Isomers that would have separated by retention
   time collapse into a single m/z. At 5 ppm, m/z 782.567 matches both
   PC(34:1) and the ether lipid PC(O-34:1) — and those can localise to
   different tissue regions.
2. **Most features stay unannotated.** On a real METASPACE dataset (human
   colon, FT-ICR, 1,866 features), accurate-mass search against a curated
   compound list annotates roughly 3% of features. The rest are unknowns.

Downstream analysis usually ignores both problems. A bare 5 ppm mass match is
treated exactly like a standard-confirmed identification, and every conclusion
silently inherits that uncertainty.

## What this package does

**Attaches an evidence trail to every annotation.** Each candidate carries an
explicit record of which criteria it satisfied — accurate mass, isotope
pattern, MS/MS, CCS, authentic standard, spatial colocalisation — and a
confidence level derived from them, following the Metabolomics Standards
Initiative.

**Propagates identities to unknowns through biotransformation networks.** Most
unknowns are not novel chemistry; they are known metabolites that have been
sulfated, glucuronidated, methylated, or conjugated in vivo. Each modification
carries a fixed exact mass. If an unknown sits at an annotated parent's m/z
plus a valid biotransformation shift **and** colocalises with that parent, that
is real evidence for a putative identity.

The spatial constraint is what makes this imaging-specific. LC-MS has molecular
networking based on MS/MS similarity; imaging experiments often have no MS/MS
at all, but they carry a spatial distribution for every ion. Colocalisation
becomes the substitute evidence channel.

**Filters on chemical plausibility.** Mass and colocalisation together still
admit assignments that are chemically impossible. A phosphatidylcholine cannot
be deaminated — its nitrogen is a quaternary ammonium with no hydrogens to
lose. Functional-group rules add a third constraint independent of both mass
and spatial distribution.

**Distinguishes adducts from fragments.** `[M+H-H2O]+` is an in-source
fragment, not an intact molecule. Seeding a biotransformation search from one
builds the network on top of an artefact — on the test dataset that accounted
for a quarter of all hits. Fragment species are excluded from seeding by
default and available via `include_fragments=True` when the goal is simply to
explain observed peaks.

**Merges split features and tracks provenance.** Binning a processed imzML
routinely splits one peak across adjacent bins, so the same relationship gets
counted twice under different names. Merged features carry an
intensity-weighted centroid, and every propagated row records which *adduct*
of the parent the mass arithmetic used — without that, a row reading
"PC(32:0) + hydration" cannot be checked by hand.

**Respects sample context.** Microbial transformations cannot occur in an
axenic cell culture, however well the mass and colocalisation agree. Setting
``sample_context="cell_culture"`` removes that chemistry from the search.

**Excludes in-source artefacts.** A feature at parent − 18.0106 is usually
water loss from the parent ion, not a dehydrated molecule, and it colocalises
perfectly *because it is the same molecule*. Adduct differences behave the
same way. On real data these artefacts produce the highest-scoring and least
meaningful hits.

**Controls the false discovery rate.** Any mass-difference search returns
coincidences. Real transformation shifts are compared against decoy shifts
drawn from the same magnitude range, and the decoy hit rate estimates the FDR.
Decoys inherit the plausibility rule of the transformation they replace, so
both sides of the comparison face the same filters — otherwise the estimate is
biased upward.

---

## Install

```bash
pip install ionresolver              # core
pip install "ionresolver[imzml]"     # + imzML reading
```

From source:

```bash
git clone https://github.com/akumar901/ionresolver.git
cd ionresolver
pip install -e ".[dev]"
pytest
```

## Quick start

```python
import numpy as np
import pandas as pd
import ionresolver as ir

# X: (n_pixels, n_features) intensities;  mzs: (n_features,) m/z axis
# merge peaks split across adjacent bins before anything else
mzs, X, _ = ir.merge_split_features(mzs, X, tol_ppm=6.0)

on = ir.on_tissue_mask(X, percentile=30)
Xn = ir.tic_normalize(X[on])

db = ir.CompoundDatabase(
    pd.DataFrame([("PC(34:1)", "C42H82NO8P"),
                  ("LPC(16:0)", "C24H50NO7P")],
                 columns=["name", "formula"]),
    polarity="positive",
)

ann = ir.annotate_peaklist(mzs, db, tol_ppm=5.0)
print(ir.confidence_summary(ann))

seeds   = ir.seeds_from_annotations(ann, mzs, min_level="L4")
adducts = ir.seed_adducts_from_annotations(ann, mzs, min_level="L4")

prop = ir.propagate(
    Xn, mzs, seeds,
    seed_adducts=adducts,          # so each row records the parent ion used
    tol_ppm=5.0,
    coloc_min=0.6,
    sample_context="cell_culture", # drops microbial chemistry
)

# separate homologous-series relationships from genuine modifications
split = ir.split_by_category(prop)
print(split["biotransformation"])

result = ir.validate_propagation(Xn, mzs, seeds, coloc_min=0.6, n_decoy=20)
print(result.summary())
```

## Confidence levels

| Level | Meaning |
|-------|---------|
| L1  | Confirmed against an authentic standard under matched conditions |
| L2a | Library match on both MS/MS and CCS |
| L2  | Library match on MS/MS or CCS |
| L3  | Putative — accurate mass plus spatial corroboration |
| L4  | Molecular formula only, or unknown |

Propagated annotations are reported as **L3**. They are hypotheses worth
testing by MS/MS, not confirmed identifications.

---

## Proof of concept

Run against a public METASPACE dataset — human colon 3D culture, MALDI, DHB
matrix, FT-ICR at 80,000 resolving power, positive mode, 4,850 pixels, 1,866
features after occupancy filtering.

Seeded from 20 compounds (58 annotated features), with an 8 ppm tolerance and a
0.6 colocalisation threshold:

The dataset is a **3D cell culture**, not tissue — there is no microbiome and
no hepatic conjugation machinery, so the chemistry available is essentially
membrane lipids. It is a sound technical test and a limited biological one.

Effect of each filter, seeded from 20 compounds (58 annotated features) at
8 ppm and a 0.6 colocalisation threshold:

| Stage | Hits |
|-------|------|
| Mass + colocalisation only | 119 |
| + collapse mass-degenerate transformations | 119 |
| + chemical plausibility | 73 |
| + in-source artefact exclusion | 62 |
| + merge split bins, exclude microbial (cell culture) | 49 |
| + exclude in-source fragments as seeds | 39 |

Final validation:

| Metric | Value |
|--------|-------|
| Propagated annotations | 39 |
| Decoy hits (n = 50 runs) | 6.6 ± 4.6 |
| Estimated FDR | 15.0% (95% CI 0.5–31.8%) |
| Enrichment over chance | 6.6× |
| z-score vs decoy null | 8.2 |

Of the 39, **35 are lipid-series relationships** and **4 are genuine
modifications**. The split matters: members of a homologous lipid series share
membranes and colocalise almost by construction, so those relationships are
real chemistry but weak evidence.

The four survivors, with the parent adduct that the arithmetic used:

| m/z | Parent | Adduct | Transformation | Coloc |
|-----|--------|--------|----------------|-------|
| 398.326 | Palmitoylcarnitine | [M+H]+ | dehydrogenation | 0.963 |
| 814.538 | PC(34:1) | [M+K]+ | hydroxylation | 0.897 |
| 790.537 | PC(32:0) | [M+K]+ | hydration | 0.800 |
| 788.521 | PC(32:0) | [M+K]+ | epoxidation | 0.739 |

Three of four derive from potassium adducts, which is worth checking rather
than assuming — a pattern that concentrated on one adduct would suggest an
unmodelled adduct relationship rather than real chemistry.

**The FDR interval is wide on purpose.** The point estimate moves by roughly
ten percentage points between random seeds at n=10 decoy runs. Reporting a
single number would overstate what the data supports.

Threshold sweep:

| coloc_min | real hits | decoy mean | FDR |
|-----------|-----------|------------|-----|
| 0.4 | 157 | 29.8 | 18.9% |
| 0.5 | 138 | 23.5 | 17.0% |
| 0.6 | 119 | 17.8 | 14.9% |
| 0.7 |  86 | 11.5 | 13.4% |
| 0.8 |  53 |  6.6 | 12.5% |
| 0.9 |  15 |  1.8 | 11.7% |

Recovered relationships include palmitoylcarnitine → stearoylcarnitine (C2H4
chain extension, coloc 0.974), LPC(16:0) → LPC(16:1) (desaturation, 0.906), and
PC(34:1) → PC(34:2) (desaturation, 0.919) — correct lipid chemistry recovered
without any MS/MS.

Reproduce with `examples/poc_metaspace.py`.

---

## Validation of the transformation catalogue

The Phase II conjugations are checked in the test suite against published
metabolite pairs:

| Reaction | Expected shift | Catalogue |
|----------|---------------|-----------|
| benzoate → hippurate (glycine) | 57.0215 | 57.0215 |
| p-cresol → p-cresol sulfate (sulfation) | 79.9568 | 79.9568 |
| phenylacetate → phenylacetylglutamine (glutamine) | 128.0586 | 128.0586 |

These three are host conjugates of gut microbial metabolites — the same
chemistry reported as biomarkers of methotrexate non-response in
[Kumar et al., *Biomed Pharmacother* 2025;193:118755](https://doi.org/10.1016/j.biopha.2025.118755).

---

## Limitations

- Propagated annotations are **Level 3**. They require MS/MS confirmation.
- Colocalisation cannot distinguish a biotransformation product from an
  in-source fragment of the same molecule. The artefact list handles the common
  cases; it is not exhaustive.
- Functional groups are inferred from compound class and formula, not from
  structure. Supply a `groups` column in the compound database for anything
  where the inference is likely wrong.
- Recall depends on the seed set in a way that is not yet characterised.
  Twenty seed compounds produced 58 seeded features across their adducts;
  results shift noticeably when that list changes.
- Recall scales with the seed database. Twenty seed compounds annotate a small
  fraction of features; a full HMDB-scale seed set is the obvious next step.
- Colocalisation assumes a metabolite and its conjugate occupy related tissue
  compartments. That is often but not always true — a conjugate destined for
  export may localise to different structures than its parent.
- The FDR estimate depends on decoy shifts being chemically impossible. Decoys
  are offset to avoid coinciding with real elemental compositions, but the
  null is approximate.
- No CCS prediction yet. On ion-mobility instruments this is the single largest
  available gain, since CCS separates isomers that mass cannot.

## Roadmap

- HMDB / LIPID MAPS seed loading via `chemresolver`
- CCS as a scored evidence dimension, measured and predicted
- Direct imzML ingestion, including processed-mode binning
- Isotope-pattern scoring rather than a boolean flag
- Export to Cytoscape for network visualisation

## License

MIT
