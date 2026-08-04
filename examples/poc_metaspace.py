"""
Proof of concept: ionresolver on a real METASPACE dataset.

Dataset: human colon (3D culture), MALDI, DHB matrix, FT-ICR @ 80,000 RP,
positive mode, 4,850 pixels. From the METASPACE engine test data.
"""
import numpy as np, pandas as pd, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ionresolver as ir

DATA = "/home/claude/real"
X   = np.load(f"{DATA}/X.npy")
mzs = np.load(f"{DATA}/mzs.npy")
print(f"loaded {X.shape[0]} pixels x {X.shape[1]} features")

# --- preprocess ---
on = ir.on_tissue_mask(X, percentile=30)
Xn = ir.tic_normalize(X[on])
print(f"on-tissue: {on.sum()} pixels\n")

# --- seed database ---
DB = pd.DataFrame([
 ("Choline","C5H13NO"),("Betaine","C5H11NO2"),("Carnitine","C7H15NO3"),
 ("Creatine","C4H9N3O2"),("Phenylalanine","C9H11NO2"),("Tryptophan","C11H12N2O2"),
 ("Adenine","C5H5N5"),("Adenosine","C10H13N5O4"),("Glutathione","C10H17N3O6S"),
 ("Spermidine","C7H19N3"),("Glucose","C6H12O6"),("Palmitoylcarnitine","C23H45NO4"),
 ("LPC(16:0)","C24H50NO7P"),("LPC(18:0)","C26H54NO7P"),("LPC(18:1)","C26H52NO7P"),
 ("SM(d18:1/16:0)","C39H79N2O6P"),("PC(32:0)","C40H80NO8P"),("PC(34:1)","C42H82NO8P"),
 ("PC(36:2)","C44H86NO8P"),("PC(38:4)","C46H86NO8P"),
], columns=["name","formula"])

db = ir.CompoundDatabase(DB, polarity="positive")
print(f"database: {db.n_compounds} compounds x {len(db.adducts)} adducts = {len(db)} ions\n")

# --- annotate ---
ann = ir.annotate_peaklist(mzs, db, tol_ppm=8.0)
print("CONFIDENCE DISTRIBUTION")
print(ir.confidence_summary(ann).to_string(index=False))

seeds = ir.seeds_from_annotations(ann, mzs, min_level="L4", tol_ppm=8.0)
print(f"\nseeds: {len(seeds)} features annotated by accurate mass")
print(f"unknowns: {len(mzs)-len(seeds)}\n")

# --- propagate ---
prop = ir.propagate(Xn, mzs, seeds, tol_ppm=8.0, coloc_min=0.60)
print("="*72); print("PROPAGATED ANNOTATIONS"); print("="*72)
if len(prop):
    print(prop[["mz_unknown","proposed_name","transformation","ppm","coloc"]]
          .head(20).to_string(index=False))
    print(f"\n{len(prop)} unknowns given a putative identity")
else:
    print("none")

# --- validate ---
print("\n"+"="*72); print("DECOY VALIDATION"); print("="*72)
v = ir.validate_propagation(Xn, mzs, seeds, tol_ppm=8.0, coloc_min=0.60, n_decoy=20)
print(v.summary())

print("\n"+"="*72); print("THRESHOLD SWEEP"); print("="*72)
print(ir.threshold_sweep(Xn, mzs, seeds, n_decoy=8, tol_ppm=8.0).to_string(index=False))

prop.to_csv(f"{DATA}/ionresolver_poc_results.csv", index=False)
print(f"\nsaved results")
