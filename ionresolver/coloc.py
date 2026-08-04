"""
ionresolver.coloc
-----------------
Spatial colocalisation between ion images.

In LC-MS, molecular networking uses MS/MS similarity to link related compounds.
Imaging experiments frequently have no MS/MS at all — but they carry something
LC-MS does not: a spatial distribution for every ion. Two compounds that share
a biosynthetic relationship tend to occupy related tissue compartments, so
colocalisation can stand in as an independent evidence channel.

The default metric is median-thresholded cosine similarity, which is the
established choice for imaging data: hard-thresholding at the median of the
non-zero intensities suppresses the diffuse low-level background that otherwise
makes every pair of ion images look correlated.
"""

from __future__ import annotations

import numpy as np


def _median_threshold(img: np.ndarray) -> np.ndarray:
    """Boolean mask of pixels at or above the median of the non-zero intensities.

    The comparison is inclusive. A strict ``>`` empties the mask whenever the
    non-zero intensities are all equal — two identical ion images would then
    score zero colocalisation, which is plainly wrong.
    """
    nz = img[img > 0]
    if nz.size == 0:
        return np.zeros_like(img, dtype=bool)
    return img >= np.median(nz)


def cosine_coloc(a: np.ndarray, b: np.ndarray, min_pixels: int = 20) -> float:
    """Median-thresholded cosine similarity between two ion images.

    Parameters
    ----------
    a, b : ndarray
        Flat intensity vectors over the same pixel ordering.
    min_pixels : int
        Minimum number of pixels in the combined mask. Below this the estimate
        is too unstable to report and 0.0 is returned.

    Returns
    -------
    float
        Similarity in [0, 1]. Returns 0.0 when either image is empty or the
        mask is too small.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")

    mask = _median_threshold(a) | _median_threshold(b)
    if mask.sum() < min_pixels:
        return 0.0

    u, v = a[mask], b[mask]
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    if denom == 0:
        return 0.0
    return float(np.clip(u @ v / denom, 0.0, 1.0))


def pearson_coloc(a: np.ndarray, b: np.ndarray) -> float:
    """Plain Pearson correlation, clipped to [0, 1].

    Provided for comparison. It is more sensitive to background than
    :func:`cosine_coloc` and generally reports higher values for unrelated
    ion pairs, so it should not be the default.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.clip(np.corrcoef(a, b)[0, 1], 0.0, 1.0))


def coloc_matrix(
    X: np.ndarray,
    indices: list[int] | None = None,
    metric: str = "cosine",
) -> np.ndarray:
    """Pairwise colocalisation for selected features.

    Parameters
    ----------
    X : ndarray, shape (n_pixels, n_features)
        Intensity matrix, ideally already TIC-normalised.
    indices : list of int, optional
        Feature columns to include. ``None`` uses every column, which is
        O(n_features^2) and only advisable for small feature sets.
    metric : {"cosine", "pearson"}

    Returns
    -------
    ndarray, shape (k, k)
        Symmetric matrix with 1.0 on the diagonal.
    """
    fn = {"cosine": cosine_coloc, "pearson": pearson_coloc}[metric]
    idx = list(range(X.shape[1])) if indices is None else list(indices)
    k = len(idx)
    M = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            M[i, j] = M[j, i] = fn(X[:, idx[i]], X[:, idx[j]])
    return M


def tic_normalize(X: np.ndarray) -> np.ndarray:
    """Total-ion-current normalisation, scaled to the median pixel TIC.

    Corrects for uneven matrix deposition and laser energy drift, which
    otherwise appear as intensity gradients across the section that have
    nothing to do with biology.
    """
    X = np.asarray(X, dtype=float)
    tic = X.sum(axis=1, keepdims=True)
    tic[tic == 0] = 1.0
    return X / tic * np.median(tic)


def on_tissue_mask(X: np.ndarray, percentile: float = 30.0) -> np.ndarray:
    """Boolean mask of on-tissue pixels, thresholded on total ion current.

    A crude but effective first pass. Where an optical or H&E image is
    available, prefer registering to that instead.
    """
    tic = np.asarray(X, dtype=float).sum(axis=1)
    return tic > np.percentile(tic, percentile)
