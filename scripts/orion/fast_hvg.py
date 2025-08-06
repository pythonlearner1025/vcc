"""
fast_hvg.py – ultra‑lean HVG selection for Scanpy ≥ 1.9

Usage
-----
import fast_hvg               # noqa: F401  (side‑effect: monkey‑patches Scanpy)
...
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=2000)
"""

from __future__ import annotations

import warnings
from functools import wraps
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

import scanpy as sc
from scanpy._utils import check_nonnegative_integers
from scanpy.pp._utils import _get_mean_var

# Define _get_obs_rep locally since it's not available in current scanpy version
def _get_obs_rep(adata, *, layer=None):
    """Get the observation representation from adata.X or a specific layer."""
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' not found in adata.layers")
        return adata.layers[layer]
    return adata.X

# --------------------------------------------------------------------- #
# 1.  Tiny helpers                                                      #
# --------------------------------------------------------------------- #

try:
    from skmisc.loess import loess
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Seurat‑v3 HVG flavour needs `scikit‑misc`; "
        "install via  `pip install scikit-misc`."
    ) from e


def _expected_var_loess(mean: np.ndarray, var: np.ndarray, span: float) -> np.ndarray:
    """LOESS fit in log10‑space (identical to Seurat) but vectorised."""
    ok = var > 0
    x, y = np.log10(mean[ok]), np.log10(var[ok])
    mdl = loess(x, y, span=span, degree=2)
    mdl.fit()
    exp = np.full_like(var, np.nan, dtype=np.float32)
    exp[ok] = mdl.outputs.fitted_values
    return np.power(10.0, exp)          # back‑transform


def _sum_and_sumsq_csr(
    mat: sparse.csr_matrix, clip: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per‑column ∑x and ∑x² after clipping to *clip[col]* – all in vectorised NumPy.

    Parameters
    ----------
    mat
        CSR matrix (cells × genes).
    clip
        1‑D array with length = n_genes.
    """
    data = np.minimum(mat.data.astype(np.float32, copy=False), clip[mat.indices])
    idx = mat.indices
    # `minlength` guarantees full length even when the last genes are all‑zero
    s1 = np.bincount(idx, weights=data, minlength=mat.shape[1]).astype(np.float64)
    s2 = np.bincount(idx, weights=data * data, minlength=mat.shape[1]).astype(np.float64)
    return s1, s2


# ---- NEW: native CSC path ------------------------------------------------ #
def _sum_and_sumsq_csc(
    mat: sparse.csc_matrix, clip: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    data = mat.data.astype(np.float32, copy=False)
    indptr = mat.indptr
    # Clip in‑place without an explicit per‑element loop
    np.minimum(data, np.repeat(clip, np.diff(indptr)), out=data)

    # Per‑column reductions in O(nnz) time, no Python loop
    sums   = np.add.reduceat(data, indptr[:-1]).astype(np.float64)
    sums2  = np.add.reduceat(data * data, indptr[:-1]).astype(np.float64)
    return sums, sums2


# --------------------------------------------------------------------- #
# 2.  Fast Seurat‑v3 implementation                                     #
# --------------------------------------------------------------------- #

def _highly_variable_genes_seurat_v3_fast(          # noqa: PLR0913
    adata: AnnData,
    *,
    layer: str | None,
    n_top_genes: int,
    batch_key: str | None,
    check_values: bool,
    span: float,
    subset: bool,
    inplace: bool,
) -> pd.DataFrame | None:
    """
    Drop‑in replacement for Scanpy's original `_highly_variable_genes_seurat_v3`
    that **never materialises an n_cells × n_genes dense matrix**.

    Complexity
    ----------
    *Memory* ∝ `nnz(X)` + `n_genes × n_batches` (typically a few × MB)  
    *Time*    linear in `nnz(X)`; scales to 10‑100 M cells on a laptop.
    """
    X = _get_obs_rep(adata, layer=layer)
    if check_values and not check_nonnegative_integers(X):
        warnings.warn(
            "`flavor='seurat_v3'` expects raw UMI counts; non‑integers found.",
            UserWarning,
            stacklevel=3,
        )

    # ------------- batching --------------------------------------------------
    if batch_key is None:
        batch_vec = np.zeros(adata.n_obs, dtype=np.int8)
    else:
        batch_vec = adata.obs[batch_key].to_numpy()

    batches: Sequence[int] | Sequence[str] = np.unique(batch_vec)
    n_batches, n_genes = len(batches), adata.n_vars
    norm_vars = np.empty((n_batches, n_genes), dtype=np.float32)

    for bi, b in enumerate(batches):
        rows = batch_vec == b
        X_b = X[rows]
        N = X_b.shape[0]

        mean, var = _get_mean_var(X_b)
        exp_var = _expected_var_loess(mean, var, span)
        reg_std = np.sqrt(exp_var)
        vmax = np.sqrt(N)
        clip_val = reg_std * vmax + mean          # shape (n_genes,)

        if sparse.issparse(X_b):
            if sparse.isspmatrix_csr(X_b):
                sum_, sumsq_ = _sum_and_sumsq_csr(X_b, clip_val)
            elif sparse.isspmatrix_csc(X_b):
                sum_, sumsq_ = _sum_and_sumsq_csc(X_b, clip_val)
            else:                          # fallback – avoid exotic formats
                X_b = X_b.tocsr(copy=False)
                sum_, sumsq_ = _sum_and_sumsq_csr(X_b, clip_val)
        else:  # dense fallback (already small datasets)
            X_b = np.minimum(X_b.astype(np.float32, copy=False), clip_val)
            sum_ = X_b.sum(axis=0, dtype=np.float64)
            sumsq_ = np.square(X_b, dtype=np.float32).sum(axis=0, dtype=np.float64)

        norm_var = (1.0 / ((N - 1) * reg_std**2)) * (
            (N * mean**2) + sumsq_ - 2.0 * sum_ * mean
        )
        norm_vars[bi] = norm_var.astype(np.float32, copy=False)

    # ------------- rank aggregation -----------------------------------------
    ranks = np.argsort(np.argsort(-norm_vars, axis=1), axis=1).astype(np.float32)
    hv_nbatch = (ranks < n_top_genes).sum(axis=0)
    ranks[ranks >= n_top_genes] = np.nan
    med_rank = np.nanmedian(ranks, axis=0)

    # ------------- compose output -------------------------------------------
    df = pd.DataFrame(index=adata.var_names)
    df["means"], df["variances"] = _get_mean_var(X)
    df["variances_norm"] = norm_vars.mean(axis=0)
    df["highly_variable_rank"] = med_rank
    df["highly_variable_nbatches"] = hv_nbatch

    order = np.lexsort((-hv_nbatch, med_rank))
    hv_mask = np.zeros(n_genes, dtype=bool)
    hv_mask[order[: n_top_genes]] = True
    df["highly_variable"] = hv_mask

    # ------------- write back / return --------------------------------------
    if inplace:
        adata.uns["hvg"] = {"flavor": "seurat_v3_fast"}
        # keep the same field names Scanpy writes
        adata.var["highly_variable"] = hv_mask
        adata.var["means"] = df["means"].to_numpy()
        adata.var["variances"] = df["variances"].to_numpy()
        adata.var["variances_norm"] = df["variances_norm"].to_numpy()
        adata.var["highly_variable_rank"] = med_rank
        adata.var["highly_variable_nbatches"] = hv_nbatch
        if subset:
            adata._inplace_subset_var(hv_mask)
        return None

    if subset:
        return df.loc[hv_mask]
    return df


# --------------------------------------------------------------------- #
# 3.  Monkey‑patch scanpy.pp.highly_variable_genes                       #
#     (delegate to the original for every other flavour)                #
# --------------------------------------------------------------------- #

_orig_hvg = sc.pp.highly_variable_genes   # keep a reference


@wraps(_orig_hvg)
def _hvg_wrapper(                         # noqa: PLR0913
    adata: AnnData,
    *,
    layer: str | None = None,
    n_top_genes: int | None = None,
    min_disp: float = 0.5,
    max_disp: float = np.inf,
    min_mean: float = 0.0125,
    max_mean: float = 3,
    span: float = 0.3,
    n_bins: int = 20,
    flavor: Literal["seurat", "cell_ranger", "seurat_v3", "seurat_v3_paper"] = "seurat",
    subset: bool = False,
    inplace: bool = True,
    batch_key: str | None = None,
    check_values: bool = True,
):
    if flavor == "seurat_v3":
        if n_top_genes is None:
            n_top_genes = 2000     # Scanpy default
        return _highly_variable_genes_seurat_v3_fast(
            adata,
            layer=layer,
            n_top_genes=n_top_genes,
            batch_key=batch_key,
            check_values=check_values,
            span=span,
            subset=subset,
            inplace=inplace,
        )
    # fall back to the unmodified Scanpy implementation
    return _orig_hvg(
        adata,
        layer=layer,
        n_top_genes=n_top_genes,
        min_disp=min_disp,
        max_disp=max_disp,
        min_mean=min_mean,
        max_mean=max_mean,
        span=span,
        n_bins=n_bins,
        flavor=flavor,
        subset=subset,
        inplace=inplace,
        batch_key=batch_key,
        check_values=check_values,
    )


# overwrite the function Scanpy exposes
sc.pp.highly_variable_genes = _hvg_wrapper          # type: ignore[attr-defined]

# --------------------------------------------------------------------- #
# 4.  (optional) clean namespace                                        #
# --------------------------------------------------------------------- #
del _hvg_wrapper, _orig_hvg
