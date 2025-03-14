import functools
from pathlib import Path
import itertools
from tqdm import tqdm
from typing import Literal, Union

import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.decomposition import non_negative_factorization
from sklearn.metrics.pairwise import cosine_similarity
from anndata import AnnData
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Reading spatial Query Dataset (as a counts csv) and annotate with the nico generated annotation adata object
adata_sp = (lambda f: lambda counts_csv_path, adata_annotation_path: f(sc.read(counts_csv_path).transpose(), sc.read_h5ad(adata_annotation_path)))(
    lambda adata, adata_annotation: sc.AnnData(X=adata.X, obs=adata.obs.join(adata_annotation.obs, how="outer"), var=adata.var)
)(counts_csv_path=Path.cwd() / "data" / "inputQuery" / "gene_by_cell.csv",
  adata_annotation_path=Path.cwd() / "data" / "nico_out" / "nico_celltype_annotation.h5ad")

# Reading single cell reference dataset
adata_sc = sc.read_h5ad(Path.cwd() / "data" / "inputRef" / "input_ref.h5ad")

sc.pp.filter_genes(adata_sp, min_counts=1)
sc.pp.filter_genes(adata_sc, min_counts=1)

shared_genes_mask_sc = np.isin(adata_sc.var_names, adata_sp.var_names)
shared_genes_mask_sp = np.isin(adata_sp.var_names, adata_sc.var_names)
shared_genes = adata_sc[:, shared_genes_mask_sc].var_names

print(f"{len(shared_genes)=}")

def nmf_transfer(
        adata_query: AnnData, 
        adata_reference: AnnData, 
        shared_genes_mask_reference: np.ndarray, 
        gene_filter_mask: np.ndarray | None = None,
        n_components: int | None = None, 
        nmf_func: callable = non_negative_factorization,
        W_init_reference: np.ndarray | None = None,
        H_init_reference: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    shared_genes = adata_reference[:, shared_genes_mask_reference].var_names

    if gene_filter_mask is None:
        gene_filter_mask = np.ones(len(shared_genes), dtype=bool)

    init_reference = "custom" if W_init_reference is not None or H_init_reference is not None else None

    if W_init_reference is not None:
        W_init_reference = W_init_reference.copy(order="C")

    W, H, _, W_init, H_init = (
        lambda W_init, H_init, _: (
            *nmf_func(
                adata_query[:, shared_genes][:, gene_filter_mask].X.toarray(), 
                H=H_init[:, gene_filter_mask], 
                init="custom", 
                update_H=False
            ), 
            W_init, 
            H_init)
        )(
            *nmf_func(
                adata_reference[:, shared_genes_mask_reference].X.toarray(), 
                init=init_reference, 
                W=W_init_reference, 
                H=H_init_reference, 
                n_components=n_components
            )
        )
    return W, H, W_init, H_init

df = (
    lambda adata_query, adata_reference, shared_genes_mask_reference, n_components_range, shared_genes, celltypes: pd.DataFrame(
        [
            (
                celltype,
                n_components, 
                pearsonr(
                    np.mean(
                        adata_query[:, shared_genes].X, 
                        axis=0
                    ),
                    np.mean(
                        np.hstack(
                            [
                                (lambda W, H, W_init, H_init, arg: W @ H_init[:, ~arg])(
                                    *(lambda func, arg: (*func(arg), arg))(
                                        functools.partial(
                                            nmf_transfer, 
                                            adata_query[adata_query.obs["celltype"] == celltype], 
                                            adata_reference[adata_reference.obs["celltype"] == celltype], 
                                            shared_genes_mask_reference, 
                                            n_components=n_components), 
                                        ~np.isin(shared_genes, gene)
                                    )
                                )
                                for gene in shared_genes
                            ]
                        ),
                        axis=0
                    )
                )[0]
            )
        for celltype, n_components in tqdm(itertools.product(celltypes, n_components_range))]
    )
)(
    adata_sp,
    adata_sc,
    shared_genes_mask_sc,
    n_components_range=[2, 3, 4, 6, 8],
    shared_genes=shared_genes,
    celltypes=list(set.intersection(set(adata_sp.obs["celltype"]), set(adata_sc.obs["celltype"])))
).to_csv("./beaseline_reconstruction_error.csv")