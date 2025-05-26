from typing import Any, Callable, Generator, Hashable, Iterable, Optional

from anndata import AnnData

import numpy as np
from numpy.typing import NDArray

from pandas import DataFrame, Series, Index
from pandas._typing import Axis, Scalar

from typing import Callable, Optional
from pandas._typing import Axis, Scalar
import pandas as pd
from pandas import DataFrame, Series

def subsample_adata(adata: AnnData, n_obs: int, seed: Optional[int] = None) -> AnnData:
    """
    Subsample a given AnnData object to a specified number of observations (cells).

    Parameters
    ----------
    adata : AnnData
        The input annotated data matrix to subsample from.
    n_obs : int
        The number of observations (cells) to sample without replacement.
    seed : int, optional
        Seed for the random number generator for reproducibility.

    Returns
    -------
    AnnData
        A new AnnData object containing the subsampled observations.
    """
    rng = np.random.default_rng(seed=seed)
    cell_id = rng.choice(np.arange(adata.n_obs), n_obs, replace=False)
    return adata[cell_id].copy()

def sort_df(
    df: DataFrame,
    func: Callable[[Series], Scalar],
    axis: Axis = 0,
    ascending: bool = True
) -> DataFrame:
    values = df.apply(func, axis=axis) # type: ignore
    sorter = values.sort_values(ascending=ascending).index

    axis_num = df._get_axis_number(axis) # type: ignore
    match axis_num:
        case 0:
            return df.loc[:, sorter] # type: ignore
        case 1: 
            return df.loc[sorter]
        case _:
            raise ValueError()

def filter_celltypes_by_size(
    adata: AnnData,
    obs_column: str,
    min_cell_count: int,
) -> AnnData:
    """
    Keep only cell types with at least `min_cell_count` cells.

    Arguments:
        adata: AnnData object.
        obs_column: Column in `adata.obs` with cell type labels.
        min_cell_count: Minimum number of cells per cell type to keep.

    Returns:
        Filtered AnnData object.
    """
    celltype_counts = adata.obs[obs_column].value_counts()
    valid_celltypes = list(celltype_counts[celltype_counts >= min_cell_count].index)
    return adata[adata.obs[obs_column].isin(valid_celltypes)].copy()

def get_aligned_gene_sets(
    genes: tuple[Index, Index],
) -> tuple[NDArray[np.str_], NDArray[np.str_]]:
    """
    Returns aligned gene sets from query and reference gene indices.

    Arguments:
        genes: Tuple of (query_genes, reference_genes), both as Index objects.

    Returns:
        shared_genes: Genes present in both query and reference, sorted.
        reference_gene_order: Reference genes with shared genes first, followed by reference-only genes (both sorted).
    """
    query_genes, reference_genes = genes
    shared_genes = np.intersect1d(query_genes, reference_genes)
    reference_only_genes = np.setdiff1d(reference_genes, shared_genes)
    reference_gene_order = np.hstack([shared_genes, reference_only_genes])

    return shared_genes, reference_gene_order

def split_into_parts(
    arr: NDArray[Any], 
    n_parts: int
) -> Generator[tuple[NDArray[Any], NDArray[Any]], None, None]:
    """
    Yields each of n parts of a 1D array along with the concatenation of the remaining parts.
    """
    parts = np.array_split(arr, n_parts)
    for i in range(n_parts):
        yield parts[i], np.concatenate([part for j, part in enumerate(parts) if j != i])

def cross_feature_prediction(
    query_matrix: NDArray,
    reference_matrix: NDArray,
    predictor: Callable[[NDArray, NDArray], NDArray]
) -> NDArray:
    """
    Performs feature-wise cross-prediction using the given predictor function.

    Splits features into parts, holding out each part (compidx) while predicting it
    from the remaining features (subidx) using query and reference matrices.

    Assumes that features are aligned between query and reference matrices.

    Arguments:
        query_matrix: Array of shape (n_samples, n_features).
        reference_matrix: Array of shape (n_reference_samples, n_features).
        predictor: A function that takes (query_submatrix, reference_submatrix)
                   and returns predicted features.

    Returns:
        Array of shape (n_samples, n_features) with predicted values.
    """
    assert query_matrix.shape[1] == reference_matrix.shape[1]
    n_features = query_matrix.shape[1]

    predicted_chunks = []
    for compidx, subidx in split_into_parts(np.arange(n_features), 3):
        predicted_chunk = predictor(
            query_matrix[:, subidx],
            reference_matrix[:, np.hstack([subidx, compidx])]
        )
        predicted_chunks.append(predicted_chunk)

    return np.hstack(predicted_chunks)

def mean_randomized_cross_feature_prediction(
    query_matrix: NDArray,
    reference_matrix: NDArray,
    predictor: Callable[[NDArray, NDArray], NDArray],
    iterations: int,
    seed: Optional[int] = None,
) -> NDArray:
    """
    Applies feature-wise cross-prediction multiple times with random feature permutations,
    then returns the mean prediction.

    Arguments:
        query_matrix: Array of shape (n_samples, n_features), with query observations.
        reference_matrix: Array of shape (n_reference_samples, n_features), with reference data.
        predictor: A function that takes (query_submatrix, reference_submatrix) and returns predicted features.
        iterations: Number of random permutations

        Assumes that features are aligned between query and reference matrices.

    Returns:
        Array of shape (n_samples, n_features) containing the averaged predictions.
    """
    rng = np.random.default_rng(seed=seed)
    res = []
    for _ in range(iterations):
        feature_order: NDArray[np.int64] = rng.permutation(np.arange(query_matrix.shape[1]))
        prediction = cross_feature_prediction(
            query_matrix[:, feature_order],
            reference_matrix[:, feature_order],
            predictor
        )
        res.append(prediction[:, np.argsort(feature_order)])
    res = np.stack(res).mean(axis=0)
    return res

def group_by_common_obs(
    adata1: AnnData,
    adata2: AnnData,
    obs_column: str,
) -> Generator[tuple[Hashable, AnnData, AnnData], None, None]:
    """
    Yields (value, adata1_subset, adata2_subset) for each value shared in the given .obs column.

    Both AnnData objects are filtered to include only rows (cells) where the specified column
    has a shared value.

    Example:
        adata1.obs[obs_column] = [A, B, C]
        adata2.obs[obs_column] = [B, C, D]

    Yields:
        ("B", adata1[obs == "B"], adata2[obs == "B"])
        ("C", adata1[obs == "C"], adata2[obs == "C"])
    """
    common_values = np.intersect1d(
       adata1.obs[obs_column].unique(), 
       adata2.obs[obs_column].unique()
    )
    for value in common_values:
        adata1_filtered, adata2_filtered = map(
            lambda adata: adata[adata.obs[obs_column] == value].copy(),
            [adata1, adata2]
        )
        yield value, adata1_filtered, adata2_filtered

