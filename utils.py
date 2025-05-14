from typing import Any, Callable, Generator

import numpy as np
from numpy.typing import NDArray
from pandas import Index

def get_aligned_gene_sets(
    query_genes: Index,
    reference_genes: Index,
) -> tuple[NDArray[np.str_], NDArray[np.str_]]:
    """
    Identifies gene subsets from query and reference gene indices.

    Returns:
    - shared_genes: genes common to both query and reference, in sorted order.
    - reference_gene_order: genes in reference composed of shared genes
      followed by reference-specific genes (both sorted).
    """
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

    Parameters:
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
) -> NDArray:
    """
    Applies feature-wise cross-prediction multiple times with random feature permutations,
    then returns the mean prediction.

    Parameters:
        query_matrix: Array of shape (n_samples, n_features), with query observations.
        reference_matrix: Array of shape (n_reference_samples, n_features), with reference data.
        predictor: A function that takes (query_submatrix, reference_submatrix) and returns predicted features.
        iterations: Number of random permutations

        Assumes that features are aligned between query and reference matrices.

    Returns:
        Array of shape (n_samples, n_features) containing the averaged predictions.
    """
    rng = np.random.default_rng()
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