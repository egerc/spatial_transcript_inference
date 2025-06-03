from typing import Any, Callable, Iterable, Literal, Optional, TypeVar

from anndata import AnnData
from sklearn.decomposition import non_negative_factorization
from sklearn.metrics import mean_squared_error
from kneed import KneeLocator
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import scanpy as sc

def non_negative_factorization_reimplementation(
        X: np.ndarray, 
        W: np.ndarray | None = None, 
        H: None | np.ndarray=None, 
        n_components: int | None = None, 
        init: Literal["custom"] | None =None,
        update_H: bool=True,
        max_iter: int=200
) -> tuple[np.ndarray, np.ndarray, int]:

    if n_components is None:
        if H is not None:
            n_components = H.shape[0]
        else:
            n_components = min(X.shape)
    if W is None and init != "custom":
        W = np.ones((X.shape[0], n_components))
    if H is None and init != "custom":
        H = np.ones((n_components, X.shape[1]))
    
    for _ in tqdm(range(max_iter)):
        W = (W.T * (H @ X.T) / (H @ H.T @ W.T + 1e-9)).T
        if update_H:
            H = (H * (W.T @ X) / (W.T @ W @ H + 1e-9))
    return W, H, max_iter

def select_optimal_components(
    X: NDArray[Any],
    n_components_range: Iterable[int],
    *args: Any,
    **kwargs: Any
) -> tuple[Optional[Any], list[float]]:
    """
    Select the optimal number of NMF components by computing reconstruction error (MSE)
    for each candidate, then finding the 'knee' point on the error curve.

    Args:
        X (NDArray): Input data matrix to factorize (non-negative values).
        n_components_range (list[int]): List of candidate numbers of components to test.
        *args, **kwargs: Additional arguments passed to `KneeLocator`.

    Returns:
        Optional[int]: The number of components at the knee of the MSE curve, or None if no knee is found.
    """

    def locate_knee(
        x: Iterable[int],
        y: Iterable[float],
    ) -> Optional[Any]:
        """Identify the knee point on a curve."""
        knee: Optional[Any] = KneeLocator(x, y, *args, **kwargs).knee
        return knee

    def compute_mse_for_n(n: int) -> float:
        """Compute the reconstruction MSE for a given number of components."""
        W, H, _ = non_negative_factorization(X, n_components=n)
        X_hat: NDArray[Any] = W @ H
        return float(mean_squared_error(X, X_hat))

    MSEs: list[float] = [compute_mse_for_n(n) for n in n_components_range]
    knee: Optional[int] = locate_knee(n_components_range, MSEs)
    return knee, MSEs

def initialize_nmf_factors_from_clustering(
    arr: NDArray, 
    n_clusters: int
) -> tuple[NDArray, NDArray]:
    """
    Initialize W and H matrices for NMF using cluster-specific ranked gene scores.

    This function clusters the input `AnnData` object using Leiden clustering,
    finds the resolution that yields the desired number of clusters, ranks genes 
    per cluster, and constructs:
    - W: A matrix of gene scores (non-negative)
    - H: One-hot encoded cluster membership per cell, with small non-zero entries
    
    Parameters:
        arr: The input array
        n_clusters (int): Desired number of clusters for initialization.

    Returns:
        Tuple[np.ndarray, np.ndarray]: W and H matrices for NMF initialization.
    """
    adata = AnnData(arr)
    
    T = TypeVar("T")

    def find_input_for_output(
        f: Callable[[float], T],
        g: Callable[[T], int],
        target: int,
        max_iter: int = 500
    ) -> T:
        """Binary-like recursive search for an input that makes g(f(x)) == target."""

        def search(x: float, lower: Optional[float], upper: Optional[float], iter: int) -> T:
            t: T = f(x)
            value: int = g(t)

            if iter >= max_iter:
                print(f"max_iter exceeded, closest value found: {value}")
                return t
                #raise RuntimeError("Maximum search iterations reached without convergence.")
            if value == target:
                return t

            match value > target, lower, upper:
                case (True, None, _):
                    return search(x / 2, lower, x, iter + 1)
                case (True, _, _):
                    assert lower is not None
                    return search((lower + x) / 2, lower, x, iter + 1)
                case (False, _, None):
                    return search(x * 2, x, upper, iter + 1)
                case (False, _, _):
                    assert upper is not None
                    return search((x + upper) / 2, x, upper, iter + 1)

        return search(1.0, None, None, 0)

    adata_copy: AnnData = adata.copy()
    sc.pp.normalize_total(adata_copy)
    sc.pp.log1p(adata_copy)
    sc.tl.pca(adata_copy)
    sc.pp.neighbors(adata_copy)
    current_target = n_clusters

    while current_target >= 1:
        try:
            result: AnnData = find_input_for_output(
                f=lambda x: sc.tl.leiden(adata_copy, resolution=x, copy=True),  # type: ignore
                g=lambda adata: adata.obs["leiden"].nunique(),
                target=current_target
            )

            sc.tl.rank_genes_groups(result, 'leiden')
            break 

        except ValueError as e:
            print(f"ValueError at {current_target} clusters: {e}")
            current_target -= 1

    else:
        raise RuntimeError("Could not complete clustering and gene ranking with any cluster count.")


    W = pd.get_dummies(result.obs["leiden"], dtype="float").replace(0, 0.1).values.astype(arr.dtype)
    H: NDArray[Any] = (lambda arr: np.where(arr < 0, 0.1, arr))(
        pd.concat(
            [sc.get.rank_genes_groups_df(result, index).set_index("names")["scores"] for index in list(result.uns["rank_genes_groups"]["names"].dtype.names)], 
            axis=1
        ).reindex(result.var.index).values
    ).T.astype(arr.dtype)
    return W.copy(order="C"), H.copy(order="C")

def nmf_predictor(
    query_matrix: NDArray,
    reference_matrix: NDArray,
    *args,
    **kwargs
) -> NDArray:
    """
    Predicts feature counts for a query matrix based on the NMF feature by factor embedding in a reference dataset

    Parameters:
        query_matrix: numpy array containing the query datasets counts of the shared features between query and reference dataset
        reference_matrix: numpy array containing the shared features between query and reference as well as the features to be predicted.
        It is assumed that the ordering of the shared features is identical and that the to be predicted features come "after" the shared features.
        n_components: the number of components used for nmf decomposition
        *args, **kwargs: additional arguments provided to the two non_negative_factorization calls, for example to pass a different max_iter in case of ConvergenceWarning.
    Returns:
        Array containing the predicted features for the query observations
    """
    n_shared_features = query_matrix.shape[1]

    n_components, losses = select_optimal_components(
        reference_matrix,
        range(2, 12),
        curve="convex",
        direction="decreasing"
    )
    assert n_components
    n_components = int(n_components)

    W_init, H_init = initialize_nmf_factors_from_clustering(
        reference_matrix,
        n_clusters=n_components
    )

    _, H_ref, _ = non_negative_factorization(
        reference_matrix,
        W=W_init,
        H=H_init,
        init="custom"
    )

    H_query, H_predicted = np.hsplit(H_ref, [n_shared_features])

    W_query, _, _ = non_negative_factorization(
        X=query_matrix,
        H=H_query,
        init="custom",
        update_H=False,
        *args,
        **kwargs
    )
    predicted_features = W_query @ H_predicted
    return predicted_features