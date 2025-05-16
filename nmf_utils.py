from typing import Any, Callable, Optional

from sklearn.decomposition import non_negative_factorization
import numpy as np
from numpy.typing import NDArray

def nmf_predictor(
    query_matrix: NDArray,
    reference_matrix: NDArray,
    nmf_func: Callable[[NDArray], tuple[NDArray, NDArray, int]],
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

    _, H_ref, _ = nmf_func(
        reference_matrix,
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