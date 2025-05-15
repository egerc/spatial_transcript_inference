import numpy as np
import tangram as tg
from numpy.typing import NDArray
from anndata import AnnData

def tangram_predictor(
    query_matrix: NDArray,
    reference_matrix: NDArray
) -> NDArray:
    """
    Predicts feature counts for a query matrix based on the Tangram gene projection from a reference dataset.

    Parameters:
        query_matrix: numpy array containing the query dataset's counts of the shared features between query and reference dataset
        reference_matrix: numpy array containing the shared features between query and reference as well as the features to be predicted.
        It is assumed that the ordering of the shared features is identical and that the to be predicted features come "after" the shared features.
        *args, **kwargs: additional arguments provided to the Tangram preprocessing or mapping steps (if applicable).

    Returns:
        Array containing the predicted features for the query observations
    """
    adata_query, adata_reference = map(
        AnnData,
        (query_matrix, reference_matrix)
    )
    tg.pp_adatas(adata_reference, adata_query)
    ad_map = tg.map_cells_to_space(
        adata_sc=adata_reference,
        adata_sp=adata_query,
    )
    ad_ge = tg.project_genes(ad_map, adata_reference)
    assert isinstance(ad_ge.X, np.ndarray)
    predicted_features = ad_ge.X[:, adata_query.n_vars:]
    return predicted_features