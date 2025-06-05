from dataclasses import dataclass
import numpy as np
from functools import partial
from typing import Callable, Optional, Tuple
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import cosine
from numpy.typing import NDArray

# because our count matrix is cells by genes, mean over axis=0 means mean over cells -> return value is the genes, and the reverse for axis=1
mean_genes = partial(np.mean, axis=0)
mean_cells = partial(np.mean, axis=1)
log_transform = lambda arr: np.log1p(arr + 1e-3)


def _transform_inputs(
    metric: Callable[[NDArray, NDArray], float], 
    transform: Callable[[NDArray], NDArray]
) -> Callable[[NDArray, NDArray], float]:
    return lambda arr1, arr2: metric(transform(np.nan_to_num(arr1)), transform(np.nan_to_num(arr2)))


def _mse_metric(arr1: NDArray, arr2: NDArray) -> float:
    return float(mean_squared_error(arr1, arr2))


def _pearsonr_metric(arr1: NDArray, arr2: NDArray) -> float:
    return float(pearsonr(arr1, arr2)[0])


def _spearmanr_metric(arr1: NDArray, arr2: NDArray) -> float:
    return float(spearmanr(arr1, arr2)[0])


def _cosine_metric(arr1: NDArray, arr2: NDArray) -> float:
    return 1.0 - float(cosine(arr1, arr2))


# MSE doesn't require mean transform since it operates on full 2D arrays
mse_fn = _mse_metric
mse_log_fn = _transform_inputs(_mse_metric, log_transform)

# Pearson
pearson_fn_genes = _transform_inputs(_pearsonr_metric, mean_genes)
pearson_fn_cells = _transform_inputs(_pearsonr_metric, mean_cells)
pearson_log_fn_genes = _transform_inputs(_pearsonr_metric, lambda arr: mean_genes(log_transform(arr)))
pearson_log_fn_cells = _transform_inputs(_pearsonr_metric, lambda arr: mean_cells(log_transform(arr)))

# Spearman (no log variant needed)
spearman_fn_genes = _transform_inputs(_spearmanr_metric, mean_genes)
spearman_fn_cells = _transform_inputs(_spearmanr_metric, mean_cells)

# Cosine
cosine_fn_genes = _transform_inputs(_cosine_metric, mean_genes)
cosine_fn_cells = _transform_inputs(_cosine_metric, mean_cells)
cosine_log_fn_genes = _transform_inputs(_cosine_metric, lambda arr: mean_genes(log_transform(arr)))
cosine_log_fn_cells = _transform_inputs(_cosine_metric, lambda arr: mean_cells(log_transform(arr)))

@dataclass
class Metric:
    func: Callable[[np.ndarray, np.ndarray], float]
    plot_range: Optional[Tuple[float, float]] # for sensible plotting 

metric_registry: dict[str, Metric] = {
    "mse": Metric(func=mse_fn, plot_range=None),
    "pearson_genes": Metric(func=pearson_fn_genes, plot_range=(0, 1)),
    "pearson_log_genes": Metric(func=pearson_log_fn_genes, plot_range=(0, 1)),
    "spearman_genes": Metric(func=spearman_fn_genes, plot_range=(0, 1)),
    "cosine_genes": Metric(func=cosine_fn_genes, plot_range=(0, 1)),
    "cosine_log_genes": Metric(func=cosine_log_fn_genes, plot_range=(0, 1)),
}