### Adapted NMF

# The following code contains an adapted version of sklearns NMF algorithm. 
# The changes include removing the necessity to create an instance of a sklearn.decomposition.NMF object, 
# removing the multiplicative update algorithm and condensing the original _fit_coordinate_descent function. 
# The changes allow to freeze the H matrix in the update algorithm for the purpose of generating this matrix based on single cell data 
# and generating read counts in imaging based spatial data.

import warnings

import numpy as np

from sklearn.decomposition._cdnmf_fast import _update_cdnmf_fast
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils import check_array, check_random_state
from sklearn.decomposition._nmf import _check_init, _initialize_nmf, _beta_divergence
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import validate_data

def _fit_coordinate_descent_adapted(
    X,
    W,
    H,
    tol=1e-4,
    max_iter=200,
    l1_reg_W=0,
    l1_reg_H=0,
    l2_reg_W=0,
    l2_reg_H=0,
    update_H = True,
):
    """Compute Non-negative Matrix Factorization (NMF) with Coordinate Descent

    The objective function is minimized with an alternating minimization of W
    and H. Each minimization is done with a cyclic (up to a permutation of the
    features) Coordinate Descent.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Constant matrix.

    W : array-like of shape (n_samples, n_components)
        Initial guess for the solution.

    H : array-like of shape (n_components, n_features)
        Initial guess for the solution.

    tol : float, default=1e-4
        Tolerance of the stopping condition.

    max_iter : int, default=200
        Maximum number of iterations before timing out.

    l1_reg_W : float, default=0.
        L1 regularization parameter for W.

    l1_reg_H : float, default=0.
        L1 regularization parameter for H.

    l2_reg_W : float, default=0.
        L2 regularization parameter for W.

    l2_reg_H : float, default=0.
        L2 regularization parameter for H.

    Returns
    -------
    W : ndarray of shape (n_samples, n_components)
        Solution to the non-negative least squares problem.

    H : ndarray of shape (n_components, n_features)
        Solution to the non-negative least squares problem.

    n_iter : int
        The number of iterations done by the algorithm.

    References
    ----------
    .. [1] :doi:`"Fast local algorithms for large scale nonnegative matrix and tensor
       factorizations" <10.1587/transfun.E92.A.708>`
       Cichocki, Andrzej, and P. H. A. N. Anh-Huy. IEICE transactions on fundamentals
       of electronics, communications and computer sciences 92.3: 708-721, 2009.
    """
    # so W and Ht are both in C order in memory
    Ht = check_array(H.T, order="C")
    X = check_array(X, accept_sparse="csr")

    _update_coordinate_descent = lambda X, W, Ht, l1_reg, l2_reg: _update_cdnmf_fast(
        W,
        np.dot(Ht.T, Ht) + (np.eye(Ht.shape[1], dtype=X.dtype) * (l2_reg)), 
        safe_sparse_dot(X, Ht) - l1_reg,
        np.arange(Ht.shape[1])
    )

    for n_iter in range(1, max_iter + 1):
        violation = 0.0
        # Update W
        violation += _update_coordinate_descent(X, W, Ht, l1_reg_W, l2_reg_W)
        # Update H
        if update_H:
            violation += _update_coordinate_descent(X.T, Ht, W, l1_reg_H, l2_reg_H)

        if n_iter == 1:
            violation_init = violation
        if violation_init == 0:
            break
        if violation / violation_init <= tol:
            break
    return W, Ht.T, n_iter


def _check_w_h(X, n_components, W, H, init=None, update_H=True):
    """Check W and H, or initialize them."""
    n_samples, n_features = X.shape

    if init == "custom" and update_H:
        _check_init(H, (n_components, X.shape[0]), "NMF (input H)")
        _check_init(W, (X.shape[1], n_components), "NMF (input W)")

        if H.dtype != X.dtype or W.dtype != X.dtype:
            raise TypeError(
                "H and W should have the same dtype as X. Got "
                "H.dtype = {} and W.dtype = {}.".format(H.dtype, W.dtype)
            )

    elif not update_H:
        if W is not None:
            warnings.warn(
                "When update_H=False, the provided initial W is not used.",
                RuntimeWarning,
            )

        _check_init(H, (n_components, X.shape[0]), "NMF (input H)")

        if H.dtype != X.dtype:
            raise TypeError(
                "H should have the same dtype as X. Got H.dtype = {}.".format(
                    H.dtype
                )
            )

        W = np.zeros((X.shape[1], n_components), dtype=X.dtype)

    else:
        if W is not None or H is not None:
            warnings.warn(
                (
                    "When init!='custom', provided W or H are ignored. Set "
                    " init='custom' to use them as initialization."
                ),
                RuntimeWarning,
            )

        W, H = _initialize_nmf(
            X, n_components, init=init
        )

    return W, H

def _fit_transform_adapted(X, n_components, y=None, W=None, H=None, update_H=True, alpha_W=0.0, alpha_H=0.0, l1_ratio=0.0, tol=1e-4, max_iter=200):
        """Learn a NMF model for the data X and returns the transformed data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data matrix to be decomposed

        y : Ignored

        W : array-like of shape (n_samples, n_components), default=None
            If `init='custom'`, it is used as initial guess for the solution.
            If `update_H=False`, it is initialised as an array of zeros, unless
            `solver='mu'`, then it is filled with values calculated by
            `np.sqrt(X.mean() / self._n_components)`.
            If `None`, uses the initialisation method specified in `init`.

        H : array-like of shape (n_components, n_features), default=None
            If `init='custom'`, it is used as initial guess for the solution.
            If `update_H=False`, it is used as a constant, to solve for W only.
            If `None`, uses the initialisation method specified in `init`.

        update_H : bool, default=True
            If True, both W and H will be estimated from initial guesses,
            this corresponds to a call to the 'fit_transform' method.
            If False, only W will be estimated, this corresponds to a call
            to the 'transform' method.

        Returns
        -------
        W : ndarray of shape (n_samples, n_components)
            Transformed data.

        H : ndarray of shape (n_components, n_features)
            Factorization matrix, sometimes called 'dictionary'.

        n_iter_ : int
            Actual number of iterations.
        """
        # check parameters
        #self._check_params(X)

        #if X.min() == 0 and self._beta_loss <= 0:
        #    raise ValueError(
        #        "When beta_loss <= 0 and X contains zeros, "
        #        "the solver may diverge. Please add small values "
        #        "to X, or use a positive beta_loss."
        #    )

        # initialize or check W and H
        W, H = _check_w_h(X=X, n_components=n_components, W=W, H=H, update_H=update_H)

        W, H, n_iter = _fit_coordinate_descent_adapted(
            X,
            W,
            H,
            l1_reg_W=X.shape[0] * alpha_W * l1_ratio,
            l1_reg_H=X.shape[1] * alpha_H * l1_ratio,
            l2_reg_W=X.shape[0] * alpha_W * (1.0 - l1_ratio),
            l2_reg_H=X.shape[1] * alpha_H * (1.0 - l1_ratio),
            
        )
        if n_iter == max_iter and tol > 0:
            warnings.warn(
                "Maximum number of iterations %d reached. Increase "
                "it to improve convergence." % max_iter,
                ConvergenceWarning,
            )

        return W, H, n_iter