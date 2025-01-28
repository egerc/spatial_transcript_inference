from sklearn.decomposition._nmf import _BaseNMF

class fixed_H_NMF(_BaseNMF):
    def __init__(self, n_components: None | int = None, *, init=None, beta_loss: str = "frobenius", tol: float = 0.0001, max_iter: int = 200, random_state=None, alpha_W: float = 0, alpha_H: str = "same", l1_ratio: float = 0, verbose: int = 0) -> None:
        super().__init__(n_components, init=init, beta_loss=beta_loss, tol=tol, max_iter=max_iter, random_state=random_state, alpha_W=alpha_W, alpha_H=alpha_H, l1_ratio=l1_ratio, verbose=verbose)

class fixed_W_NMF(_BaseNMF):
    def __init__(self, n_components: None | int = None, *, init=None, beta_loss: str = "frobenius", tol: float = 0.0001, max_iter: int = 200, random_state=None, alpha_W: float = 0, alpha_H: str = "same", l1_ratio: float = 0, verbose: int = 0) -> None:
        super().__init__(n_components, init=init, beta_loss=beta_loss, tol=tol, max_iter=max_iter, random_state=random_state, alpha_W=alpha_W, alpha_H=alpha_H, l1_ratio=l1_ratio, verbose=verbose)