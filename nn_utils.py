from typing import Any, Callable, Literal

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import linregress
from sklearn.neighbors import NearestNeighbors

def intrinsic_dimensionality_mle(X: np.ndarray, k: int = 10) -> float:
    """
    Estimate intrinsic dimensionality of data X using Levina-Bickel MLE method.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
        k: Number of nearest neighbors to use (usually 5 <= k <= 20)
    
    Returns:
        Estimated intrinsic dimensionality (float)
    """
    n_samples = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    distances, _ = nbrs.kneighbors(X)
    # distances[:, 0] is zero (distance to self), ignore it
    distances = distances[:, 1:]  # shape (n_samples, k)

    # Compute MLE estimator for each point
    logs = np.log(distances[:, -1][:, None] / distances[:, :-1])
    inv_dims = (1 / (k - 1)) * np.sum(logs, axis=1)
    dims = 1 / inv_dims
    # Return the average intrinsic dimension
    return np.mean(dims)

def converged(
    losses: list[float], 
    window: int = 10, 
    cutoff: float = 0
) -> bool:
    """
    Checks if training has converged based on recent loss slope.

    Args:
        losses: List of recorded loss values.
        window: Number of epochs to consider for slope.
        cutoff: Minimum slope to consider as not converged.

    Returns:
        True if slope ≥ cutoff, False otherwise.
    """
    if len(losses) < window:
        return False
    y = losses[-window:]
    x = list(range(len(y)))
    slope, _, _, _, _ = linregress(x, y)
    return slope >= cutoff

class Model(nn.Module):
    def __init__(
        self,
        input_tensor: Tensor,
        n_latent: int,
        target_tensor: Tensor
    ) -> None:
        """
        Initializes a simple 2-layer MLP.

        Args:
            input_tensor: Input tensor for training (n_samples × n_input).
            n_latent: Number of latent units in the hidden layer.
            target_tensor: Target tensor for training (n_samples × n_output).
        """
        super().__init__()
        self.input_tensor = input_tensor
        self.target_tensor = target_tensor

        n_input = input_tensor.shape[1]
        n_output = target_tensor.shape[1]

        self.fc1 = nn.Linear(n_input, n_latent)
        self.fc2 = nn.Linear(n_latent, n_output)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after forward pass.
        """
        x = F.sigmoid(self.fc1(x))
        return self.fc2(x)


    def train(
        self,
        optimizer: Callable[..., torch.optim.Optimizer],
        lr: float,
        loss_fn: Any,
        max_n_epochs: int,
        verbose: bool = False,
        convergence_cutoff: float = 0

    ) -> list[float]:
        """
        Trains the model until convergence or max epochs.

        Args:
            optimizer: Optimizer class (e.g. torch.optim.Adam).
            lr: Learning rate.
            loss_fn: Loss function.
            max_n_epochs: Maximum number of training epochs.
            verbose: Whether to print final epoch info.

        Returns:
            List of loss values per epoch.
        """
        optimizer = optimizer(self.parameters(), lr=lr)
        losses = []
        window = 10
        for epoch in range(max_n_epochs):
            output = self(self.input_tensor)
            loss = loss_fn(output, self.target_tensor)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if epoch % window == 0 and converged(losses, window=window, cutoff=convergence_cutoff):
                print(f"convergence reached, training stopped at {epoch}")
                break
        if verbose:
            print(f"training stopped at {max_n_epochs=}")
        return losses


def nn_predictor(
    query_matrix: NDArray,
    reference_matrix: NDArray,
    *args,
    **kwargs,
) -> NDArray:
    """
    Predicts target values from a reference using a trained neural network.

    Args:
        query_matrix: Matrix to predict from (n_samples × n_features).
        reference_matrix: Matrix used to train the model 
                          (n_samples × (n_input + n_output)).
        *args, **kwargs: passed to training function

    Returns:
        Predicted values as a NumPy array.
    """
    n_shared_genes = query_matrix.shape[1]
    input_tensor, target_tensor = (Tensor(arr) for arr in np.hsplit(reference_matrix, [n_shared_genes]))
    #n_latent = int(intrinsic_dimensionality_mle(reference_matrix))
    model = Model(input_tensor, 5, target_tensor)
    model.train(
        torch.optim.AdamW,
        0.1,
        torch.nn.MSELoss(),
        10000,
        convergence_cutoff=-0.001,
        *args,
        **kwargs,
    )
    query_tensor: Tensor = Tensor(query_matrix)
    predicted_counts = model(query_tensor)
    return predicted_counts.detach().numpy()

def reparameterize(
    mu: torch.Tensor,
    std: torch.Tensor
):
    return mu

class SampledNormalLayer(nn.Module):
    def __init__(
        self, 
        layer_mu: nn.Module, 
        layer_std: nn.Module, 
    ):
        super().__init__()
        self.layer_mu = layer_mu
        self.layer_std = layer_std

    def forward(
        self,
        x,
    ):
        mu = self.layer_mu(x)
        std = self.layer_std(x)
        res = reparameterize(mu, std)
        return res

class VAE(nn.Module):
    def __init__(
        self,
        input_tensor: torch.Tensor,
        n_latent: int,
        target_tensor: torch.Tensor
    ) -> None:
        super().__init__()
        self.input_tensor = input_tensor
        self.target_tensor = target_tensor

        n_input = input_tensor.shape[1]
        n_output = target_tensor.shape[1]

        self.mu = nn.Linear(n_input, n_latent)
        self.std = nn.Linear(n_input, n_latent)
        self.sampling_layer = SampledNormalLayer(
            self.mu,
            self.std
        )
        self.fc2 = nn.Linear(n_latent, n_output)

    def forward(
        self, 
        x: torch.Tensor, 
        mode: Literal["eval", "train"] = "eval"
    ) -> torch.Tensor:
        #x = F.relu(self.fc1(x))
        x = {
            "eval": self.mu,
            "train": self.sampling_layer,
        }[mode](x)
        x = F.relu(x)

        return self.fc2(x)

    def train(
        self,
        optimizer: Callable[..., torch.optim.Optimizer],
        lr: float,
        loss_fn: Any,
        max_n_epochs: int,
        verbose: bool = False
    ) -> list[float]:
        optimizer = optimizer(self.parameters(), lr=lr)
        losses = []
        window = 10
        for epoch in range(max_n_epochs):
            output = self(self.input_tensor, mode="train")
            loss = loss_fn(output, self.target_tensor)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if epoch % window == 0 and converged(losses, window=window, cutoff=5e-1):
                print(f"convergence reached, training stopped at {epoch}")
                break
        if verbose:
            print(f"training stopped at {max_n_epochs=}")
        return losses
    
def vae_predictor(
    query_matrix: NDArray,
    reference_matrix: NDArray,
    *args,
    **kwargs,
) -> NDArray:

    n_shared_genes = query_matrix.shape[1]
    input_tensor, target_tensor = (torch.Tensor(arr) for arr in np.hsplit(reference_matrix, [n_shared_genes]))
    model = VAE(input_tensor, 10, target_tensor)
    model.train(
        torch.optim.Adam,
        1e-1,
        torch.nn.MSELoss(),
        1000,
        *args,
        **kwargs,
    )
    query_tensor  = torch.Tensor(query_matrix)
    predicted_counts = model(query_tensor)
    return predicted_counts.detach().numpy()