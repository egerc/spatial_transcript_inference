from typing import Any, Callable, Literal

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import linregress
from sklearn.neighbors import NearestNeighbors


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
        self.bn1 = nn.BatchNorm1d(n_latent)
        self.fc2 = nn.Linear(n_latent, n_output)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after forward pass.
        """
        x = F.relu(self.bn1(self.fc1(x)))
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
            output = self(torch.log1p(self.input_tensor))
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
    model = Model(input_tensor, 6, target_tensor)
    model.train(
        torch.optim.AdamW,
        0.01,
        torch.nn.HuberLoss(),
        10000,
        convergence_cutoff=-0.001,
        *args,
        **kwargs,
    )
    query_tensor: Tensor = Tensor(query_matrix)
    predicted_counts = model(torch.log1p(query_tensor))
    return predicted_counts.detach().numpy()

def reparameterize(mu: torch.Tensor, std: torch.Tensor):
    eps = torch.randn_like(std)
    x = mu + std * eps
    return x, mu, std

def kl_divergence(mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(1 + torch.log(std**2 + 1e-8) - mu**2 - std**2) / mu.size(0)

class SampledNormalLayer(nn.Module):
    def __init__(self, layer_mu: nn.Module, layer_std: nn.Module):
        super().__init__()
        self.layer_mu = layer_mu
        self.layer_std = layer_std

    def forward(self, x):
        mu = self.layer_mu(x)
        std = torch.exp(0.5 * self.layer_std(x)) 
        z, mu, std = reparameterize(mu, std)
        return z, mu, std

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
        self.bn = nn.BatchNorm1d(n_latent)
        self.fc2 = nn.Linear(n_latent, n_output)

    def forward(
        self, 
        x: torch.Tensor, 
        mode: Literal["eval", "train"] = "eval"
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if mode == "eval":
            x = self.mu(x)
            x = self.bn(x)
            x = F.relu(x)
            return self.fc2(x)
        else:
            z, mu, std = self.sampling_layer(x)
            z = self.bn(z)
            z = F.relu(z)
            return self.fc2(z), mu, std

    def train(
        self,
        optimizer: Callable[..., torch.optim.Optimizer],
        lr: float,
        loss_fn: Any,
        max_n_epochs: int,
        verbose: bool = False,
        convergence_cutoff: float = 0
    ) -> list[float]:
        optimizer = optimizer(self.parameters(), lr=lr)
        losses = []
        window = 10
        for epoch in range(max_n_epochs):
            output, mu, std = self(torch.log1p(self.input_tensor), mode="train")
            recon_loss = loss_fn(output, self.target_tensor)
            kl_loss = kl_divergence(mu, std)
            loss = recon_loss + kl_loss
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
    
def vae_predictor(
    query_matrix: NDArray,
    reference_matrix: NDArray,
    *args,
    **kwargs,
) -> NDArray:

    n_shared_genes = query_matrix.shape[1]
    input_tensor, target_tensor = (torch.Tensor(arr) for arr in np.hsplit(reference_matrix, [n_shared_genes]))
    model = VAE(input_tensor, 6, target_tensor)
    model.train(
        torch.optim.AdamW,
        0.01,
        torch.nn.HuberLoss(),
        10000,
        convergence_cutoff=-0.001,
        *args,
        **kwargs,
    )
    query_tensor  = torch.Tensor(query_matrix)
    predicted_counts = model(torch.log1p(query_tensor))
    return predicted_counts.detach().numpy()