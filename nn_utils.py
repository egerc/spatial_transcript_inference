from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import linregress


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
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def converged(self, losses: list[float], window: int = 10, cutoff: float = 0) -> bool:
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

    def train(
        self,
        optimizer: Callable[..., torch.optim.Optimizer],
        lr: float,
        loss_fn: Any,
        max_n_epochs: int,
        verbose: bool = False
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
            if epoch % window == 0 and self.converged(losses, window=window, cutoff=5e-1):
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

    Returns:
        Predicted values as a NumPy array.
    """
    n_shared_genes = query_matrix.shape[1]
    input_tensor, target_tensor = (Tensor(arr) for arr in np.hsplit(reference_matrix, [n_shared_genes]))
    model = Model(input_tensor, 10, target_tensor)
    model.train(
        torch.optim.Adam,
        1e-1,
        torch.nn.MSELoss(),
        1000,
        *args,
        **kwargs,
    )
    query_tensor: Tensor = Tensor(query_matrix)
    predicted_counts = model(query_tensor)
    return predicted_counts.detach().numpy()