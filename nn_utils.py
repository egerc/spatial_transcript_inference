from typing import Any
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
        super().__init__()
        self.input_tensor = input_tensor
        self.target_tensor = target_tensor

        n_input = input_tensor.shape[1]
        n_output = target_tensor.shape[1]

        self.fc1 = nn.Linear(n_input, n_latent)
        self.fc2 = nn.Linear(n_latent, n_output)

    def forward(self, x) -> Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def converged(self, losses: list[float], window: int = 10) -> bool:
        """
        Check convergence based on the slope of the loss curve over the last `window` epochs.
        
        Returns True if slope ≥ 0, indicating no further improvement.
        """
        if len(losses) < window:
            return False
        y = losses[-window:]
        x = list(range(len(y)))
        slope, _, _, _, _ = linregress(x, y)
        return slope >= 0

    def train(
        self,
        optimizer,
        lr,
        loss_fn,
        max_n_epochs
    ) -> list[Any]:
        optimizer = optimizer(self.parameters(), lr=lr)
        losses = []
        for _ in range(max_n_epochs):
            output = self(self.input_tensor)
            loss = loss_fn(output, self.target_tensor)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if self.converged(losses, window=10):
                print("convergence reached")
                break
        return losses

def nn_predictor(
    query_matrix: NDArray,
    reference_matrix: NDArray
) -> NDArray:
    n_shared_genes = query_matrix.shape[1]
    input_tensor, target_tensor = (Tensor(arr) for arr in np.hsplit(reference_matrix, [n_shared_genes]))
    model = Model(input_tensor, 10, target_tensor)
    model.train(
        torch.optim.Adam,
        1e-3,
        torch.nn.MSELoss(),
        1000
    )
    query_tensor: Tensor = Tensor(query_matrix)
    predicted_counts = model(query_tensor)
    return predicted_counts.detach().numpy()