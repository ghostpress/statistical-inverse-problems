import numpy as np
import torch
from torch import nn

###############################################
# Model 1: Approximate u with feed-forward NN #
###############################################

class NeuralNetwork(nn.Module):
    def __init__(self, in_dim: int, layers: list[int], out_dim=1) -> None:
        """Initialize the neural network according to arguments defining the architecture.

        Args:
            in_dim (int): The spatial dimension of the problem, which determines the dimension of the first layer of the NN.
            layers (list[int]): The number of hidden layers in the NN presented in list format, e.g. [2, 64, 128, 128, 64].
            out_dim (int, optional): The dimensions of the output. Defaults to 1 for real-valued functions.
        """

        super().__init__()

        network = []
        architecture = [in_dim + 1] + layers + [out_dim]
        for layer_i, layer_j in zip(architecture, architecture[1:]):
            network.append(nn.Linear(layer_i, layer_j))
            network.append(nn.Tanh())
        network.pop()

        self.network = nn.Sequential(*network)

    def forward(self, input: torch.Tensor, mean: torch.Tensor, sd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform a forward pass of the network. 

        Args:
            input (torch.Tensor): Input Tensor with dimensions (N, n_dim).
            mean (torch.Tensor): The mean of the full dataset.
            sd (torch.Tensor): The standard deviation of the full dataset.

        Returns:
            (torch.Tensor, torch.Tensor): The output of the forward pass, and the coordinates (with gradients preserved).
        """
        standard_inputs = self.standardize_data(mean, sd, input)
        out = self.network(standard_inputs)
        coordinates = standard_inputs.clone().requires_grad_(True)  # FIXME: why output coordinates?
        return out, coordinates

    def standardize_data(self, mean: torch.Tensor, sd: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        """Extra 'layer' of the network that standardizes input data. Implemented here so that backpropagation preserves gradients with respect 
        to the original inputs.

        Args:
            mean (torch.Tensor): The mean of the full dataset.
            sd (torch.Tensor): The standard deviation of the full dataset.
            input (torch.Tensor): An input tensor to standardize, assumed to be a minibatch of the full dataset. 

        Returns:
            torch.Tensor: A standardized tensor with mean 0 and standard deviation 1 across all components. 
        """
        standard = (input.clone() - mean)/sd
        return standard


# PDE-READ architecture notes:
# - feed-forward network w/ 5 hidden layers, 50 units/layer

# Deep-MoD architecture notes:
# - feed-forward network w/ 5 hidden layers, 20 units/layer
# - Adam optimizer
    