import torch
from torch.utils.data import DataLoader
import numpy as np


def compute_param_gradients(model, epoch_gradient_norms: dict) -> None:
    """Helper function to compute the norms of parameter gradients for each parameter in a given model, and update the dictionary storing them by name.

    Args:
        model (NeuralNetwork): The model whose parameters we wish to compute gradients for.
        epoch_gradient_norms (dict): A dictionary storing the norms for each parameter by name over the course of training.
    """

    # Record the norm for every weight and bias tensor.
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            grad_norm = parameter.grad.detach().norm().item()
            epoch_gradient_norms[name].append(grad_norm)


def train(optimizer:torch.optim, model, loss_fn, dataloader: DataLoader, device: torch.accelerator, Niter=10000, verbose=False) -> tuple[dict, torch.Tensor]:
    """Training loop for a given model, loss function, data, optimizer, and device. With the option to print the results every iteration (verbose=True).

    Args:
        optimizer (torch.optim): The pyTorch optimizer to use in minimizing the loss during training.
        model (_type_): The model to train. Example from the models module: models.nn.
        loss_fn (_type_): A loss function to minimize. Example from pyTorch: torch.nn.MSELoss().
        dataloader (DataLoader): The dataloader containing training data.
        device (torch.accelerator): The device on which computations are done.
        Niter (int, optional): The total number of epochs for training. Defaults to 10000.
        verbose (bool, optional): Whether to print results by epoch. Defaults to False.

    Returns:
        tuple[dict, torch.Tensor]: A dictionary of training history, containing loss values and parameter gradient norms for each epoch; and the trained model.
    """

    # Grab needed values from the dataloader
    data_stats = dataloader.dataset.compute_stats()
    batch_size = dataloader.batch_size
    length = len(dataloader.dataset)

    # Create objects to store the losses and gradient history over training (over all epochs)
    epoch_losses = []

    gradient_history = {
        name: []
        for name, parameter in model.named_parameters()
        if parameter.requires_grad}

    for iteration in torch.arange(0, Niter): 
        # For debugging, print results at every epoch 
        if verbose:
            print(f"Epoch {iteration+1}\n-------------------------------")

        # Create objects to store the losses and gradient history just over this batch of data
        batch_losses = []
        batch_param_gradients = []

        # Store each minibatch's gradient norm separately
        epoch_gradient_norms = {
            name: []
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        
        for batch, (X,y) in enumerate(dataloader):
            # Reset optimizer, send data to the GPU if using
            X.requires_grad_(True)
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)

            # Compute predictions and loss
            pred = model(X, torch.from_numpy(data_stats[0]), torch.from_numpy(data_stats[1]))
            loss = loss_fn(pred[0][:,0], y)

            # Backpropagation and compute gradients wrt model parameters
            loss.backward()
            compute_param_gradients(model, epoch_gradient_norms) 

            # Update model parameters
            optimizer.step()
            optimizer.zero_grad()
            
            # Update the tallies for the batch
            current = batch * batch_size + len(X)
            batch_losses.append(loss.item())
            
            # For debugging, print results at every epoch
            if verbose:
                print(f"loss: {loss.item():>7f}  [{current:>5d}/{length:>5d}]")

        # Update the tallies for the epoch
        epoch_losses.append(np.mean(batch_losses))

        for name in gradient_history:
            gradient_history[name].append(
                np.mean(epoch_gradient_norms[name])
                if epoch_gradient_norms[name]
                else float("nan"))

        # For debugging, print results at every epoch
        if verbose:
            print(f"-------------------------------\nEpoch {iteration+1} average loss: {epoch_losses[iteration]:>7f}")
        else:
            if (iteration+1) % 100 == 0:
                print(f"Epoch {iteration+1} average training loss: {epoch_losses[iteration]:>7f}")

    # Update the dictionary containing the training history
    history = {"epoch_losses": epoch_losses, "parameter_gradients": gradient_history}
    return history, model


def test(dataloader: DataLoader, model, loss_fn, device: torch.accelerator) -> None:
    """A function to test a trained model given new data.

    Args:
        dataloader (DataLoader): The dataloader containing new data not seen in training.
        model (_type_): The trained model to test.
        loss_fn (_type_): The loss function to use to evaluate model accuracy. Example from pyTorch: nn.L1Loss().
        device (torch.accelerator): The device on which computations should be done.
    """
    data_stats = dataloader.dataset.compute_stats()
    model.eval()

    X = dataloader.dataset.get_data()
    y = dataloader.dataset.get_y()

    with torch.no_grad():
        output = model(X, torch.from_numpy(data_stats[0]), torch.from_numpy(data_stats[1]))
        pred = torch.squeeze(output[0]) 
        test_loss = loss_fn(pred, y).item()
        print(f"Test error (MAE): {test_loss:>8f} \n")

    return pred, test_loss
