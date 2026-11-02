import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_training_loss(losses: torch.Tensor, Niter: int, destination: str, model_name: str) -> None:
    """Function to plot training losses over a given number of epochs. Saves the plot in .png format to a file in a
    given destination folder.

    Args:
        losses (torch.Tensor): The training loss, either as a torch.Tensor or a numpy.ndarray.
        Niter (int): The total number of epochs during which training occured. 
        destination (str): The desination of the saved image.
        model_name (str): The name of the model trained, for tracking in the filename.
    """
    # Convert pyTorch tensors to numpy arrays if necessary
    if isinstance(losses, torch.Tensor):
        losses = losses.detach().cpu().numpy()

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, Niter + 1), losses) 
    plt.xlabel("epoch")
    plt.ylabel("Average loss (MSE)")
    plt.title("Training loss by epoch")
    plt.grid(True)

    # Save the plot
    fname = f"{destination}/{model_name}_trainingLoss_E={Niter}.png" 
    plt.savefig(fname)


def plot_parameter_gradients(gradients: dict, Niter: int, destination: str, model_name: str) -> None:
    """Function to plot the norm of parameter gradients over the course of training. Saves the plots in .png format to a file in a
    given destination folder.

    Args:
        gradients (dict): A dictionary of the training history of the gradient norms for each parameter.
        Niter (int): The total number of epochs during which training occured. 
        destination (str): The desination of the saved image.
        model_name (str): The name of the model trained, for tracking in the filename.
    """
    # Convert pyTorch tensors to numpy arrays if necessary
    if isinstance(gradients, torch.Tensor):
        gradients = gradients.detach().cpu().numpy()

    for param, grads in gradients.items():
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, Niter + 1), grads) 
        plt.xlabel("epoch")
        plt.ylabel("Average norm")
        plt.title(f"Norm of gradients for parameter {param} over training")
        plt.grid(True)

        # Save the plot
        fname = f"{destination}/{model_name}_gradient_norm_{param}_E={Niter}.png" 
        plt.savefig(fname)
        
# def plot_predictions(X: torch.Tensor, y: torch.Tensor, pred: torch.Tensor, destination: str, model_name: str, all_steps=False, num_steps=5) -> None:
#     """A function to plot the predicted function value against the true value, for each input in the tensor X. Saves the plots in .png format 
#     to a file in a given destination folder.

#     Args:
#         X (torch.Tensor): The input tensor.
#         y (torch.Tensor): The target function values for each input.
#         pred (torch.Tensor): The predicted function values for each input.
#         destination (str): The desination of the saved image.
#         model_name (str): The name of the model trained, for tracking in the filename.
#         all_steps (bool, optional): Whether to plot the predictions for each timestep. Defaults to False.
#         num_steps (int, optional): If not all_steps, how many steps to plot. Defaults to 5.
#     """

#     print(y.shape)

#     # Convert pyTorch tensors to numpy arrays if necessary
#     if isinstance(X, torch.Tensor):
#         X = X.detach().cpu().numpy()
#         X = np.asarray(X)
#     if isinstance(y, torch.Tensor):
#         y = y.detach().cpu().numpy()
#         y = np.asarray(y).reshape(-1)
#     if isinstance(pred, torch.Tensor):
#         pred = pred.detach().cpu().numpy()
#         pred = np.asarray(pred).reshape(-1)

#     times = np.unique(X[:, 0])

#     if all_steps:
#         for step, t in enumerate(times):  
#             mask = X[:,0] == t
#             x_t = X[mask,1]                              
#             y_t = y[mask]
#             pred_t = pred[mask]

#             plt.figure(figsize=(10, 5))
#             plt.plot(x_t, y_t, label='true u')          
#             plt.plot(x_t, pred_t, label='predicted u')  
#             plt.xlabel("x")
#             plt.ylabel("u(t,x)")
#             plt.title(f"Model {model_name} predictions at time t={t}")
#             plt.grid(True)

#             # Save the plot
#             fname = f"{destination}/{model_name}_predictions_t={t}.png" 
#             plt.savefig(fname)
#     else:
#         idx = np.round(np.linspace(0, len(times) - 1, num_steps)).astype(int)
#         steps_to_plot = times[idx]

#         # Create subplots
#         fig, axes = plt.subplots(num_steps, 1, figsize=(10, 3*num_steps), sharex=True, squeeze=False)
#         for i, (ax, t) in enumerate(zip(axes[:,0], steps_to_plot)):
#             mask = X[:,0] == t
#             # print(mask.shape)
#             # print(y.shape)
#             # print(pred.shape)
#             x_t = X[mask, 1]
#             y_t = y[mask]
#             pred_t = pred[mask]

#             ax.plot(x_t, y_t, label='true u')
#             ax.plot(x_t, pred_t, label='predicted u')
#             ax.set_title(f'Model {model_name} predictions at time t={t}')
#             ax.set_xlabel('x')
#             ax.set_ylabel('u(t,x)')
#             ax.grid(True)
        
#         plt.tight_layout()
#         plt.show()

    # # Randomly sample observations
    # rng = np.random.default_rng(seed)
    # n_samples = min(n_samples, len(y))
    # idx = rng.choice(len(y), size=n_samples, replace=False)

    # # Create subplots
    # fig, axes = plt.subplots(
    #     n_samples, 1,
    #     figsize=(10, 3 * n_samples),
    #     squeeze=False
    # )

    # for i, ax in enumerate(axes[:, 0]):
    #     j = idx[i]

    #     time = X[j, 0]
    #     space = X[j, 1]

    #     ax.bar(
    #         ["True value", "Prediction"],
    #         [y[j], pred[j]],
    #         alpha=0.8
    #     )

    #     ax.set_title(
    #         f"Sample {i+1}: time={time:.3f}, space={space:.3f}"
    #     )
    #     ax.set_ylabel("Function value")
    #     ax.grid(axis="y", alpha=0.3)

    # plt.tight_layout()
    # plt.show()
