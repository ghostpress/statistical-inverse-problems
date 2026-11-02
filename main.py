import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import matplotlib.pyplot as plt

import utils
import training.training as training
from problems.heat_equation import construct_heat_data
from models import neuralNetwork  

#############################
# 0. Set problem parameters #
#############################

torch.manual_seed(1026)
Niter = 100
batch_size = 512

################
# 1. Load data #
################

training_data = construct_heat_data.HeatNoForcingDataset(N=10000)
train_dataloader = DataLoader(training_data, batch_size=batch_size)
data_stats = train_dataloader.dataset.compute_stats()

# Display some training examples and statistics from the dataloader
item = next(iter(train_dataloader))
print(f"Shape of X: {item[0].shape} {item[0].dtype}")
print(f"Shape of y: {item[1].shape} {item[1].dtype}")
print(f"Example row in X: {item[0][8]}")
print(f"Mean and std of X: {torch.from_numpy(data_stats[0]), torch.from_numpy(data_stats[1])}")

######################
# 2. Construct model #
######################

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
model = neuralNetwork.NeuralNetwork(1, [50,50,50,50,50])

##################
# 3. Train model #
##################

loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
training_outcomes = training.train(optimizer, model, loss, train_dataloader, device, Niter)  # now outputs a dictionary and the trained model

#################
# 4. Test model #
#################

test_loss = torch.nn.L1Loss()
test_data = construct_heat_data.HeatNoForcingDataset(N=1000)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
test_stats = test_dataloader.dataset.compute_stats()
test_outcomes = training.test(test_dataloader, model, test_loss, device)

###################
# 5. Plot results #
###################

#utils.plot_training_loss(training_outcomes[0]["epoch_losses"], Niter)
#utils.plot_parameter_gradients(training_outcomes[0]["parameter_gradients"], Niter, "./images", "testNN")
#utils.plot_predictions(test_data.get_data(), test_data.get_y(), test_outcomes[0], '', 'testNN')

# TODOs:
# 1. update utils.plot_predictions()
# 2. implement pytorch Dataset classes instead of using numpy in construct_heat_data
# 3. set training.training.train() to save model checkpoints every 1000 epochs
# 4. train on scc for 10000 epochs
# 5. consult Indrajit on noise added to simulated data - line from PDE-READ paper
# 6. continue DeepMoD implementation
