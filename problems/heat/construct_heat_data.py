import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List

###############################################
# Example 1: 1D Heat Equation with no forcing #
###############################################

class HeatNoForcingDataset(Dataset):
    def __init__(self, theta_star=0.5, T=1.0, L=np.pi, N=100, sigma=1.0):
        """Function to initialize the dataset for the heat equation.

        Args:
            theta_star (float, optional): True value for the heat conductivity parameter. Defaults to 0.5.
            T (float, optional): Time boundary value. Defaults to 1.0.
            L (_type_, optional): Space boundary value. Defaults to np.pi.
            N (int, optional): Number of data points. Defaults to 100.
            sigma (float, optional): Standard deviation of Gaussian noise. Defaults to None.
        """

        # Construct the X array with both time and space points
        tvec = np.linspace(start=0, stop=T, num=N)
        xvec = np.linspace(start=0, stop=L, num=N)
        # np.save('data/tvec.npy', tvec)
        # np.save('data/xvec.npy', xvec)

        # Construct the noisy y values
        u_star = self.compute_solution(tvec, xvec, theta_star)
        u_noisy = self.add_noise(u_star, sigma)
        # np.save('data/heatEquation_soln.npy', u_star)
        # np.save(f'data/heatEquation_noise={sigma}.npy', u_noisy)

        # Construct the full dataset
        self.data = np.array([tvec, xvec], dtype=np.float32).T             
        self.y = u_noisy.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        sample = self.data[idx]
        y = self.y[idx]
        return sample, y

    def compute_solution(self, t: float, x: List[float], theta: float):
        u = np.sin(x) * np.exp(-theta * t)
        return u

    def add_noise(self, u: List[float], sigma: float):        
        sd = np.std(u)
        u_noisy = u + (sd * np.random.normal(loc=0, scale=sigma))  # TODO: check this            
        return u_noisy

    def compute_stats(self):
        data_mean = np.mean(self.data, axis=0)
        data_std = np.std(self.data, axis=0)

        return data_mean, data_std

    def get_data(self):
        return torch.from_numpy(self.data)

    def get_y(self):
        return torch.from_numpy(self.y)

