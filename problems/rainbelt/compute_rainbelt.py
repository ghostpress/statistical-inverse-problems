import os
import xarray as xr
import netCDF4
import numpy as np


def _compute_one_Z(data_slice, tau=10, interpolate_grid=None):
    """Function to compute the rainbelt line for one time period, with optional spatial interpolation.

    Args:
        data_slice (numpy.ndarray): Time x lon x lat slice of the dataset, representing one time period for the curve.
        tau (int, optional): The tempering parameter for computing the curve. Defaults to 10.
        interpolate_grid (int, optional): The number of longitude points along which to integrate - between (1, 400). Defaults to None.

    Returns:
        np.ndarray: The rainbelt line for the given period.
    """    

    # Extract values needed
    precip = data_slice['precipitation'].values  # shape: (time, lon, lat)  
    lon = data_slice['lon'].values
    lat = data_slice['lat'].values
    period_mean = np.mean(precip, axis=0)        # mean along time, lon, lat axes 
    
    # Compute curve using discretized (Reimann sum) softmax function
    numerator = lat * (np.cos(np.deg2rad(lat)) ** tau) * (period_mean ** tau)
    denominator = (np.cos(np.deg2rad(lat)) ** tau) * (period_mean ** tau)
    Z = np.sum(numerator, axis=1) / np.sum(denominator, axis=1)

    # Interpolation
    # TODO add checks for valid values of the grid; and maybe set a default value of 400 (the maximum - no interpolation)
    if interpolate_grid is not None:
        lon_int = np.linspace(lon.min(), lon.max(), interpolate_grid)
        Z_int = np.interp(lon_int, lon, Z)
        Z = Z_int
    
    return Z


def compute_periodic_curves(dataset, period=7, tau=10, one_date=False, date=None, interpolate_grid=None):
    """Function to compute periodic rainbelt curves using Equation (1) in Kalita et. al., 2026. Can compute a single curve for one date,
    or the curves for all the periods in the dataset. If one_date is True and a date is given, the dataset will be sliced to the relevant
    period. Optionally, the curves can be interpolated along the longitude axis.

    Args:
        dataset (xarray.core.dataset.Dataset): Daily GPM rainfall data over AOI; dimensions: (N, M, L) in time, longitude, and latitude.
        period (int, optional): Number of days in period, e.g. 7 (weekly). Defaults to 7.
        tau (int, optional): Softmax parameter controlling the sharpness of the rainbelt estimate. Defaults to 10.
        one_date (bool, optional): Whether to compute the line for just one date. Defaults to False.
        date (np.datetime.date, optional): n64 datetime date for which to compute just one curve. Defaults to None.
        interpolate_grid (int, optional): If not None, the interpolation parameter. Defaults to None.

    Returns:
        np.ndarray: Latitude location of the rainbelt over the period; dimensions: (N//period, M).
        list: List of n64 datetime dates that mark the beginning of each period in the dataset.
    """

    #  TODO add checks for valid values of the grid; and maybe set a default value of 400 (the maximum - no interpolation)

    if one_date and date is not None:
        curr_period = dataset.sel(time=slice(date, date + np.timedelta64(period,'D')))
        curr_period = curr_period.drop_vars('time_bnds')  # drop redundant time_bnds variable
        Z = _compute_one_Z(curr_period, tau=tau, interpolate_grid=interpolate_grid)
        return Z, date
        
    else:
        T = dataset['time'].shape[0]      # total amount of time in dataset
        t = T // period                   # total number of full periods in dataset
        locations = [] #np.empty(shape=(t, dataset.sizes['lon']))                           # TODO update for interpolation
        dates = []
        period_start = 0
        period_end = period
        for per in range(t):
            # Determine the current period and save the date marking its start
            curr_period = dataset.isel(time=slice(period_start, period_end))
            curr_period = curr_period.drop_vars('time_bnds')  # drop redundant time_bnds variable
            dates.append(curr_period['time'].values[0])

            # Compute the Z curve for the current period
            Z = _compute_one_Z(curr_period, tau=tau, interpolate_grid=interpolate_grid)
            
            # Save the curve and update the current period
            #locations[per,] = Z                                                           # TODO update for interpolation
            locations.append(Z)
            period_start += period
            period_end += period        

        return np.array(locations), dates

