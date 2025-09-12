# Load Packages
import xarray as xr
import numpy as np
import pandas as pd
import xesmf as xe
import xcdat as xc

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.basemap import Basemap
import matplotlib.dates as mdates

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from scipy import stats
from scipy.stats import pearsonr
from scipy.stats import zscore
from scipy.stats import t
from scipy.signal import butter, filtfilt, detrend, welch

## Filtering for 2D Data (pandas)

def apply_filter_series(series, order=8, cutoff=1/10, fs=1.0, btype='low', padtype='even', padlen=None):
    """
    Apply Butterworth filter to a pandas Series (e.g. ENSO time series).
    
    Parameters:
    - series: pandas.Series with datetime or numeric index
    - order: filter order
    - cutoff: frequency cutoff (e.g., 1/10 for 10-year lowpass)
    - fs: sampling frequency (e.g., 12 for monthly, 1 for annual)
    - btype: 'low' or 'high'
    - padtype: type of padding ('even' ~ MATLAB symmetric)
    - padlen: int or None (optional)

    Returns:
    - Filtered pandas.Series
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    
    filtered = filtfilt(b, a, series.values, method='pad', padtype=padtype, padlen=padlen)
    return pd.Series(filtered, index=series.index, name=f"{series.name}_filtered")


## Filtering for 3D Data (xarrays)

def apply_filter_xarray(da, order=8, cutoff=1/10, fs=1.0, btype='low', padtype='even', padlen=None):
    """
    Apply Butterworth filter to an xarray.DataArray with a 'time' dimension.

    Parameters:
    - da: xarray.DataArray with 'time' as one dimension
    - order: int, Butterworth filter order
    - cutoff: float, cutoff frequency (e.g., 1/10 for 10-year lowpass)
    - fs: float, sampling frequency (e.g., 12 for monthly, 1 for annual)
    - btype: str, 'low' or 'high'
    - padtype: str, padding type passed to filtfilt
    - padlen: int or None, optional

    Returns:
    - xarray.DataArray with filtered values and same time index
    """
    # Check time dimension
    if "time" not in da.dims:
        raise ValueError("DataArray must have a 'time' dimension.")

    # Drop NaNs for safe filtering
    da_nonan = da.dropna(dim="time")

    # Design Butterworth filter
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)

    # Apply filter
    filtered_values = filtfilt(b, a, da_nonan.values, method='pad', padtype=padtype, padlen=padlen)

    # Reconstruct DataArray
    filtered_da = xr.DataArray(
        filtered_values,
        coords={"time": da_nonan["time"]},
        dims=["time"],
        name=f"{da.name}_filtered" if da.name else None,
        attrs=da.attrs
    )

    return filtered_da


def detrend_dim(da, dim):
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=1)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit