# Load Packages
import xarray as xr
import numpy as np
import pandas as pd
#import xesmf as xe
#import xcdat as xc

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
#from mpl_toolkits.basemap import Basemap
#import matplotlib.dates as mdates

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# import scipy

# from scipy import stats
# from scipy.stats import pearsonr
# from scipy.stats import zscore
# from scipy.stats import t
from scipy.signal import butter, filtfilt, detrend, welch

def single_plot( var_select, overall_title,
               var_cmap, split, var_vval, var_vval_2, var_label):
    # Define the Robinson projection

    robinson = ccrs.Robinson()

    # Create 1x1 grid of subplots with the Robinson projection and adjust hspace
    fig, axes = plt.subplots(1, 1, figsize=(8, 10), subplot_kw={'projection': robinson}) 

    axes.coastlines(resolution='110m', linewidth=0.5)
    axes.add_feature(cfeature.BORDERS, linewidth=0.25, linestyle='--')
    axes.gridlines()

            # Create a grid of lats and lons for pcolormesh

    lats = var_select.lat
    lons = var_select.lon
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    gl = axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=1, color='gray', alpha=0.025, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

            # Create Discrete Color Map
            #cmap = var_cmap
    cmap = plt.get_cmap(var_cmap, split) 

            # !!!!!!!!!!!!!!!! Plot Data on Map !!!!!!!!!!!!!!!!
    pcm = axes.pcolormesh(lon_grid, lat_grid, var_select, 
                                cmap=cmap, vmin = (var_vval), vmax = var_vval_2, 
                                transform=ccrs.PlateCarree())
    
    cbar = plt.colorbar(
        pcm,
        ax=axes,
        orientation='horizontal',
        location='bottom', 
        shrink=0.7,
        pad=0.05,
        extend = "both"
        #ticks=ticks
    )
    
    cbar.set_label(var_label, fontsize=12)
    plt.title(overall_title, fontweight="bold", fontsize = 15) # , x=0.25, y=1.1)
    plt.show()
    
    
def plot_index_and_psd_combined(
    raw_data,
    filtered_data=None,
    filter_label = "10-Yr Low Pass",
    label='Index',
    smoothing=6,                 # used only if filtered_data is None
    fs=12.0,                     # samples per year (12 for monthly, 1 for annual)
    nperseg=None,
    detrend='linear',
    scaling='density',
    xlim_psd=None,               # expects years, e.g. (2, 50)
    ylim_psd=None,
    fill_anoms=True):
    
    """
    Plot a climate index and its PSD (Welch).
    PSD x-axis is plotted as period (years) instead of frequency.

    Parameters
    ----------
    raw_data : xarray.DataArray or pandas.Series
        Must have a 'time' coord / datetime index.
    filtered_data : same type as raw_data or None
        If None, uses a centered rolling mean of length 'smoothing'.
    label : str
        Title for the time series plot.
    smoothing : int
        Window length for centered rolling mean (if filtered_data is None).
    fs : float
        Sampling frequency in samples/year (12 monthly, 1 annual).
    nperseg : int or None
        Welch segment length. Default min(len, 256).
    detrend : {'constant','linear'} or False
        Detrend option passed to scipy.signal.welch.
    scaling : {'density','spectrum'}
        Welch scaling.
    xlim_psd : (float,float) or None
        X-axis limits for PSD (period in years).
    ylim_psd : (float,float) or None
        Y-axis limits for PSD.
    fill_anoms : bool
        Fill positive/negative anomalies in the time series panel.

    Returns
    -------
    fig : matplotlib.figure.Figure
    (periods, pxx, peak_info) : tuple
        PSD period array, PSD values, and a dict with peak info.
    """
    # --- Normalize inputs & extract time axis ---
    if hasattr(raw_data, 'dims'):  # xarray
        if 'time' not in raw_data.dims:
            raise ValueError("Input DataArray must have 'time' as a dimension")
        da = raw_data
        if raw_data.ndim > 1:
            da = raw_data.mean(dim=[d for d in raw_data.dims if d != 'time'])
        da = da.sortby('time').dropna(dim='time')
        time = da.time.values
        y = da.values
        # filtered_data handling
        if filtered_data is None:
            smoothed = da.rolling(time=smoothing, center=True).mean()
            y_smooth = smoothed.values
        else:
            fd = filtered_data
            if hasattr(fd, 'dims'):
                if 'time' not in fd.dims:
                    raise ValueError("filtered_data must have 'time' dimension")
                if fd.ndim > 1:
                    fd = fd.mean(dim=[d for d in fd.dims if d != 'time'])
                fd = fd.sortby('time').dropna(dim='time')
                y_smooth = fd.values
            else:
                y_smooth = filtered_data.values if hasattr(filtered_data, 'values') else np.asarray(filtered_data)
    else:  # pandas Series fallback
        idx = raw_data.index if hasattr(raw_data, 'index') else None
        if idx is None:
            raise ValueError("If not xarray, raw_data must be a pandas Series with a datetime index.")
        time = idx
        y = raw_data.values
        if filtered_data is None:
            sm = raw_data.rolling(window=smoothing, center=True).mean()
            y_smooth = sm.values
        else:
            y_smooth = filtered_data.values if hasattr(filtered_data, 'values') else np.asarray(filtered_data)

    # --- PSD (Welch) ---
    if nperseg is None:
        nperseg = min(len(y), 256)
    freqs, pxx = welch(y, fs=fs, nperseg=nperseg, detrend=detrend, scaling=scaling)

    # Convert to periods in years
    periods = np.where(freqs == 0, np.inf, 1.0 / freqs)

    # Peak detection (skip inf at f=0)
    finite_mask = np.isfinite(periods)
    if np.any(finite_mask):
        peak_idx = np.argmax(pxx[finite_mask])
        peak_period = periods[finite_mask][peak_idx]
        peak_val = pxx[finite_mask][peak_idx]
        peak_info = dict(idx=int(peak_idx), period_years=float(peak_period), value=float(peak_val))
    else:
        peak_info = dict(idx=None, period_years=np.nan, value=np.nan)

    # --- Plot (two subplots, width ratio 4:1) ---
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(20, 4.5),
        gridspec_kw={'width_ratios': [4, 1], 'wspace': 0.15}
    )

    # Left: time series
    if fill_anoms:
        ax0.fill_between(time, 0, y, where=y > 0, color='pink', alpha=0.9)
        ax0.fill_between(time, 0, y, where=y < 0, color='powderblue', alpha=0.9)
    ax0.plot(time, y, color='black', linewidth=0.5, label=label)
    if y_smooth is not None:
        ax0.plot(time, y_smooth, color='black', linewidth=1.5, label=filter_label)
    ax0.axhline(0, color='k', linewidth=0.8)
    ax0.set_title(label)
    ax0.set_xlabel('Year')
    #ax0.grid(True, alpha=0.3)
    ax0.legend(frameon=False, loc='upper left')
    ax0.set_xlim(time[0], time[-1])

    # Right: PSD
    ax1.plot(periods, pxx, '-k', label='PSD')
    #if np.isfinite(peak_info['period_years']):
    ax1.axvline(peak_info['period_years'], color='gray', linestyle=':', lw=1)
    ax1.plot(peak_info['period_years'], peak_info['value'], 'o',
                 mfc='none', mec='red', mew=2)
    ax1.legend(title=f"Peak ≈ {peak_info['period_years']:.2f} yrs", frameon=False, loc='upper right')
    ax1.set_xlabel('Period (years)')
    ax1.set_ylabel('Power spectral density')
    ax1.set_title('PSD')
    
    if xlim_psd is not None:
        ax1.set_xlim(*xlim_psd)

    plt.show()
    #return fig, (periods, pxx, peak_info)
    
def single_plot_pacific_centered(var_select, overall_title,
                var_cmap, split, var_min, var_max, var_label):

    # Set up a Pacific-centered Robinson projection
    pacific_robinson = ccrs.Robinson(central_longitude=180)

    # Create figure and axis with Pacific-centered projection
    fig, axes = plt.subplots(1, 1, figsize=(8, 10), subplot_kw={'projection': pacific_robinson}) 
    axes.coastlines(resolution='110m', linewidth=1)
    axes.add_feature(cfeature.BORDERS, linewidth=0.25, linestyle='--')

    # Gridlines
    gl = axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                        linewidth=1, color='gray', alpha=0.025, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    # Shift longitudes if necessary (e.g., from 0–360 to -180–180)
    lats = var_select.lat
    lons = var_select.lon

    # Ensure lons are from -180 to 180 for Pacific-centered plotting
    lons_shifted = np.where(lons > 180, lons - 360, lons)
    
    # Sort longitudes and corresponding data for correct plotting
    sort_idx = np.argsort(lons_shifted)
    lons_sorted = lons_shifted[sort_idx]
    var_sorted = var_select[:, sort_idx]

    lon_grid, lat_grid = np.meshgrid(lons_sorted, lats)

    # Colormap
    cmap = plt.get_cmap(var_cmap, split) 

    # Plot
    pcm = axes.pcolormesh(lon_grid, lat_grid, var_sorted, 
                          cmap=cmap, vmin=var_min, vmax=var_max, 
                          transform=ccrs.PlateCarree())

    # Colorbar
    cbar = plt.colorbar(pcm, ax=axes, shrink=0.7, location='bottom', pad=0.05)
    cbar.set_label(var_label)

    # Title
    plt.title(overall_title, fontweight="bold", fontsize=15)
    plt.show()
