# Load Packages
import xarray as xr
import numpy as np
import pandas as pd
import xesmf as xe
import xcdat as xc

from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt, detrend, welch


#import scipy

# from matplotlib import pyplot as plt
# from matplotlib.gridspec import GridSpec
# from mpl_toolkits.basemap import Basemap
# import matplotlib.dates as mdates

# import cartopy.crs as ccrs
# import cartopy.feature as cfeature

# from scipy import stats
# from scipy.stats import pearsonr
# from scipy.stats import t
# from scipy.signal import butter, filtfilt, detrend, welch
#from scipy.signal import detrend

####### COMPUTING INDEXES ################

#### SST #########

def compute_enso(sst_anom: xr.DataArray, 
                 roi_lat=slice(5, -5), 
                 roi_lon=slice(-170, -120),
                 time_slice_full=slice("1900-11-01", "2024-02-28"),
                 time_slice_final=slice("1901-01-01", "2023-12-31"),
                 rolling_months=3,
                 index_name="ENSO") -> xr.DataArray:
    """
    Compute ENSO index from SST anomalies over a Pacific ROI.
    
    Parameters:
    - sst_anom: xarray.DataArray with dims ("time", "lat", "lon")
    - roi_lat: Latitude slice for region of interest (default: 5N–5S)
    - roi_lon: Longitude slice for region of interest (default: 120W–170W)
    - time_slice_full: Full time slice (include buffer for rolling)
    - time_slice_final: Final clipped time slice (removes NaNs from rolling)
    - rolling_months: Number of months for rolling mean (default: 3)
    - index_name: Name to assign to the resulting index (default: "ENSO")
    
    Returns:
    - xr.DataArray of the ENSO index (1D, time)
    """
    # Select ROI
    sst_roi = sst_anom.sel(time=time_slice_full, lat=roi_lat, lon=roi_lon)

    sst_rolling = sst_roi.rolling(time=rolling_months, center=True).mean()

    # Average over ROI
    sst_index = sst_rolling.mean(dim=("lat", "lon"))

    sst_index = sst_index.rename({"sst_anom": index_name})

    # remove edge NaNs
    sst_index = sst_index.sel(time=time_slice_final)

    return sst_index

######### PDO ###############

def _box(da, lat_min, lat_max, lon_min, lon_max):
    """Select lat/lon box via boolean masks; handles dateline crossing."""
    lat_lo, lat_hi = min(lat_min, lat_max), max(lat_min, lat_max)
    lat_mask = (da.lat >= lat_lo) & (da.lat <= lat_hi)
    if lon_min <= lon_max:
        lon_mask = (da.lon >= lon_min) & (da.lon <= lon_max)
    else:
        # dateline crossing
        lon_mask = (da.lon >= lon_min) | (da.lon <= lon_max)
    return da.where(lat_mask & lon_mask, drop=True)

def remove_weighted_global_mean(da: xr.DataArray) -> xr.DataArray:
    """
    Remove area-weighted global mean (lat-lon) from a DataArray.

    Parameters
    ----------
    da : xr.DataArray
        Must have 'lat' and 'lon' dimensions.

    Returns
    -------
    xr.DataArray
        Anomaly with global mean removed at each timestep.
    """
    weights = np.cos(np.deg2rad(da.lat))
    global_mean = da.weighted(weights).mean(dim=["lat", "lon"])
    return da - global_mean

def compute_pdo(
    sst_anom, 
    north_pacific_lat=(20., 70.),      
    north_pacific_lon=(110., -100.),   
    remove_global_mean=True,
    standardize_spatial=True,
    enforce_sign=True
):
    
    anom = sst_anom
    
    # Remove global mean anomaly (area-weighted)
    if remove_global_mean:
        w_lat = np.cos(np.deg2rad(anom.lat))
        gmean = anom.weighted(w_lat).mean(('lat', 'lon'), skipna=True)
        anom = anom - gmean

    # North Pacific domain
    dom = _box(anom, north_pacific_lat[0], north_pacific_lat[1],
                     north_pacific_lon[0], north_pacific_lon[1])

    # Weights for EOFs
    w = np.sqrt(np.cos(np.deg2rad(dom.lat))).clip(min=0)
    dom_w = dom * w

    # Keep only complete columns
    X = dom_w.stack(space=('lat', 'lon'))
    valid = np.isfinite(X).all('time')
    X = X[:, valid]

    # Standardization
    if standardize_spatial:
        std = X.std('time', skipna=True)
        std = xr.where(std == 0, np.nan, std)
        X = X / std

    # Demean over time
    X = X - X.mean('time', skipna=True)

    # SVD → PC1 (PDO) and EOF1
    X_np = np.asarray(X.values, float)
    U, svals, VT = np.linalg.svd(X_np, full_matrices=False)

    pc1 = U[:, 0] * svals[0]
    pc1 = (pc1 - pc1.mean()) / pc1.std(ddof=1)  # unit variance
    eof1_space = VT[0, :]

    # EOF back on grid 
    eof_full = xr.full_like(dom.isel(time=0), np.nan).stack(space=('lat', 'lon'))
    eof_full.loc[dict(space=X.space.values)] = eof1_space
    
    # undo standardization & weights
    if standardize_spatial:
        eof_full = eof_full * std
    W_full = (w * xr.ones_like(dom.isel(time=0))).stack(space=('lat', 'lon'))
    eof_full = (eof_full / W_full).unstack('space')

    varexpl = float((svals[0]**2) / (svals**2).sum())

    # Enforce canonical sign using a coastal box
    if enforce_sign:
        coast_box = _box(eof_full, 25, 45, -150, -125)  # 25–45N, 150–125W
        if np.nanmean(coast_box.values) < 0:
            pc1 = -pc1
            eof_full = -eof_full

    pdo = xr.DataArray(pc1, coords={'time': dom.time}, dims='time', name='PDO',
                       attrs={'units': 'std. dev.', 'note': 'PC1 of North Pacific SST anomalies (Mantua-style)'})
    pdo['PDO'] = pdo
    eof_full.name = 'PDO_EOF1'
    eof_full.attrs['note'] = 'Leading EOF pattern of North Pacific SST anomalies'
    return pdo, eof_full, varexpl

def compute_amo(
    sst_anom,
    na_lat=(0., 60.), na_lon=(-80., 0.),
    remove_global_mean=True,
    standardize=False           # return standardized index (mean 0, std 1) if True
):
    """
    Compute AMO index:
      1) North Atlantic mean (0–60N, 80W–0) with cos(lat) weights
      2) Optionally subtract global mean anomaly (area-weighted) each month

    Returns
    -------
    amo_mon : xr.DataArray (time)  # monthly AMO (unsmoothed)
    amo_smooth : xr.DataArray (time) or None  # smoothed AMO (if smooth>1)
    """
    
    anom = sst_anom

    # 2) North Atlantic box mean (0–60N, 80W–0), area-weighted
    na = _box(anom, na_lat[0], na_lat[1], na_lon[0], na_lon[1])
    w_na = np.cos(np.deg2rad(na.lat))
    na_mean = na.weighted(w_na).mean(dim=('lat', 'lon'), skipna=True)

    # 3) Optionally subtract global mean anomaly (area-weighted)
    if remove_global_mean:
        w_glob = np.cos(np.deg2rad(anom.lat))
        gmean = anom.weighted(w_glob).mean(dim=('lat', 'lon'), skipna=True)
        amo_mon = na_mean - gmean
    else:
        amo_mon = na_mean
        
    #     # Optional: remove area-weighted global mean
    # if remove_global_mean == True:
    #     amo_mon = remove_weighted_global_mean(na_mean)
    # else:
    #     amo_mon = na_mean

    # Drop auxiliary groupby coord if present
    try:
        amo_mon = amo_mon.reset_coords('month', drop=True)
    except Exception:
        pass

    # (Optional) standardize
    if standardize:
        amo_mon = (amo_mon - amo_mon.mean('time')) / amo_mon.std('time', ddof=1)
        amo_mon.attrs['units'] = 'std dev'
    else:
        amo_mon.attrs['units'] = '°C'

    amo_mon.name = 'AMO'
    amo_mon.attrs.update({
        'long_name': 'Atlantic Multidecadal Oscillation (North Atlantic minus global mean)',
        'region': 'North Atlantic 0–60°N, 80°W–0°',
        'area_weighting': 'cos(lat)',
        'global_mean_removed': bool(remove_global_mean),
    })
    
    amo_mon['AMO'] = amo_mon

    return amo_mon

def compute_iod(
    sst_anom,
    remove_global_mean = True,
):
    """
    Compute the Dipole Mode Index (DMI / IOD) from SST anomalies.

    Parameters
    ----------
    sst_anom : xr.DataArray
        SST anomaly with dimensions (time, lat, lon). Should already be subset as needed.
    remove_global_mean : bool
        If True, removes area-weighted global mean SST at each time step.

    Returns
    -------
    iod_index : xr.DataArray
        IOD index time series with name 'IOD'
    """
    # Optional: remove area-weighted global mean
    if remove_global_mean == True:
        sst_anom = remove_weighted_global_mean(sst_anom)
    else: 
         sst_anom = sst_anom

    # West box (50°–70°E, 10°S–10°N)
    west_box = sst_anom.sel(lon=slice(50, 70), lat=slice(10, -10))
    west = west_box.mean(dim=["lat", "lon"])

    # East box (90°–110°E, 10°S–0°)
    east_box = sst_anom.sel(lon=slice(90, 110), lat=slice(0, -10))
    east = east_box.mean(dim=["lat", "lon"])

    # IOD index = West - East
    iod_index = west - east
    #iod_index.name = "IOD"
    iod_index['IOD'] = iod_index
    iod_index.attrs["long_name"] = "Dipole Mode Index (West - East SST)"
    iod_index.attrs["units"] = "°C (anomaly)"
    iod_index.attrs["source"] = "Computed from SST anomalies"

    return iod_index


############ SLP ONES ##################

def compute_nao(
    slp_anom,  # SLP anomalies (time, lat, lon), units hPa or Pa (anomalies already)
    atl_lat=(20., 80.),      # 20–80°N
    atl_lon=(-90., 40.),     # 90°W to 40°E (no dateline crossing)
    standardize_spatial=False,
    enforce_sign=True
):
    """
    NAO as PC1 of North Atlantic SLP anomalies.

    Parameters
    ----------
    slp_anom : xr.DataArray
        Sea-level pressure anomalies with dims ('time','lat','lon'). If in Pa,
        unit choice doesn't matter for PC1 (linear scaling), but hPa is typical.
    atl_lat, atl_lon : tuple
        North Atlantic domain bounds (handles dateline crossing via _box()).
    standardize_spatial : bool
        Divide each grid-column by its std over time before SVD (common for EOFs).
    enforce_sign : bool
        Flip sign so positive NAO corresponds to deeper Icelandic Low vs Azores High.

    Returns
    -------
    nao : xr.DataArray (time)
        Standardized PC1 (unit variance), name='NAO'.
    eof1 : xr.DataArray (lat, lon)
        Leading EOF pattern of SLP anomalies (name='NAO_EOF1').
    varexpl : float
        Fraction of variance explained by PC1.
    """
    anom = slp_anom

    # North Atlantic domain
    dom = _box(anom, atl_lat[0], atl_lat[1], atl_lon[0], atl_lon[1])

    # Weights for EOFs (sqrt(cosφ))
    w = np.sqrt(np.cos(np.deg2rad(dom.lat))).clip(min=0)
    dom_w = dom * w

    # Keep only complete columns
    X = dom_w.stack(space=('lat', 'lon'))
    valid = np.isfinite(X).all('time')
    X = X[:, valid]

    # Standardization across time per column
    if standardize_spatial:
        std = X.std('time', skipna=True)
        std = xr.where(std == 0, np.nan, std)
        X = X / std

    # Demean over time
    X = X - X.mean('time', skipna=True)

    # SVD → PC1 (NAO) and EOF1
    X_np = np.asarray(X.values, float)
    U, svals, VT = np.linalg.svd(X_np, full_matrices=False)

    pc1 = U[:, 0] * svals[0]
    pc1 = (pc1 - pc1.mean()) / pc1.std(ddof=1)  # unit variance
    eof1_space = VT[0, :]

    # EOF back on grid
    eof_full = xr.full_like(dom.isel(time=0), np.nan).stack(space=('lat', 'lon'))
    eof_full.loc[dict(space=X.space.values)] = eof1_space

    # undo standardization & weights
    if standardize_spatial:
        eof_full = eof_full * std
    W_full = (w * xr.ones_like(dom.isel(time=0))).stack(space=('lat', 'lon'))
    eof_full = (eof_full / W_full).unstack('space')

    varexpl = float((svals[0]**2) / (svals**2).sum())

    # Enforce canonical NAO sign using Iceland vs Azores boxes
    if enforce_sign:
        az = _box(eof_full, 36, 40, -35, -20).mean().item()   # Azores
        ic = _box(eof_full, 60, 70, -30, -15).mean().item()   # Iceland
        if np.isfinite(az) and np.isfinite(ic) and (ic - az) < 0:
            pc1 = -pc1
            eof_full = -eof_full

    nao = xr.DataArray(pc1, coords={'time': dom.time}, dims='time', name='NAO',
                       attrs={'units': 'std. dev.',
                              'note': 'PC1 of North Atlantic SLP anomalies (PC-based NAO)'})
    eof_full.name = 'NAO_EOF1'
    eof_full.attrs['note'] = 'Leading EOF pattern of North Atlantic SLP anomalies'
    return nao, eof_full#, varexpl



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

def filter_and_plot_seasonal_index(index_seasonal_grouped,
                                   order=8, cutoff=1/10, fs=1,
                                   btype='low', padtype='even', padlen=120):
    """
    Apply a low-pass filter to each season of an index, plot them together,
    and return the combined seasonal grouped DataArray.

    Returns:
    index_10_yr_seasonal_grouped : xarray.core.groupby.GroupBy
        The filtered index series grouped by season.
    """

    # Apply filter per season
    index_10_yr_DJF = apply_filter_xarray(index_seasonal_grouped['DJF'],
                                          order=order, cutoff=cutoff, fs=fs,
                                          btype=btype, padtype=padtype, padlen=padlen)
    index_10_yr_MAM = apply_filter_xarray(index_seasonal_grouped['MAM'],
                                          order=order, cutoff=cutoff, fs=fs,
                                          btype=btype, padtype=padtype, padlen=padlen)
    index_10_yr_JJA = apply_filter_xarray(index_seasonal_grouped['JJA'],
                                          order=order, cutoff=cutoff, fs=fs,
                                          btype=btype, padtype=padtype, padlen=padlen)
    index_10_yr_SON = apply_filter_xarray(index_seasonal_grouped['SON'],
                                          order=order, cutoff=cutoff, fs=fs,
                                          btype=btype, padtype=padtype, padlen=padlen)

    # Plot the filtered time series
    index_10_yr_DJF.plot(figsize=(14, 3), label='DJF')
    index_10_yr_MAM.plot(label='MAM')
    index_10_yr_JJA.plot(label='JJA')
    index_10_yr_SON.plot(label='SON')
    plt.legend(ncol=4)
    plt.show()

    # Combine the list of seasonal-filtered DataArrays into one
    index_10_yr_combined = xr.concat(
        [index_10_yr_DJF, index_10_yr_MAM, index_10_yr_JJA, index_10_yr_SON],
        dim='time'
    )

    # Sort by time
    index_10_yr_combined = index_10_yr_combined.sortby('time')

    # Group by season
    index_10_yr_seasonal_grouped = index_10_yr_combined.groupby('time.season')

    return index_10_yr_seasonal_grouped

def _z_overall(da, name_suffix="_z"):
    z = (da - da.mean("time", skipna=True)) / da.std("time", skipna=True, ddof=1)
    z = z.copy(deep=False)
    z.name = f"{da.name}{name_suffix}" if da.name else "zscore"
    z.attrs.update(da.attrs)
    z.attrs["units"] = "standard deviations"
    z.attrs["standardized_over"] = "overall-time"
    return z

def _z_by_group(da, group, name_suffix="_zgrp"):
    g = da.groupby(group)
    mu = g.mean("time", skipna=True)
    sig = g.std("time", skipna=True, ddof=1)
    z = (g - mu) / sig
    z = z.copy(deep=False)
    z.name = f"{da.name}{name_suffix}" if da.name else "zscore_grouped"
    z.attrs.update(da.attrs)
    z.attrs["units"] = "standard deviations"
    z.attrs["standardized_over"] = f"grouped:{group}"
    return z

def process_climate_index_w_z(var, varname, label, 
                               time_slice=('1901-01-01', '2023-12-30'), 
                               time_slice2=('1902-12-01', '2023-11-30'),
                               filter_period=1/10, padlen_set = None):
    """
    Load, detrend, filter, and seasonally process a monthly climate index,
    plus z-score (std dev) versions for monthly and seasonal data,
    including filtered z-scores.
    """

    index = var[varname].sortby('time').sel(time=slice(*time_slice))
    index.name = label

    # Detrend full time series
    detrended_index = xr.DataArray(
        detrend(index, type='linear'),
        coords=index.coords,
        dims=index.dims,
        name=f"{label}_detrended"
    )

    monthly_dt_filtered = apply_filter_xarray(
        detrended_index, order=8, cutoff=1/10, fs=12,
        btype='low', padtype='even', padlen=padlen_set
    )

    # Seasonal resample
    seasonal = index.sel(time=slice(*time_slice2)).resample(time='QS-DEC').mean(dim='time', skipna=False)
    seasonal.name = f"{label}_seasonal"
    seasonal_grouped = seasonal.groupby('time.season')

    seasonal_filtered_by_each_season_grouped = filter_and_plot_seasonal_index(seasonal_grouped)

    # Seasonal detrend
    seasonal_detrended = xr.DataArray(
        detrend(seasonal, type='linear'),
        coords=seasonal.coords,
        dims=seasonal.dims,
        name=f"{label}_seasonal_detrended"
    )
    seasonal_grouped_dt = seasonal_detrended.groupby('time.season')

    # Filter seasonal detrended
    seasonal_dt_filtered = apply_filter_xarray(
        seasonal_detrended, order=8, cutoff=filter_period, fs=4,
        btype='low', padtype='even', padlen=padlen_set
    )
    seasonal_dt_filtered_by_each_season_grouped = filter_and_plot_seasonal_index(seasonal_grouped_dt)

    # Z-SCORES (standard scores)
    
    monthly_z = _z_overall(index, name_suffix="_z")
    monthly_z_by_month = _z_by_group(index, 'time.month', name_suffix="_z_by_month")

    monthly_dt_z = _z_overall(detrended_index, name_suffix="_z")
    monthly_dt_z_by_month = _z_by_group(detrended_index, 'time.month', name_suffix="_z_by_month")

    seasonal_z = _z_overall(seasonal, name_suffix="_z")
    seasonal_z_by_season = _z_by_group(seasonal, 'time.season', name_suffix="_z_by_season")

    seasonal_dt_z = _z_overall(seasonal_detrended, name_suffix="_z")
    seasonal_dt_z_by_season = _z_by_group(seasonal_detrended, 'time.season', name_suffix="_z_by_season")

    # FILTERED Z-SCORES
    monthly_z_filtered = apply_filter_xarray(
        monthly_z, order=8, cutoff=1/10, fs=12,
        btype='low', padtype='even', padlen=padlen_set
    )
    monthly_dt_z_filtered = apply_filter_xarray(
        monthly_dt_z, order=8, cutoff=1/10, fs=12,
        btype='low', padtype='even', padlen=padlen_set
    )

    seasonal_z_filtered = apply_filter_xarray(
        seasonal_z, order=8, cutoff=filter_period, fs=4,
        btype='low', padtype='even', padlen=padlen_set
    )
    seasonal_dt_z_filtered = apply_filter_xarray(
        seasonal_dt_z, order=8, cutoff=filter_period, fs=4,
        btype='low', padtype='even', padlen=padlen_set
    )
    seasonal_dt_z_grouped = seasonal_dt_z.groupby('time.season')
    seasonal_dt_z_filtered_by_each_season_grouped = filter_and_plot_seasonal_index(seasonal_dt_z_grouped)

    return {
        # index
        "monthly": index,
        "monthly_dt": detrended_index,
        "monthly_dt_filtered": monthly_dt_filtered,
        "seasonal": seasonal,
        "seasonal_grouped": seasonal_grouped,
        "seasonal_dt": seasonal_detrended,
        "seasonal_dt_filtered": seasonal_dt_filtered,
        "seasonal_filtered_by_each_season_grouped": seasonal_filtered_by_each_season_grouped,
        "seasonal_dt_filtered_by_each_season_grouped": seasonal_dt_filtered_by_each_season_grouped,

        # z-scores
        "monthly_z": monthly_z,
        "monthly_z_by_month": monthly_z_by_month,
        "monthly_dt_z": monthly_dt_z,
        "monthly_dt_z_by_month": monthly_dt_z_by_month,
        "seasonal_z": seasonal_z,
        "seasonal_z_by_season": seasonal_z_by_season,
        "seasonal_dt_z": seasonal_dt_z,
        "seasonal_dt_z_by_season": seasonal_dt_z_by_season,
        "seasonal_dt_z_grouped": seasonal_dt_z_grouped,
        "seasonal_dt_z_filtered_by_each_season_grouped": seasonal_dt_z_filtered_by_each_season_grouped,

        # filtered z-scores
        "monthly_z_filtered": monthly_z_filtered,
        "monthly_dt_z_filtered": monthly_dt_z_filtered,
        "seasonal_z_filtered": seasonal_z_filtered,
        "seasonal_dt_z_filtered": seasonal_dt_z_filtered,
    }
