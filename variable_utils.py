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




def pct_anomalies(
    da,
    var_name="sst",
    ref_sel=('1900-01-01', '2023-12-31'),
):


    # --- 1. Compute seasonal climatology over the reference period ---
    clim_ds = da.temporal.climatology(
        var_name,
        freq="season",
        weighted=True,
        reference_period=ref_sel,
        season_config={"dec_mode": "DJF", "drop_incomplete_djf": True},
    )

    clim = clim_ds[var_name] if isinstance(clim_ds, xr.Dataset) else clim_ds
    clim_by_season = clim.groupby("time.season").mean("time", skipna=True)

    # --- 2. Compute seasonal anomalies (departures) ---
    departures_ds = da.temporal.departures(
        var_name,
        freq="season",
        weighted=True,
        reference_period=ref_sel,
        season_config={"dec_mode": "DJF", "drop_incomplete_djf": True},
    )

    anomalies = departures_ds[var_name] if isinstance(departures_ds, xr.Dataset) else departures_ds

    # --- 3. Compute percentage anomalies ---
    # Same broadcasting logic as your std_anomalies:
    pct_anomalies = anomalies.groupby("time.season") / clim_by_season * 100.0

    return pct_anomalies

ppt_pct_anoms = pct_anomalies(
    ppt,
    var_name="pre",
    #ref_sel=('1981-01-01', '2010-12-31')
)