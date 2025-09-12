# HYDRA: HYdrological Drought & Rainfall Atlas

This repository contains a set of streamlined Python functions for performing seasonal lagged regression analysis between large-scale climate modes (e.g., ENSO, PDO, IOD) and hydroclimate fields (e.g., PDSI, SST, soil moisture). It is designed for use with datasets that are already seasonally averaged (e.g., DJF, MAM, JJA, SON), and leverages xarray, numpy, and matplotlib for efficient geospatial analysis and visualization.

**Key Features** 

* Lagged Regression Analysis: Supports both positive and negative lags, allowing for "index leads" or "field leads" logic, with customizable lags in both seasons and years.
* Flexible Inputs: Works directly with xarray.DataArray objects containing seasonal time-series data (no need for groupby preprocessing).
* Detrending & Significance Testing: All regressions include safe linear detrending and p-value estimation using a t-distribution, with hatching overlays for non-significant areas.
* Multi-variable Support: Simultaneously regresses both PDSI and SST fields on a given index in a consistent framework.
* Custom Plotting Functions: Includes modular plotting options for multi-panel seasonal maps and joint maps that overlay ocean only and land only variables on the same projection.
* Season-aware Shifting: Internally handles time alignment for different lag configurations without requiring a separate "season" coordinate.

**Applications**

* Exploring teleconnections between climate indices and drought fields
* Attribution of hydroclimatic anomalies to seasonal ENSO/PDO/IOD phases
* Visualization of global wetting/drying patterns
* Teaching tools for climate dynamics and statistical analysis

**Dependencies**

xarray, numpy, scipy, matplotlib, cartopy

**Structure**

* regression.py – main regression routines
* plotting.py – multi-panel and dual-variable map plotting functions
* utils.py – detrending, time shifting, and alignment helpers
* notebooks/ – example Jupyter notebooks demonstrating usage on real data
