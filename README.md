# HYDRA: HYdrological Drought & Rainfall Atlas

This repository contains a set of streamlined Python functions for performing regression analysis between large-scale climate modes (e.g., ENSO, PDO, IOD) and hydroclimate fields (e.g., PDSI, SST, soil moisture). It is designed for use with datasets that are already seasonally averaged (e.g., DJF, MAM, JJA, SON), and leverages xarray, numpy, and matplotlib for geospatial analysis and visualization.

**Key Features** 

* Lagged Regression Analysis: Supports both positive and negative lags, allowing for "index leads" or "field leads" logic, with customizable lags in both seasons and years.
* Flexible Inputs: Works directly with xarray.DataArray objects containing seasonal time-series data.
* Detrending & Significance Testing: All regressions include safe linear detrending and p-value estimation using a t-distribution, with hatching overlays for non-significant areas.
* Custom Plotting Functions: Includes plotting options for multi-panel seasonal maps and joint maps that overlay ocean only and land only variables on the same projection.

**Applications**

* Exploring teleconnections between climate indices and drought fields
* Attribution of hydroclimatic anomalies to seasonal phases of climate modes
* Visualization of global wetting/drying patterns

**Dependencies**

xarray, numpy, scipy, matplotlib, cartopy

**Structure** [NEED TO UPDATED]

* regression.py – main regression routines
* plotting.py – multi-panel and dual-variable map plotting functions
* utils.py – detrending, time shifting, and alignment helpers
* notebooks/ – example Jupyter notebooks demonstrating usage on real data

* Project website (GitHub Pages):  [WAITING ON VERFICATION]
[https://aandishah.github.io/hydra](https://aandishah.github.io/hydra)

