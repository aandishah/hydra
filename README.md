This is the readme file for the dataset 'HYDRA' (HYdrological Drought & Rainfall Atlas) developed by Aandishah Tehzeeb Samara.

HYDRA is a standardized dataset of ten climate variability indices and their teleconnections with global hydroclimate. It includes five ocean-based indices (ENSO, PDO, AMO, IOD, TPI) computed from HadISST 1.1 sea surface temperature, and five atmospheric circulation-based indices (AO, AAO, NAO, NPI, SAM) computed from ERA5 geopotential height and sea level pressure.
HYDRA covers three analysis periods: long term (LT, 1901–2024) for the ocean-based indices, intermediate term (IT, 1941–2024) for the atmospheric circulation-based indices, and short term (ST, 1981–2024) for all ten indices. Indices are computed monthly and aggregated to seasonal means (DJF, MAM, JJA, SON). Hydroclimate fields are drawn from CRU TS v4.09 (precipitation, temperature, PDSI), GLEAM v4.2a (soil moisture) and GPCP v2.3 (precipitation). Gridded products are provided in NetCDF-4 format; index time series and extreme states are also provided in CSV format.
The folder 'indices' contains the computed climate index time series (monthly and seasonal) for all periods.
The subfolder 'validation' contains the index series correlated against published reference indices for validation.
The folder 'teleconnections' contains the seasonal regression and correlation maps between each index and each hydroclimate field: LT for the ocean-based indices, IT for the atmospheric circulation-based indices, and ST for all indices.
The folder 'agreements' contains the agreement composite files, which show where the top and bottom 15th percentile years of each index coincide with extreme hydroclimate years, for the LT, IT and ST periods.
The labels 'LT', 'IT' and 'ST' in file names refer to the long-term, intermediate-term and short-term periods. 'TOP' and 'BOT' refer to the top and bottom 15th percentile of index years.
A conda environment file (hydra_env.yml) is included to help install the Python package dependencies.
Please find more details (e.g., index definitions, validation results, and the agreement composite method) in the reference: The HYdroclimate Drought and Rainfall Atlas (HYDRA): A global atlas of hydroclimate variability linked to large-scale ocean-atmosphere climate modes. The manuscript has been submitted to Scientific Data for peer review.
Project website: https://aandishah.github.io/hydra
Please contact Aandishah Tehzeeb Samara at Columbia University (Lamont-Doherty Earth Observatory) at aandishah.s@columbia.edu for any questions.
