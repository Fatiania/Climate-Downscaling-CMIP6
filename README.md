# Climate Downscaling and CMIP6 Projections

Statistical bias adjustment and monthly climate-variable downscaling for hydroclimatic applications in Morocco, using MIROC6 projections and gridded reference data.

## Source code

### `src/classical_downscaling.py`
Implements variable-specific monthly corrections for MIROC6 climate data. Precipitation and runoff are treated with separate occurrence and positive-intensity adjustments; temperature is corrected using additive monthly differences, whereas other positive hydroclimatic variables use multiplicative monthly factors. Calibration is conducted over 1981–2005, validation over 2006–2025, and future projections extend from 2026 to 2100. Correction parameters are estimated separately by spatial point and calendar month.

### `archive/cmip6_miroc6_earlier_calibration.py`
Implements an alternative bias-adjustment framework with blocked cross-validation for method selection. Candidate methods include monthly scaling, empirical quantile mapping and quantile delta mapping, with variable-specific treatment of precipitation, temperature, soil moisture, potential evaporation and shortwave radiation. Historical calibration, independent validation and future scenario processing are handled separately.

## Input and output data

Monthly reference and climate-model files are indexed using spatial identifiers and time labels. Outputs comprise adjusted climate series and method-evaluation diagnostics. Data and original path configurations are maintained separately from the repository.
