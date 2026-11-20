# SmooNet and Functional Differential Equation (FDE) Methods

## Overview

This repository contains source code and simulation scripts for the implementation and study of **SmooNet** and Functional Differential Equation (FDE)-based methodologies.

The repository includes:

- Core implementation files in C++ and R.
- Simulation studies for method evaluation.
- Example analyses using sunspot time series data.
- Supporting scripts for reproducing numerical experiments.

## Repository Contents

### Core Implementation

- `c_dynamic_FDE.cpp`  
  C++ implementation of the SmooNet-based neural FDE methodology.

- `dynamic.FDE.R`  
  R interface and functions for fitting and analyzing FDE models.

### Simulation Studies

- `sim.1.R`
- `sim.2.R`
- `sim.3.R`
- `sim.4.R`
- `sim.5.R`

These scripts reproduce simulation experiments used to evaluate the proposed methodology under different settings.

### Example Data Analysis

- `sunspot_10days.R`  
  Example analysis using sunspot data.

- `sunspots_1849_10days.RData`  
  Sunspot dataset used in the demonstration.

## Requirements

The code is primarily written in **R**, with computational components implemented in **C++**.

Typical R packages that may be required include:

- Rcpp
- stats
- graphics
- methods

Additional package requirements may depend on the specific simulation script being executed.

## Getting Started

Clone the repository:

```bash
git clone https://github.com/RRuiyan/SmooNet.git
```

Open R or RStudio and run the desired script, for example:

```r
source("dynamic.FDE.R")
source("sim.1.R")
```

For the sunspot example:

```r
load("sunspots_1849_10days.RData")
source("sunspot_10days.R")
```

## Reproducibility

The simulation scripts are provided to facilitate reproduction of the numerical experiments associated with the proposed methodology. Users are encouraged to inspect the simulation settings and adapt them to their own applications.

## Citation

If you use this software in your research, please cite the associated publication describing the SmooNet and FDE methodology.

Citation information will be added when the related manuscript is available.

## License

Specify the license for this repository (e.g., MIT License, GPL-3.0, or another license of your choice).
