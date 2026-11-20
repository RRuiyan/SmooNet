# SmooNet and Functional Differential Equation (FDE) 

## Overview

SmooNet (Smooth Operator Neural Network) is a neural-network-based framework for learning and approximating functional differential equations (FDEs). Unlike ordinary differential equation (ODE) models, FDEs allow system dynamics to depend on both the current state and historical trajectories, enabling the modeling of memory and delay effects in complex dynamical systems. A unique feature of the SmooNet is that its hidden layer is a continuum with infinite number of nodes, which we call a hidden string, and the complexity of SmooNet is completely determined by the smoothness of the learnable functions. 

This repository contains the implementation, simulation studies, and examples accompanying the published SmooNet methodology.

The repository includes:

- Core implementation files in C++ and R.
- Simulation studies for method evaluation.
- Example analyses using sunspot time series data.
- Supporting scripts for reproducing numerical experiments.

## Repository Contents

### Core Implementation

- `c_dynamic_FDE.cpp`  
  C++ implementation of the SmooNet/FDE framework.

- `dynamic.FDE.R`  
  R functions for model fitting, estimation, and prediction.

### Simulation Studies

- `sim.1.R`
- `sim.2.R`
- `sim.3.R`
- `sim.4.R`
- `sim.5.R`

Simulation scripts used to evaluate model performance under multiple data-generating mechanisms.


### Example Data Analysis

- sunspot_10days.R
- sunspots_1849_10days.RData
Example analysis and prediction of sunspot time-series data.


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


## Reference

Luo, R., & Qi, X. (2026). *SmooNet: Smooth operator neural network and functional differential equation*. Neural Networks, 199, 108688.

DOI: https://doi.org/10.1016/j.neunet.2026.108688

## Citation

If you use this repository in your research, please cite:

```bibtex
@article{luo2026smoonet,
  author  = {Luo, Ruiyan and Qi, Xin},
  title   = {SmooNet: Smooth operator neural network and functional differential equation},
  journal = {Neural Networks},
  volume  = {199},
  pages   = {108688},
  year    = {2026},
  doi      = {10.1016/j.neunet.2026.108688}
}
```

## License

Specify the license for this repository (e.g., MIT License, GPL-3.0, or another license of your choice).
