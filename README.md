# hybrid-drt

`hybrid-drt` is a Python package for probabilistic analysis of electrochemical data. 

The philosophy underpinning `hybrid-drt` is that distribution of relaxation times (DRT) analysis should fulfill two objectives: (1) determine the relaxation magnitude as a function of timescale and (2) identify the distinct processes that comprise the total distribution. The first objective is a regression task that is fulfilled by conventional DRT algorithms, while the second objective is a pseudo-classification task that is ignored by most DRT algorithms. `hybrid-drt` unites the regression and classification views of DRT inversion to clarify DRT interpretation and meaningfully express uncertainty, following the framework developed in [this paper](https://www.sciencedirect.com/science/article/abs/pii/S001346862300066X). The package currently provides several methods for analyzing electrochemical impedance spectroscopy (EIS) data:
* Conventional DRT estimation via a self-tuning hierarchical Bayesian model
* Probabilistic EIS deconvolution using the probability function of relaxation times (PFRT)
* A "dual inversion" algorithm for autonomous discrete model generation, comparison, and selection
* Methods for scoring the accuracy of DRT estimates using regression, classification, and hybrid metrics

Additional tutorials and new functionality will be added soon.

*Disclaimer:* `hybrid-drt` is experimental and under active development. The code is provided to demonstrate several conceptual approaches to electrochemical analysis, but the details of the implementation may change in the future.

## Requirements
`hybrid-drt` requires the `mittag-leffler` package, which is available at https://github.com/jdhuang-csm/mittag-leffler.
`hybrid-drt` also requires the following packages:
* numpy
* matplotlib
* scipy
* pandas
* cvxopt
* scikit-learn

## Installation
Install `hybrid-drt` from the repository files using either conda or pip. See `installation.txt` for step-by-step instructions.

