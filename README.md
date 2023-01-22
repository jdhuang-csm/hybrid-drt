# hybrid-drt

`hybrid-drt` is a Python package for probabilistic analysis of electrochemical data. It currently provides several functions:
* Conventional distribution of relaxation times (DRT) estimation via a self-tuning hierarchical Bayesian model
* Probabilistic deconvolution of electrochemical impedance spectroscopy (EIS) data using the probability function of relaxation times (PFRT)
* Autonomous discrete model generation, comparison, and selection
* Methods for scoring the accuracy of DRT estimates using regression, classification, and hybrid metrics

Tutorials and new functionality will be added soon.

*Disclaimer:* `hybrid-drt` is experimental and under active development. The code is provided to demonstrate several conceptual approaches to electrochemical analysis, but the details of the implementation may change in the future.

## Requirements
`hybrid-drt` requires the `mittag-leffler` package, which is available at [https://github.com/jdhuang-csm/mittag-leffler].
`hybrid-drt` also requires the following packages:
* numpy
* matplotlib
* scipy
* pandas
* cvxopt
* scikit-learn

## Installation
Install `hybrid-drt` from the repository files using either conda or pip. See `installation.txt` for step-by-step instructions.

