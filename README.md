# hybrid-drt

`hybrid-drt` is a Python package for probabilistic analysis of electrochemical impedance data using the distribution of relaxation times (DRT) and distribution of phasances (DOP). The key features of `hybrid-drt` are:
1. DRT estimation using a robust, self-tuning hierarchical Bayesian approach (no manual tuning needed)
2. Introduction of the [DOP]() for handling constant-phase features like (pseudo)-inductance, (pseudo)-capacitance, or Warburg-like diffusion
3. DRT/DOP estimation from accelerated joint time/frequency-domain impedance measurements
4. Probabilistic DRT analysis methods (distinguishing pseudo-peaks from true peaks)
5. Automatic creation of equivalent circuit analogs from the DRT
6. Multi-dimensional DRT fitting and analysis

For methodology details, see [this work on probabilistic DRT analysis](https://www.sciencedirect.com/science/article/pii/S001346862300066X) and [this work on joint-domain impedance measurements and DOP transformation](https://www.sciencedirect.com/science/article/pii/S001346862300066X).

*Disclaimer:* `hybrid-drt` is experimental and under active development. The code is provided to demonstrate several conceptual approaches to electrochemical analysis, but the details of the implementation may change in the future.

## Probabilistic DRT analysis and interpretation
The philosophy underpinning the probabilistic methods in `hybrid-drt` is that distribution of relaxation times (DRT) analysis should fulfill two objectives: (1) determine the relaxation magnitude as a function of timescale and (2) identify the distinct processes that comprise the total distribution. The first objective is a regression task that is fulfilled by conventional DRT algorithms, while the second objective is a pseudo-classification task that is ignored by most DRT algorithms. `hybrid-drt` unites the regression and classification views of DRT inversion to clarify DRT interpretation and meaningfully express uncertainty, following the framework developed in [this paper](https://www.sciencedirect.com/science/article/abs/pii/S001346862300066X). The package currently provides several methods for analyzing electrochemical impedance spectroscopy (EIS) data:
* Conventional DRT estimation via a self-tuning hierarchical Bayesian model
* Probabilistic EIS deconvolution using the probability function of relaxation times (PFRT)
* A "dual inversion" algorithm for autonomous discrete model generation, comparison, and selection
* Methods for scoring the accuracy of DRT estimates using regression, classification, and hybrid metrics

## Joint-domain impedance measurement and DRT transformation
Joint-domain ("hybrid") impedance measurements use time-domain chronopotentiometry measurements to quickly probe low-frequency processes ($\leq \sim 100$ Hz), while still accessing high frequencies ($\geq \sim 100$ Hz) with conventional frequency-domain impedance measurements. The two data components can then be joined with the DRT transform, thereby allowing recovery of the full impedance spectrum with high accuracy. This allows full-frequency-range impedance measurements to be completed more than 10x faster than with conventional impedance. For more details, see [this paper](https://www.sciencedirect.com/science/article/pii/S001346862300066X).


## Tutorials
You can find a variety of tutorials in the corresponding folder. There are not yet tutorials for multi-dimensional DRT analysis, as this is a more complex topic. Please feel free to reach out if you are interested.

## Webinar
I gave a webinar providing a general introduction to the DRT, which may be helpful for understanding the DRT and this package. You can find the webinar recording and materials in the `webinar` folder. This uses `hybrid-drt` for a short software demonstration, which can be found in the same folder.

## Requirements
`hybrid-drt` requires the [`mittag-leffler`](https://github.com/jdhuang-csm/mittag-leffler) package.
`hybrid-drt` also requires the following packages:
* numpy
* matplotlib
* scipy
* pandas
* cvxopt
* scikit-learn
* scikit-image
  
The [`galvani`](https://github.com/echemdata/galvani) package can optionally be installed for direct reading of EC-Lab `mpr` files.

## Installation
Install `hybrid-drt` from the repository files using either conda or pip. See `installation.txt` for step-by-step instructions.

## Citing `hybrid-drt`
If you use `hybrid-drt` in published work, please consider citing the revelant work(s) linked above (bibliographic details below):

@article{huang2023reliable,
  title={How reliable is distribution of relaxation times (DRT) analysis? A dual regression-classification perspective on DRT estimation, interpretation, and accuracy},
  author={Huang, Jake and Sullivan, Neal P and Zakutayev, Andriy and O’Hayre, Ryan},
  journal={Electrochimica Acta},
  volume={443},
  pages={141879},
  year={2023},
  publisher={Elsevier}
}

@article{huang2024rapid,
  title={Rapid mapping of electrochemical processes in energy-conversion devices},
  author={Huang, Jake D and Meisel, Charlie and Sullivan, Neal P and Zakutayev, Andriy and O’Hayre, Ryan},
  journal={Joule},
  volume={8},
  number={7},
  pages={2049--2072},
  year={2024},
  publisher={Elsevier}
}


