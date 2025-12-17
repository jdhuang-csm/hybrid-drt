<!-- Short, actionable instructions for AI coding agents working on hybrid-drt -->
# hybrid-drt — Copilot instructions

This file contains concise, repo-specific guidance to help an AI coding agent be productive quickly.

**Quick Goal:** `hybrid-drt` is a Python package for probabilistic DRT (distribution of relaxation times) analysis. Changes should preserve numerical behavior and notebook-driven examples in `tutorials/` and `webinar/`.

**Quick Start (how contributors run and test locally)**
- `requirements.txt`: install runtime deps with `pip install -r requirements.txt`.
- Preferred editable install for development: from repo root run `pip install -e .` (or use `conda develop .` per `installation.txt`).
- If using Stan models, install `cmdstanpy` and a CmdStan toolchain. The code uses `hybdrt/mapping/stan_utils.py` which calls `CmdStanModel`.
  - Example: `pip install cmdstanpy` then run `python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"` (or follow CmdStan docs).
- The package also depends on the external `mittag-leffler` package — install that first (see `installation.txt`).

**Where to look (big-picture architecture & key files)**
- Package root: `hybdrt/` — primary code lives here (modules grouped by feature).
- Data ingestion: `hybdrt/dataload/` — readers, `srcconvert.py`, and standardizers. Example entry point: `hybdrt/dataload/reader.py`.
- Core models & inversion: `hybdrt/models/` — `drt1d.py`, `drtbase.py` implement DRT model classes and fitting logic.
- Mapping / Stan integration: `hybdrt/mapping/` — statistical mapping, utilities, and `stan_models/` (.stan files). Use `hybdrt/mapping/stan_utils.load_model(name)` to compile/load models.
- Matrix & numerical helpers: `hybdrt/matrices/` (basis construction, phasance matrices), `hybdrt/utils/` (EIS helpers, statistics).
- Notebooks & examples: `tutorials/` and `webinar/` demonstrate typical workflows and are the best functional tests for user-facing behavior.

**Project-specific conventions & patterns**
- Feature grouping: code organized by domain areas (dataload, mapping, models, matrices, utils). When adding code, place it in the natural domain folder.
- Stan workflow: `.stan` files live in `hybdrt/mapping/stan_models/`. Use `stan_utils.load_model('name')` (no `.stan` suffix required) so runtime code can compile/load via `CmdStanModel`.
- Notebooks are canonical examples: prefer changes that keep notebooks runnable without heavy manual edits (check `tutorials/Fitting_EIS_data.ipynb`).
- Numerical determinism: many algorithms rely on interpolation grids and precomputed matrices. See notebook note: `DRT(interpolate_integrals=False)` toggles grid generation. Avoid small numeric changes that alter default behavior.

**Developer workflows & non-obvious commands**
- Create the conda env from `hybdrt.yml` (recommended for reproducibility): `conda env create -f hybdrt.yml` then `conda activate hybdrt` (see `installation.txt`).
- Install local dependency `mittag-leffler` before `hybdrt` (see `installation.txt`).
- To compile/run Stan models: install `cmdstanpy` and CmdStan. After installing CmdStan you can compile models via `hybdrt/mapping/stan_utils.load_model(name)` from Python.

**Integration & external deps**
- Runtime deps in `requirements.txt` (numpy, pandas, scipy, matplotlib, cvxopt, scikit-learn, scikit-image). Also required: `mittag-leffler` (external repo) and `cmdstanpy` + CmdStan for Stan models.
- Optional: `galvani` can be installed for direct `.mpr` reading (not required for core algorithms).

**Useful code examples to reference**
- Load & standardize a spectrum: `hybdrt/dataload/reader.py` -> `standardize_z_data` in `srcconvert.py`.
- Load a Stan model: `from hybdrt.mapping.stan_utils import load_model; m = load_model('gp_marginal')` which loads `hybdrt/mapping/stan_models/gp_marginal.stan`.
- Notebook tip: default DRT creation precomputes interpolation grids. To avoid that during tests or minor refactors, construct with `DRT(interpolate_integrals=False)`.

**When changing numeric algorithms**
- Add a notebook example that reproduces the behavior change (put in `tutorials/` or a new prose notebook). Notebooks are used as functional regression tests by maintainers.
- Search for usages of the changed API across `tutorials/` and `webinar/` to avoid breaking examples.

If anything above is unclear or you want sections expanded (e.g., exact CmdStan install steps for Windows, or a short checklist for modifying Stan models), tell me which part and I will update this file.
