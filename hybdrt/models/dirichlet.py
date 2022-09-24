import numpy as np

from ..utils import stats


def dirichlet_mode(alpha):
    if np.min(alpha) > 1:
        alpha_0 = np.sum(alpha)
        return (alpha - 1) / (alpha_0 - len(alpha))
    else:
        raise ValueError('Mode undefined when alpha <= 1')


def evaluate_log_posterior(h, llh, alpha, include_constants=True):
    h = np.atleast_2d(h)
    llh = np.atleast_2d(llh)
    c = np.max(llh, axis=1)
    llh_offset = llh - c
    lp = np.log(np.sum(h * np.exp(llh_offset), axis=1)) + stats.log_pdf_dirichlet(h, alpha, include_constants)
    if include_constants:
        lp += c

    return lp
