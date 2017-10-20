import numpy as np

WATER_DENS = 9.982e2
WATER_DYN_VISC = 1.002e-03


def p_vol(D):
    """Return particle volume"""
    return 0.5235987755982988 * D**3


def Reynolds(D, U, rho=WATER_DENS, mu=WATER_DYN_VISC):
    """Return Reynolds number"""
    return rho * D * U / mu


def Stokes(D, U, mu=WATER_DYN_VISC):
    """Return Stokes force"""
    return 9.42477796076938 * mu * D * U


def calc_mean_diams(D):
    """Return mean diameter by frequency, by mass"""
    vols = p_vol(D)
    return D.mean(), np.dot(D, vols)/vols.sum()


def create_diam_bins(D, nbins):
    """Split diameters into ``nbins`` bins. 
    Return indices of diameters in each bin, bin counts"""
    eps = 1.e-10 * D.min()
    bins = np.linspace(D.min() - eps, D.max() + eps, nbins+1)
    bin_idxs = [
        np.where((D > bins[i]) & (D <= bins[i+1]))[0]
        for i in range(nbins)
    ]
    bin_sizes = np.array([len(b) for b in bin_idxs])
    return bin_idxs, bin_sizes
