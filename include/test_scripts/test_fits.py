import numpy as np
from matplotlib import pyplot as plt
import astropy.io.fits as fits

def read_fits_c_order(filename, nrows, ncols):
    from astropy.io import fits
    data = fits.getdata(filename)
    return np.array(data, copy=False).reshape((ncols, nrows), order='F').T
    
plop = fits.getdata("../results/test.fits")

plop2d = fits.getdata("../results/test_2d.fits")