# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 11:00:15 2018

@author: Adam
"""
import sys
import os
import numpy as np
from .fastadjust import FastAdjust
PATH = "C:\Program Files\SIMION-8.1\lib\python"

# checks
assert int(sys.version_info[0]) == 2, "SIMION python API only supports python 2.x"
assert os.path.exists(PATH), "SIMION python API not found. Check PATH in fastadjust.simion.py"

# load SIMION.PA
if PATH not in sys.path:
    sys.path.append(PATH)
from SIMION.PA import PA

def pa2npy(fils):
    """ Load SIMION PA files and convert them to normalised numpy.arrays()
    
        Parameters
        ----------
        fils  : a list of paths to fast adjust SIMION files, e.g., [example.PA1, example.PA2, example.PA3]

        Returns
        -------
        np.array(shape=(nx, ny, nz), dtype=bool), np.array(shape=(nx, ny, nz, num_el), dtype=float64), dict()

        electrode : a boolean array that represents electrode / free space voxels.
        pa        : normalised fast adjust potential arrays for each electrode
        attrs     : attributes extracted from the potential array, (ni, di, etc.).
    """
    num = len(fils)
    # first file
    _pa = PA(file=fils[0])
    # get structure from first file (should be consistent)
    xn = np.arange(_pa.nx())
    yn = np.arange(_pa.ny())
    zn = np.arange(_pa.nz())
    X, Y, Z = np.meshgrid(xn, yn, zn, indexing='ij')
    # attrs
    nx, ny, nz = np.shape(X)
    attrs = dict([('nx', nx), ('ny', ny), ('nz', nz), 
                  ('dx', 1e-3 * _pa.dx_mm()), ('dy', 1e-3 * _pa.dy_mm()), ('dz', 1e-3 * _pa.dz_mm())])
    # find electrode points
    electrode_vec = np.vectorize(_pa.electrode)
    electrode = electrode_vec(X.flatten(), Y.flatten(), Z.flatten()).reshape(np.shape(X))
    # loop over all files
    pa = np.empty([nx, ny, nz, num], dtype='float64')
    for i, fil in enumerate(fils):
        _pa = PA(file=fil)
        pot_vec = np.vectorize(_pa.potential)
        phi = pot_vec(X.flatten(), Y.flatten(), Z.flatten()).reshape(np.shape(X))
        # normalise
        phi = (phi - np.min(phi)) / (np.max(phi) - np.min(phi))
        # output
        pa[:, :, :, i] = phi
    return electrode, pa, attrs

def pa2fa(fils):
    """ Load SIMION PA files and convert them to an instance of FastAdjust()
   
        Parameters
        ----------
        fils  : a list of paths to fast adjust SIMION files, e.g., [example.PA1, example.PA2, example.PA3]

        Returns
        -------
        FastAdjust()
    """
    electrode, pa, attrs = pa2npy(fils)
    fa = FastAdjust(electrode, pa)
    for key, value in attrs.items():
        setattr(fa, key, value)
    return fa