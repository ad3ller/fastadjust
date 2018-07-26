# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 11:00:15 2018

@author: Adam
"""
import h5py
import numpy as np
from .fastadjust import FastAdjust

def h5read(fil):
    """ Read a HDF5 file that contains fast-adjust potential array data.
   
        Parameters
        ----------
        fil  : path to HDF5 file

        Returns
        -------
        FastAdjust()
    """
    with h5py.File(fil, 'r') as dfil:
        attrs = dict(dfil.attrs)
        pa = np.array(dfil['/pa'])
        electrode =  np.array(dfil['/electrode'])
        dfil.close()
    fa = FastAdjust(electrode, pa)
    fa.dx = attrs['dx']
    fa.dy = attrs['dy']
    fa.dz = attrs['dz']
    fa.x0 = attrs['x0']
    fa.y0 = attrs['y0']
    fa.z0 = attrs['z0']
    return fa

def h5write(fil, fa):
    """ Write an instance of FastAdjust() to an HDF5 file.
   
        Parameters
        ----------
        fil  : path to output HDF5 file
        fa   : instance of FastAdjust()
    """
    attrs = dict([('nx', fa.nx), ('ny', fa.ny), ('nz', fa.nz), 
                  ('dx', fa.dx), ('dy', fa.dy), ('dz', fa.dz),
                  ('x0', fa.x0), ('y0', fa.y0), ('z0', fa.z0)])
    with h5py.File(fil, 'w') as dfil:
        dfil.create_dataset('electrode', data=fa.electrode)
        dfil.create_dataset('pa', data=fa.pa)
        for key, val in attrs.items():
            dfil.attrs[key] = val
        dfil.close()
