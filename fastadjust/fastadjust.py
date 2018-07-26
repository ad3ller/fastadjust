# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 11:00:15 2018

@author: Adam
"""
from math import floor
import numpy as np

class FastAdjust(object):
    """ A class for working with three-dimensional fast-adjust potential arrays.
    
        Parameters
        ----------
            electrode  : numpy.array() of electrode data (bool)
            pa         : numpy.array() fast-adjust potential array data

        Attributes
        ----------
            nx, ny, nz      : sizes of the potential array grid
            x0, y0, z0      : real coordinates of the grid position [0, 0, 0]
            dy, dy, dz      : real sizes of the grid spacing
            shape           : (nx, ny, nz)
            delta           : (dx, dy, dz)
            num_el          : number of fast-adjust electrodes
            extent          : real grid extents, [xmin, xmax, ymin, ymax, zmin, zmax]
            xvals, yvals, zvals
                            : real axes of the grid

        Methods
        -------
            grid            : xyz meshgrid for pa
            potential       : electric potential for a set of applied voltages
            field           : electric field for a set of applied voltages
            amp_field       : amplitude of the electric field for a set of applied voltages
            grad_field      : gradient of the electric field for a set of applied voltages
            
            real_g          : convert grid coordinates to real coordinates
            grid_r          : convert real coordinates to grid coordinates
            inside_r        : is (x, y, z) inside the pa?
            electrode_r     : is (x, y, z) an electrode?
            interp_r        : interpolate grid values at (x, y, z)
            grad_r          : gradient of grid values at (x, y, z)
            pa_r            : fast-adjust potentials interpolated at (x, y, z)
            potential_r     : electric potential at (x, y, z) for a set of applied voltages
            field_r         : electric field at (x, y, z) for a set of applied voltages
            amp_field_r     : amplitude of the electric field at (x, y, z) for a set of applied voltages
            grad_field_r    : gradient of the amplitude of the electric field at (x, y, z) for a set of applied voltages

        Notes
        -----
            (x, y, z)      : coordinates in real space
            (xg, yg, zg)   : non-integer grid coordinates
            (xn, yn, zn)   : integer grid coordinates
    
    """
    def __init__(self, electrode, pa):
        self.electrode = electrode
        self.pa = pa
        # checks
        assert isinstance(self.electrode, (np.ndarray))
        assert isinstance(self.pa, (np.ndarray))
        self.shape = np.shape(self.electrode) # (x, y, z)
        assert len(self.shape) == 3
        assert np.shape(self.pa)[:-1] == self.shape
        # properties
        self.nx, self.ny, self.nz, self.num_el = np.shape(self.pa)
        # default grid position and spacing
        self.x0 = self.y0 = self.z0 = 0.0
        self.dx = self.dy = self.dz = 1.0
        
    @property
    def delta(self):
        """ Grid spacing

            Returns
            -------
            (dx, dy, dz)
        """
        return self.dx, self.dy, self.dz

    @property
    def xvals(self):
        """ The real x-coordinates of the pa.

            Returns
            -------
            numpy.array()
        """
        return np.arange(self.nx) * self.dx + self.x0

    @property
    def yvals(self):
        """ The real y-coordinates of the pa.

            Returns
            -------
            numpy.array()
        """
        return np.arange(self.ny) * self.dy + self.y0

    @property
    def zvals(self):
        """ The real z-coordinates of the pa.

            Returns
            -------
            numpy.array()
        """
        return np.arange(self.nz) * self.dz + self.z0

    @property
    def extent(self):
        """ The real coordinates of the pa extents.

            Returns
            -------
            numpy.array([xmin, xmax, ymin, ymax, zmin, zmax])
        """
        return np.array([self.x0, self.x0 + (self.nx - 1) * self.dx,
                         self.y0, self.y0 + (self.ny - 1) * self.dy,
                         self.z0, self.z0 + (self.nz - 1) * self.dz])

    """ ==========================================================
        array methods
        ==========================================================
    """ 

    def grid(self):
        """ Grid corresponding to pa.

            Returns
            -------
            np.array(), np.array(), np.array()
                X, Y, Z
        """
        return np.meshgrid(self.xvals, self.yvals, self.zvals, indexing='ij')

    def potential(self, voltages):
        """ The electric potential.

            Parameters
            ----------
            voltages  : float64 or list or 1-D numpy.ndarray 
                        (array length must match number of electrodes)
            Returns
            -------
            numpy.array()
                sum(pa * voltage[i])
        """
        return np.sum(self.pa * voltages, axis=-1)

    def field(self, voltages):
        """ The electric field.

            Parameters
            ----------
            voltages  : float64 or list or 1-D numpy.ndarray 
                        (array length must match number of electrodes)

            Returns
            -------
            [numpy.array(), numpy.array(), numpy.array()]
                ex, ey, ez
        """
        return np.gradient(self.potential(voltages), *self.delta)

    def amp_field(self, voltages, subset=None):
        """ The amplitude of the electric field.

            Parameters
            ----------
            voltages  : float64 or list or 1-D numpy.ndarray 
                        (array length must match number of electrodes)

            Returns
            -------
            numpy.array()
                
                sum(ei^2)^0.5
        """
        ei = np.array(self.field(voltages))
        return np.sum(ei**2.0, axis=0)**0.5

    def grad_field(self, voltages):
        """ The gradient of the electric field.

            Parameters
            ----------
            voltages  : float64 or list or 1-D numpy.ndarray 
                        (array length must match number of electrodes)

            Returns
            -------
            [numpy.array(), numpy.array(), numpy.array()]
                gx, gy, gz
        """
        return np.gradient(self.amp_field(voltages), *self.delta)

    """ ==========================================================
        point (xyz) methods
        ==========================================================
    """

    def real_g(self, grid_coord):
        """ Convert grid coordinates to real coordinates (i.e., the inverse of grid_r) 
        
            Parameters
            ----------
            grid_coord    : tuple (xg : float64, yg : float64, zg : float64)

            Returns
            -------
            float64, float64, float64
                (x, y, z)
        """
        xg, yg, zg = grid_coord
        x = xg * self.dx + self.x0
        y = yg * self.dy + self.y0
        z = zg * self.dz + self.z0
        return x, y, z


    def grid_r(self, coord):
        """ Convert real coordinates into grid coordinates.

            Parameters
            ----------
            coord    : tuple (x, y, z)

            Returns
            -------
            float64, float64, float64
                (xg, yg, zg) non-integer pa grid coordinate corresponding to (x, y, z)
        """
        x, y, z = coord
        xg = (x - self.x0) / self.dx
        yg = (y - self.y0) / self.dy
        zg = (z - self.z0) / self.dz 
        return xg, yg, zg

    def inside_r(self, coord):
        """ Is coord=(x, y, z) inside the pa boundary?

            Parameters
            ----------
            coord    : tuple (x, y, z)

            Returns
            -------
            bool
        """
        xg, yg, zg = self.grid_r(coord)
        return (0.0 <= xg <= self.nx - 1.0) and (0.0 <= yg <= self.ny - 1.0) and (0.0 <= zg <= self.nz - 1.0)

    def electrode_r(self, coord, outside=True):
        """ Is nearest grid point to coord=(x, y, z) an electrode? 

            Parameters
            ----------
            coord    : tuple (x, y, z)
            outside  : return if coord is outside pa, e.g., bool or np.nan 
                       (default: True, i.e., anywhere outside the pa is considered an electrode)
            
            Returns
            -------
            bool (or outside)
        """
        # grid coordinate
        xg, yg, zg = self.grid_r(coord)
        if (0 <= xg <= self.nx - 1) and (0 <= yg <= self.ny - 1) and (0 <= zg <= self.nz - 1):
            # nearest grid point
            xn = int(round(xg))
            yn = int(round(yg))
            zn = int(round(zg))
            return self.electrode[xn, yn, zn]
        else:
            return outside

    def interp_r(self, phi, coord):
        """ interpolate phi at coord=(x, y, z)
        
            Parameters
            ----------
            phi      : np.array() 
            coord    : tuple (x, y, z)

            Returns
            -------
            float64
        """
        assert list(phi.shape)[:3] == list(self.shape), "shape of phi must match FastAdjust()"
        # grid coordinate
        xg, yg, zg = self.grid_r(coord)
        if (0 <= xg <= self.nx - 1) and (0 <= yg <= self.ny - 1) and (0 <= zg <= self.nz - 1):
            if xg == int(xg) and yg == int(yg) and zg == int(zg):
                return phi[int(xg), int(yg), int(zg)]
            else:
                # try trilinear interpolation
                try:
                    ## enclosing cube coordinates
                    x0 = int(floor(xg))
                    x1 = x0 + 1
                    y0 = int(floor(yg))
                    y1 = y0 + 1
                    z0 = int(floor(zg))
                    z1 = z0 + 1
                    ## interpolate along x
                    wx = (xg - x0)
                    c00 = phi[x0, y0, z0] * (1 - wx) + phi[x1, y0, z0] * wx
                    c01 = phi[x0, y0, z1] * (1 - wx) + phi[x1, y0, z1] * wx
                    c10 = phi[x0, y1, z0] * (1 - wx) + phi[x1, y1, z0] * wx
                    c11 = phi[x0, y1, z1] * (1 - wx) + phi[x1, y1, z1] * wx
                    ## interpolate along y
                    wy = (yg - y0)
                    c0 = c00 * (1 - wy) + c10 * wy
                    c1 = c01 * (1 - wy) + c11 * wy
                    ## interpolate along z
                    wz = (zg - z0)
                    c = c0 * (1 - wz) + c1 * wz
                    return c
                except IndexError:
                    return np.nan
                except:
                    raise
        else:
            return np.nan

    def grad_r(self, phi, coord):
        """ gradient of phi at coord=(x, y, z)

            Parameters
            ----------
            phi      : np.array() 
            coord    : tuple (x, y, z)

            Returns
            -------
            numpy.array()
        """
        x, y, z = coord
        gx = (self.interp_r(phi, (x + self.dx / 2.0, y, z)) - 
              self.interp_r(phi, (x - self.dx / 2.0, y, z))) / self.dx
        gy = (self.interp_r(phi, (x, y + self.dy / 2.0, z)) - 
              self.interp_r(phi, (x, y - self.dy / 2.0, z))) / self.dy
        gz = (self.interp_r(phi, (x, y, z + self.dz / 2.0)) - 
              self.interp_r(phi, (x, y, z - self.dz / 2.0))) / self.dz
        return gx, gy, gz

    def pa_r(self, coord):
        """ fast-adjust potential array at coord=(x, y, z)

            Parameters
            ----------
            coord    : tuple (x, y, z)

            Returns
            -------
            numpy.array()
        """
        return self.interp_r(self.pa, coord)

    def potential_r(self, coord, voltages):
        """ electric potential at coord=(x, y, z)

            Parameters
            ----------
            coord    : tuple (x, y, z)
            voltages : numpy.array()

            Returns
            -------
            float64
        """
        assert len(voltages == self.num_el), "length of voltages must match the number of electrodes"
        phi = np.sum(self.pa_r(coord) * voltages)
        return phi

    def field_r(self, coord, voltages):
        """ electric field at coord=(x, y, z)

            Parameters
            ----------
            coord    : tuple (x, y, z)
            voltages : numpy.array()

            Returns
            -------
            ex, ey, ez
        """
        x, y, z = coord
        ex = (self.potential_r((x + self.dx / 2.0, y, z), voltages) - 
              self.potential_r((x - self.dx / 2.0, y, z), voltages)) / self.dx
        ey = (self.potential_r((x, y + self.dy / 2.0, z), voltages) - 
              self.potential_r((x, y - self.dy / 2.0, z), voltages)) / self.dy
        ez = (self.potential_r((x, y, z + self.dz / 2.0), voltages) - 
              self.potential_r((x, y, z - self.dz / 2.0), voltages)) / self.dz
        return ex, ey, ez

    def amp_field_r(self, coord, voltages):
        """ amplitude of the field at coord=(x, y, z)

            Parameters
            ----------
            coord    : tuple (x, y, z)
            voltages : numpy.array()

            Returns
            -------
            float64
        """
        ex, ey, ez = self.field_r(coord, voltages)
        return (ex**2.0 + ey**2.0 + ez**2.0)**0.5

    def grad_field_r(self, coord, voltages):
        """ gradient of the amplitude of the field at coord=(x, y, z)

            Parameters
            ----------
            coord    : tuple (x, y, z)
            voltages : numpy.array()

            Returns
            -------
            gx, gy, gz
        """
        x, y, z = coord
        gx = (self.amp_field_r((x + self.dx / 2.0, y, z), voltages) - 
              self.amp_field_r((x - self.dx / 2.0, y, z), voltages)) / self.dx
        gy = (self.amp_field_r((x, y + self.dy / 2.0, z), voltages) - 
              self.amp_field_r((x, y - self.dy / 2.0, z), voltages)) / self.dy
        gz = (self.amp_field_r((x, y, z + self.dz / 2.0), voltages) - 
              self.amp_field_r((x, y, z - self.dz / 2.0), voltages)) / self.dz
        return gx, gy, gz
