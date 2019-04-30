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
        """ grid corresponding to pa

            Returns
            -------
            np.array(), np.array(), np.array()
                X, Y, Z
        """
        return np.meshgrid(self.xvals, self.yvals, self.zvals, indexing='ij')

    def potential(self, voltages):
        """ electric potential

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
        """ electric field

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
        """ amplitude of the electric field

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
        """ gradient of the electric field

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
        """ convert grid coordinates to real coordinates (i.e., the inverse of grid_r) 
        
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
        """ convert real coordinates into grid coordinates

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

    def inside_g(self, grid_coord):
        """ is grid_coord=(xg, yg, zg) inside the pa boundary?

            Parameters
            ----------
            grid_coord    : tuple (xg, yg, zg)

            Returns
            -------
            bool
        """
        xg, yg, zg = grid_coord
        return (0.0 <= xg <= self.nx - 1.0) and (0.0 <= yg <= self.ny - 1.0) and (0.0 <= zg <= self.nz - 1.0)

    def inside_r(self, coord):
        """ is coord=(x, y, z) inside the pa boundary?

            Parameters
            ----------
            coord    : tuple (x, y, z)

            Returns
            -------
            bool
        """
        grid_coord = self.grid_r(coord)
        return  self.inside_g(grid_coord)

    def electrode_g(self, grid_coord, outside=True):
        """ is nearest grid point to grid_coord=(xg, yg, zg) an electrode? 

            Parameters
            ----------
            grid_coord    : tuple (xg, yg, zg)
            outside       : return if coord is outside pa, e.g., bool or np.nan 
                           (default: True, i.e., anywhere outside the pa is considered an electrode)
            
            Returns
            -------
            bool (or outside)
        """
        # grid coordinate
        xg, yg, zg = grid_coord
        if (0 <= xg <= self.nx - 1) and (0 <= yg <= self.ny - 1) and (0 <= zg <= self.nz - 1):
            # nearest grid point
            xn = int(round(xg))
            yn = int(round(yg))
            zn = int(round(zg))
            return self.electrode[xn, yn, zn]
        else:
            return outside

    def electrode_r(self, coord, outside=True):
        """ is nearest grid point to coord=(x, y, z) an electrode? 

            Parameters
            ----------
            coord    : tuple (x, y, z)
            outside  : return if coord is outside pa, e.g., bool or np.nan 
                       (default: True, i.e., anywhere outside the pa is considered an electrode)
            
            Returns
            -------
            bool (or outside)
        """
        grid_coord = self.grid_r(coord)
        return self.electrode_g(grid_coord, outside=outside)

    def interp_g(self, phi, grid_coord):
        """ interpolate phi at grid_coord=(xg, yg, zg)
        
            Parameters
            ----------
            phi           : np.array() 
            grid_coord    : tuple (xg, yg, zg)

            Returns
            -------
            float64
        """
        # grid coordinate
        xg, yg, zg = grid_coord
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
        grid_coord = self.grid_r(coord)
        return self.interp_g(phi, grid_coord)
        

    def grad_g(self, phi, grid_coord):
        """ gradient of phi at grid_coord=(xg, yg, zg)

            Parameters
            ----------
            phi           : np.array() 
            grid_coord    : tuple (xg, yg, zg)

            Returns
            -------
            numpy.array()
        """
        xg, yg, zg = grid_coord
        gx = (self.interp_g(phi, (xg + 0.5, yg, zg)) - 
              self.interp_g(phi, (xg - 0.5, yg, zg))) / self.dx
        gy = (self.interp_g(phi, (xg, yg + 0.5, zg)) - 
              self.interp_g(phi, (xg, yg - 0.5, zg))) / self.dy
        gz = (self.interp_g(phi, (xg, yg, zg + 0.5)) - 
              self.interp_g(phi, (xg, yg, zg - 0.5))) / self.dz
        return gx, gy, gz

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
        grid_coord = self.grid_r(coord)
        return self.grad_g(phi, grid_coord)

    def pa_g(self, grid_coord):
        """ fast-adjust potential array at grid_coord=(xg, yg, zg)

            Parameters
            ----------
            grid_coord    : tuple (xg, yg, zg)

            Returns
            -------
            numpy.array()
        """
        return self.interp_g(self.pa, grid_coord)

    def pa_r(self, coord):
        """ fast-adjust potential array at coord=(x, y, z)

            Parameters
            ----------
            coord    : tuple (x, y, z)

            Returns
            -------
            numpy.array()
        """
        grid_coord = self.grid_r(coord)
        return self.pa_g(grid_coord)

    def potential_g(self, grid_coord, voltages):
        """ electric potential at grid_coord=(xg, yg, zg)

            Parameters
            ----------
            grid_coord    : tuple (xg, yg, zg)
            voltages      : numpy.array()

            Returns
            -------
            float64
        """
        xg, yg, zg = grid_coord
        # try trilinear interpolation
        try:
            ## enclosing cube coordinates
            xn = int(floor(xg))
            yn = int(floor(yg))
            zn = int(floor(zg))
            phi = np.sum(self.pa[xn : xn + 2, yn : yn + 2, zn : zn + 2, :] * voltages, axis=-1)
            ## interpolate along x
            wx = (xg - xn)
            c00 = phi[0, 0, 0] * (1 - wx) + phi[1, 0, 0] * wx
            c01 = phi[0, 0, 1] * (1 - wx) + phi[1, 0, 1] * wx
            c10 = phi[0, 1, 0] * (1 - wx) + phi[1, 1, 0] * wx
            c11 = phi[0, 1, 1] * (1 - wx) + phi[1, 1, 1] * wx
            ## interpolate along y
            wy = (yg - yn)
            c0 = c00 * (1 - wy) + c10 * wy
            c1 = c01 * (1 - wy) + c11 * wy
            ## interpolate along z
            wz = (zg - zn)
            c = c0 * (1 - wz) + c1 * wz
            return c
        except IndexError:
            return np.nan
        except:
            raise

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
        grid_coord = self.grid_r(coord)
        return self.potential_g(grid_coord, voltages)

    def phi_r(self, coord, voltages):
        """ electric potential at coord=(x, y, z)
            
            legacy (slower) version of potential_r().

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

    def field_g(self, grid_coord, voltages):
        """ electric field at grid_coord=(xg, yg, zg)

            Parameters
            ----------
            grid_coord    : tuple (xg, yg, zg)
            voltages      : numpy.array()

            Returns
            -------
            ex, ey, ez
        """
        xg, yg, zg = grid_coord
        # trilinear gradient interpolation
        try:
            ## enclosing cube coordinates
            xn = int(floor(xg))
            yn = int(floor(yg))
            zn = int(floor(zg))
            phi = np.sum(self.pa[xn - 1 : xn + 3, yn - 1 : yn + 3, zn - 1: zn + 3, :] * voltages, axis=-1)
            
            ## interpolation
            wx = (xg - xn)
            wy = (yg - yn)
            wz = (zg - zn)
            
            # Ex
            if wx < 0.5:
                ex11 = (phi[1, 1, 1] - phi[0, 1, 1]) * (0.5 - wx) + (phi[2, 1, 1] - phi[1, 1, 1]) * (0.5 + wx)
                ex12 = (phi[1, 1, 2] - phi[0, 1, 2]) * (0.5 - wx) + (phi[2, 1, 2] - phi[1, 1, 2]) * (0.5 + wx)
                ex21 = (phi[1, 2, 1] - phi[0, 2, 1]) * (0.5 - wx) + (phi[2, 2, 1] - phi[1, 2, 1]) * (0.5 + wx)
                ex22 = (phi[1, 2, 2] - phi[0, 2, 2]) * (0.5 - wx) + (phi[2, 2, 2] - phi[1, 2, 2]) * (0.5 + wx)
            else:
                ex11 = (phi[3, 1, 1] - phi[2, 1, 1]) * (wx - 0.5) + (phi[2, 1, 1] - phi[1, 1, 1]) * (1.5 - wx)
                ex12 = (phi[3, 1, 2] - phi[2, 1, 2]) * (wx - 0.5) + (phi[2, 1, 2] - phi[1, 1, 2]) * (1.5 - wx)
                ex21 = (phi[3, 2, 1] - phi[2, 2, 1]) * (wx - 0.5) + (phi[2, 2, 1] - phi[1, 2, 1]) * (1.5 - wx)
                ex22 = (phi[3, 2, 2] - phi[2, 2, 2]) * (wx - 0.5) + (phi[2, 2, 2] - phi[1, 2, 2]) * (1.5 - wx)
            ## interpolate along y
            ex1 = ex11 * (1 - wy) + ex21 * wy
            ex2 = ex12 * (1 - wy) + ex22 * wy
            ## interpolate along z
            ex = (ex1 * (1 - wz) + ex2 * wz ) / self.dx
            
            # Ey
            if wy < 0.5:
                ey11 = (phi[1, 1, 1] - phi[1, 0, 1]) * (0.5 - wy) + (phi[1, 2, 1] - phi[1, 1, 1]) * (0.5 + wy)
                ey12 = (phi[1, 1, 2] - phi[1, 0, 2]) * (0.5 - wy) + (phi[1, 2, 2] - phi[1, 1, 2]) * (0.5 + wy)
                ey21 = (phi[2, 1, 1] - phi[2, 0, 1]) * (0.5 - wy) + (phi[2, 2, 1] - phi[2, 1, 1]) * (0.5 + wy)
                ey22 = (phi[2, 1, 2] - phi[2, 0, 2]) * (0.5 - wy) + (phi[2, 2, 2] - phi[2, 1, 2]) * (0.5 + wy)
            else:
                ey11 = (phi[1, 3, 1] - phi[1, 2, 1]) * (wy - 0.5) + (phi[1, 2, 1] - phi[1, 1, 1]) * (1.5 - wy)
                ey12 = (phi[1, 3, 2] - phi[1, 2, 2]) * (wy - 0.5) + (phi[1, 2, 2] - phi[1, 1, 2]) * (1.5 - wy)
                ey21 = (phi[2, 3, 1] - phi[2, 2, 1]) * (wy - 0.5) + (phi[2, 2, 1] - phi[2, 1, 1]) * (1.5 - wy)
                ey22 = (phi[2, 3, 2] - phi[2, 2, 2]) * (wy - 0.5) + (phi[2, 2, 2] - phi[2, 1, 2]) * (1.5 - wy)
            ## interpolate along x
            ey1 = ey11 * (1 - wx) + ey21 * wx
            ey2 = ey12 * (1 - wx) + ey22 * wx
            ## interpolate along z
            ey = (ey1 * (1 - wz) + ey2 * wz ) / self.dy

            # Ez
            if wz < 0.5:
                ez11 = (phi[1, 1, 1] - phi[1, 1, 0]) * (0.5 - wz) + (phi[1, 1, 2] - phi[1, 1, 1]) * (0.5 + wz)
                ez12 = (phi[1, 2, 1] - phi[1, 2, 0]) * (0.5 - wz) + (phi[1, 2, 2] - phi[1, 2, 1]) * (0.5 + wz)
                ez21 = (phi[2, 1, 1] - phi[2, 1, 0]) * (0.5 - wz) + (phi[2, 1, 2] - phi[2, 1, 1]) * (0.5 + wz)
                ez22 = (phi[2, 2, 1] - phi[2, 2, 0]) * (0.5 - wz) + (phi[2, 2, 2] - phi[2, 2, 1]) * (0.5 + wz)
            else:
                ez11 = (phi[1, 1, 3] - phi[1, 1, 2]) * (wz - 0.5) + (phi[1, 1, 2] - phi[1, 1, 1]) * (1.5 - wz)
                ez12 = (phi[1, 2, 3] - phi[1, 2, 2]) * (wz - 0.5) + (phi[1, 2, 2] - phi[1, 2, 1]) * (1.5 - wz)
                ez21 = (phi[2, 1, 3] - phi[2, 1, 2]) * (wz - 0.5) + (phi[2, 1, 2] - phi[2, 1, 1]) * (1.5 - wz)
                ez22 = (phi[2, 2, 3] - phi[2, 2, 2]) * (wz - 0.5) + (phi[2, 2, 2] - phi[2, 2, 1]) * (1.5 - wz)
            ## interpolate along x
            ez1 = ez11 * (1 - wx) + ez21 * wx
            ez2 = ez12 * (1 - wx) + ez22 * wx
            ## interpolate along y
            ez = (ez1 * (1 - wy) + ez2 * wy ) / self.dz
            return ex, ey, ez
        except IndexError:
            return np.nan, np.nan, np.nan
        except:
            raise

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
        grid_coord = self.grid_r(coord)
        return self.field_g(grid_coord, voltages)

    def amp_field_g(self, grid_coord, voltages):
        """ amplitude of the field at grid_coord=(xg, yg, zg)

            Parameters
            ----------
            grid_coord    : tuple (xg, yg, zg)
            voltages      : numpy.array()

            Returns
            -------
            float64
        """
        ex, ey, ez = self.field_g(grid_coord, voltages)
        return (ex**2.0 + ey**2.0 + ez**2.0)**0.5

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
        grid_coord = self.grid_r(coord)
        return self.amp_field_g(grid_coord, voltages)

    def grad_field_g(self, grid_coord, voltages):
        """ gradient of the amplitude of the field at grid_coord=(xg, yg, zg)

            Parameters
            ----------
            grid_coord    : tuple (xg, yg, zg)
            voltages      : numpy.array()

            Returns
            -------
            gx, gy, gz
        """
        xg, yg, zg = grid_coord
        gx = (self.amp_field_g((xg + 0.5, yg, zg), voltages) - 
              self.amp_field_g((xg - 0.5, yg, zg), voltages)) / self.dx
        gy = (self.amp_field_g((xg, yg + 0.5, zg), voltages) - 
              self.amp_field_g((xg, yg - 0.5, zg), voltages)) / self.dy
        gz = (self.amp_field_g((xg, yg, zg + 0.5), voltages) - 
              self.amp_field_g((xg, yg, zg - 0.5), voltages)) / self.dz
        return gx, gy, gz

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
        grid_coord = self.grid_r(coord)
        return self.grad_field_g(grid_coord, voltages)
