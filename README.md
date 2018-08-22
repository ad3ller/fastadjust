# FastAdjust

Tools for working with 3D SIMION electric potential arrays and for converting them to HDF5.

Features:
- fast adjust potential arrays with python.
- calculate the potential or field at a given position for a set of applied voltages (e.g., for trajectory calculations with time-varying voltages)  
- convert potential array data to HDF5 (useful for subsequently working with, for example, python 3 or Julia). 

TODO
- 2D potential arrays
- slicing
- larger-than-physical-memory potential arrays (dask arrays).

## Install

Requires a copy of the SIMION python API, for which you'll also need a python 2.7 environment. To create a new environment using anaconda, see https://conda.io/docs/user-guide/tasks/manage-python.html

Clone the source,

```
git clone https://github.com/ad3ller/fastadjust  
cd fastadjust
```

If your copy of the SIMION python API is not located in "C:\Program Files\SIMION-8.1\lib\python" then modify `PATH` in fastadjust.simion.py.

Activate your python 2 environment and then install fastadjust with setuptools

```
(py27) python setup.py install
```

Python 2 is only needed for the initial conversion of SIMION potential arrays to HDF5 data.  If you plan to subsequently use fastadjust in another python environment (e.g., python 3) then install it there also (obviously).

## Usage

See notebooks.
