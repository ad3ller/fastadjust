# FastAdjust

Tools for working with 3D SIMION potential arrays and for converting them to HDF5.

Features:
- convert SIMION potential array files to HDF5
- fast adjust potential arrays for a set of applied voltages
- efficiently calculate the potential or field at a given coordinate (e.g., for trajectory calculations)

## Install

`fastadjust` includes tools for converting SIMION potential arrays (PA files) to HDF5 data.  It makes use of the SIMION 8.1 python API, for which you'll also need a python 2.7 environment. To create a new environment using anaconda, see https://conda.io/docs/user-guide/tasks/manage-python.html

Clone the source,

```
git clone https://github.com/ad3ller/fastadjust  
cd fastadjust
```

If your copy of the SIMION 8.1 python API is not located in "C:\Program Files\SIMION-8.1\lib\python" then modify `PATH` in fastadjust/simion.py.

Next, activate your python 2 environment and then install fastadjust with setuptools

```
(py27) python setup.py install
```

Python 2 is only needed for converting PAs to HDF5.  The other features of `fastadjust` work in python 2 or 3.  But to use `fastadjust` in another python environment (e.g., python 3) you'll need to install it there also (obviously).

## Usage

See notebooks.
