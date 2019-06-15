# FastAdjust

Tools for working with 3D SIMION potential arrays and for converting them to HDF5.

Features:
- convert 3D SIMION potential array files to HDF5
- fast adjust potential arrays for a set of applied voltages
- efficiently calculate the potential or field at a given coordinate (e.g., for trajectory calculations)

TODO:
 - add support for SIMION files that utilise symmetry

## Install

Clone the source,

```
git clone https://github.com/ad3ller/fastadjust  
cd fastadjust
```

If the SIMION 8.1 python API is not located in "C:\Program Files\SIMION-8.1\lib\python" then modify `PATH` in fastadjust/simion.py.

Next, install fastadjust with setuptools

```
python setup.py install
```

## Notes

The SIMION 8.1 python API, which is used by `fastadjust` to convert PAs to HDF5, does not support python 3.  The other features of `fastadjust` work in python 2 or 3. 

To create a python 2.7 environment, see https://conda.io/docs/user-guide/tasks/manage-python.html

Alternatively, you could try:

```
2to3 -w "C:\Program Files\SIMION-8.1\lib\python\SIMION\PA.py"
```

## Usage

See notebooks.
