.. hickle documentation master file, created by
   sphinx-quickstart on Fri Dec 14 15:39:45 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to hickle's documentation!
==================================


Hickle is an HDF5-based clone of `pickle`, with a twist: instead of serializing to a pickle file,
Hickle dumps to an HDF5 file (Hierarchical Data Format). It is designed to be a "drop-in" replacement for pickle (for common data objects), but is
really an amalgam of `h5py` and `dill`/`pickle` with extended functionality.

That is: `hickle` is a neat little way of dumping python variables to HDF5 files that can be read in most programming
languages, not just Python. Hickle is fast, and allows for transparent compression of your data (LZF / GZIP).



.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. automodule:: hickle
   :members: load, dump


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
