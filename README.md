Hickle
======

Hickle is a HDF5 based clone of Pickle. Instead of serializing to a pickle file,
Hickle dumps to a HDF5 file. It is designed to be as similar to pickle in usage as possible.

Created by Danny Price and Jack Hickish on 2012-05-28.
Copyright (c) 2012 The University of Oxford. All rights reserved.

Why use Hickle?
---------------

Hickle has two main advantages over Pickle:
1) LARGE PICKLE HANDLING. Unpickling a large pickle is slow, as the Unpickler reads the entire pickle 
thing and loads it into memory. In comparison, HDF5 files are designed for large datasets. Things are 
only loaded when accessed. 

2) CROSS PLATFORM SUPPORT. Attempting to unpickle a pickle pickled on Windows on Linux and vice versa
is likely to fail with errors like "Insecure string pickle". HDF5 files will load fine, as long as
both machines have h5py installed.

TODO: Add support for chunking and compression
