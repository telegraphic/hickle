Hickle
======

Hickle is a HDF5 based clone of Pickle, with a twist. Instead of serializing to a pickle file,
Hickle dumps to a HDF5 file. It is designed to be a "drop-in" replacement for pickle (for common data objects). That is: it is a neat little way of dumping python variables to file. Hickle is fast, and allows for transparent compression of your data (LZF / GZIP).

Why use Hickle?
---------------

While hickle is designed to be a drop-in replacement for pickle (and json), it works very differently. 
Instead of serializing / json-izing, it instead stores the data using the excellent h5py module.

The main reasons to use hickle are:

  1. it's faster than pickle and cPickle
  2. it stores data in HDF5
  3. You can easily compress your data.

The main reasons not to use hickle are:

  1. You don't want to store your data in HDF5. While hickle can serialize arbitrary python objects, this functionality is provided only for convenience, and you're probably better off just using the pickle module.
  2. You want to convert your data in JSON. For this, use a json or uJson.

So, if you want your data in HDF5, or if your pickling is taking too long, give hickle a try. Hickle is particularly good at storing large numpy arrays, thanks to h5py running under the hood. 


Performance comparison
----------------------

Hickle runs a lot faster than pickle with its default settings, and a little faster than pickle with `protocol=2` set:

```Python
In [1]: import numpy as np

In [2]: x = np.random.random((2000, 2000))

In [3]: import pickle

In [4]: f = open('foo.pkl', 'w')

In [5]: %time pickle.dump(x, f)  # slow by default
CPU times: user 2 s, sys: 274 ms, total: 2.27 s
Wall time: 2.74 s

In [6]: f = open('foo.pkl', 'w')

In [7]: %time pickle.dump(x, f, protocol=2)  # actually very fast
CPU times: user 18.8 ms, sys: 36 ms, total: 54.8 ms
Wall time: 55.6 ms

In [8]: import hickle

In [9]: f = open('foo.hkl', 'w')

In [10]: %time hickle.dump(x, f)  # a bit faster
dumping <type 'numpy.ndarray'> to file <HDF5 file "foo.hkl" (mode r+)>
CPU times: user 764 µs, sys: 35.6 ms, total: 36.4 ms
Wall time: 36.2 ms
```

So if you do continue to use pickle, add the `protocol=2` keyword (thanks @mrocklin for pointing this out).  
  
For storing python dictionaries of lists, hickle beats the python json encoder, but is slower than uJson. For a dictionary with 64 entries, each containing a 4096 length list of random numbers, the times are:


    json took 2633.263 ms
    uJson took 138.482 ms
    hickle took 232.181 ms


It should be noted that these comparisons are of course not fair: storing in HDF5 will not help you convert something into JSON, nor will it help you serialize a string. But for quick storage of the contents of a python variable, it's a pretty good option.

Installation guidelines (for Linux and Mac OS).
----------------------------------------------------------------------------------------------------

### Easy method
Install with `pip` by running `pip install hickle` from the command line.

### Manual install

1. You should have Python 2.7 and above installed

2. Install h5py 
(Official page: http://docs.h5py.org/en/latest/build.html)

3. Install hdf5 
(Official page: http://www.hdfgroup.org/ftp/HDF5/current/src/unpacked/release_docs/INSTALL)

4. Download `hickle`:
via terminal: git clone https://github.com/telegraphic/hickle.git
via manual download: Go to https://github.com/telegraphic/hickle and on right hand side you will find `Download ZIP` file

5. cd to your downloaded `hickle` directory

6. Then run the following command in the `hickle` directory: 
     `python setup.py install`
     

Usage example
-------------

Hickle is nice and easy to use, and should look very familiar to those of you who have pickled before:

```python
import os
import hickle as hkl
import numpy as np
    
# Create a numpy array of data
array_obj = np.ones(32768, dtype='float32')
    
# Dump to file
hkl.dump(array_obj, 'test.hkl', mode='w')
    
# Dump data, with compression
hkl.dump(array_obj, 'test_gzip.hkl', mode='w', compression='gzip')
  
# Compare filesizes
print 'uncompressed: %i bytes' % os.path.getsize('test.hkl')
print 'compressed:   %i bytes' % os.path.getsize('test_gzip.hkl')
    
# Load data
array_hkl = hkl.load('test_gzip.hkl')
    
# Check the two are the same file
assert array_hkl.dtype == array_obj.dtype
assert np.all((array_hkl, array_obj))
```

Compression options
-------------------

hickle passes keyword arguments on to h5py, so you can do things like:
  ```python
  hkl.dump(array_obj, 'test_lzf.hkl', mode='w', compression='lzf', scaleoffset=0, 
           chunks=(100, 100), shuffle=True, fletcher32=True)
  ```
Have a look at http://docs.h5py.org/en/latest/high/dataset.html for an explanation
of these keywords.
