Hickle
======

Hickle is a HDF5 based clone of Pickle, with a twist. Instead of serializing to a pickle file,
Hickle dumps to a HDF5 file. It is designed to be a "drop-in" replacement for pickle. That is: 
it is a neat little way of dumping python variables to file.

Why use Hickle?
---------------

While hickle is designed to be a drop-in replacement for pickle (and json), it works very differently. 
Instead of serializing / json-izing, it instead stores the data using the excellent h5py module.

The main reasons to use hickle are:

  1. it's faster than pickle and cPickle
  2. it stores data in HDF5
  3. You can easily compress your data.

So, if you want your data in HDF5, or if your pickling is taking too long, give hickle a try. Hickle is particularly good at storing large numpy arrays, thanks to h5py running under the hood.

Performance comparison
----------------------

For storing large numpy arrays, hickle wins hands down again both pickle and the faster cPickle. For example, on my macbook, dumping and loading a (1, 32768) numpy array:

    
    pickle took 2160.628 ms
    cPickle took 800.014 ms
    hickle took 18.020 ms
    
  
For storing python dictionaries of lists, hickle beats the python json encoder, but is slower than uJson. For a dictionary with 64 entries, each containing a 4096 length list of random numbers, the times are:


    json took 2633.263 ms
    uJson took 138.482 ms
    hickle took 232.181 ms


It should be noted that these comparisons are of course not fair: storing in HDF5 will not help you convert something into JSON, nor will it help you serialize a string. But for quick storage of the contents of a python variable, it's a pretty good option.

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
print "uncompressed: %i bytes"%os.path.getsize('test.hkl')
print "compressed:   %i bytes"%os.path.getsize('test_gzip.hkl')
    
# Load data
array_hkl = hkl.load('test_gzip.hkl')
    
# Check the two are the same file
assert array_hkl.dtype == array_obj.dtype
assert np.all((array_hkl, array_obj))
```





