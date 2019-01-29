[![Build Status](https://travis-ci.org/telegraphic/hickle.svg?branch=master)](https://travis-ci.org/telegraphic/hickle)
[![JOSS Status](http://joss.theoj.org/papers/0c6638f84a1a574913ed7c6dd1051847/status.svg)](http://joss.theoj.org/papers/0c6638f84a1a574913ed7c6dd1051847)


Hickle
======

Hickle is a [HDF5](https://www.hdfgroup.org/solutions/hdf5/) based clone of `pickle`, with a twist: instead of serializing to a pickle file,
Hickle dumps to a HDF5 file (Hierarchical Data Format). It is designed to be a "drop-in" replacement for pickle (for common data objects), but is
really an amalgam of `h5py` and `dill`/`pickle` with extended functionality.

That is: `hickle` is a neat little way of dumping python variables to HDF5 files that can be read in most programming
languages, not just Python. Hickle is fast, and allows for transparent compression of your data (LZF / GZIP).

Why use Hickle?
---------------

While `hickle` is designed to be a drop-in replacement for `pickle` (or something like `json`), it works very differently.
Instead of serializing / json-izing, it instead stores the data using the excellent [h5py](https://www.h5py.org/) module.

The main reasons to use hickle are:

  1. It's faster than pickle and cPickle.
  2. It stores data in HDF5.
  3. You can easily compress your data.

The main reasons not to use hickle are:

  1. You don't want to store your data in HDF5. While hickle can serialize arbitrary python objects, this functionality is provided only for convenience, and you're probably better off just using the pickle module.
  2. You want to convert your data in human-readable JSON/YAML, in which case, you should do that instead.

So, if you want your data in HDF5, or if your pickling is taking too long, give hickle a try.
Hickle is particularly good at storing large numpy arrays, thanks to `h5py` running under the hood.

Documentation
-------------

Documentation for hickle can be found at [telegraphic.github.io/hickle/](http://telegraphic.github.io/hickle/).


Usage example
-------------

Hickle is nice and easy to use, and should look very familiar to those of you who have pickled before.

In short, `hickle` provides two methods: a [hickle.load](http://telegraphic.github.io/hickle/toc.html#hickle.load)
method, for loading hickle files, and a [hickle.dump](http://telegraphic.github.io/hickle/toc.html#hickle.dump)
method, for dumping data into HDF5. Here's a complete example:

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
print('uncompressed: %i bytes' % os.path.getsize('test.hkl'))
print('compressed:   %i bytes' % os.path.getsize('test_gzip.hkl'))

# Load data
array_hkl = hkl.load('test_gzip.hkl')

# Check the two are the same file
assert array_hkl.dtype == array_obj.dtype
assert np.all((array_hkl, array_obj))
```

### HDF5 compression options

A major benefit of `hickle` over `pickle` is that it allows fancy HDF5 features to
be applied, by passing on keyword arguments on to `h5py`. So, you can do things like:
  ```python
  hkl.dump(array_obj, 'test_lzf.hkl', mode='w', compression='lzf', scaleoffset=0,
           chunks=(100, 100), shuffle=True, fletcher32=True)
  ```
A detailed explanation of these keywords is given at http://docs.h5py.org/en/latest/high/dataset.html,
but we give a quick rundown below.

In HDF5, datasets are stored as B-trees, a tree data structure that has speed benefits over contiguous
blocks of data. In the B-tree, data are split into [chunks](http://docs.h5py.org/en/latest/high/dataset.html#chunked-storage),
which is leveraged to allow [dataset resizing](http://docs.h5py.org/en/latest/high/dataset.html#resizable-datasets) and
compression via [filter pipelines](http://docs.h5py.org/en/latest/high/dataset.html#filter-pipeline). Filters such as
`shuffle` and `scaleoffset` move your data around to improve compression ratios, and `fletcher32` computes a checksum.
These file-level options are abstracted away from the data model.

Recent changes
--------------

* December 2018: Accepted to Journal of Open-Source Software (JOSS).
* June 2018: Major refactor and support for Python 3.
* Aug 2016: Added support for scipy sparse matrices `bsr_matrix`, `csr_matrix` and `csc_matrix`.

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
CPU times: user 764 us, sys: 35.6 ms, total: 36.4 ms
Wall time: 36.2 ms
```

So if you do continue to use pickle, add the `protocol=2` keyword (thanks @mrocklin for pointing this out).  

For storing python dictionaries of lists, hickle beats the python json encoder, but is slower than uJson. For a dictionary with 64 entries, each containing a 4096 length list of random numbers, the times are:


    json took 2633.263 ms
    uJson took 138.482 ms
    hickle took 232.181 ms


It should be noted that these comparisons are of course not fair: storing in HDF5 will not help you convert something into JSON, nor will it help you serialize a string. But for quick storage of the contents of a python variable, it's a pretty good option.

Installation guidelines (for Linux and Mac OS).
-----------------------------------------------

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

### Testing

Once installed from source, run `python setup.py test` to check it's all working.


Bugs & contributing
--------------------

Contributions and bugfixes are very welcome. Please check out our [contribution guidelines](https://github.com/telegraphic/hickle/blob/master/CONTRIBUTING.md)
for more details on how to contribute to development.


Referencing hickle
------------------

If you use `hickle` in academic research, we would be grateful if you could reference [our paper](http://joss.theoj.org/papers/0c6638f84a1a574913ed7c6dd1051847) in the [Journal of Open-Source Software (JOSS)](http://joss.theoj.org/about).

```
Price et al., (2018). Hickle: A HDF5-based python pickle replacement. Journal of Open Source Software, 3(32), 1115, https://doi.org/10.21105/joss.01115
```
