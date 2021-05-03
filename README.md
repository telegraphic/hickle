[![PyPI - Latest Release](https://img.shields.io/pypi/v/hickle.svg?logo=pypi&logoColor=white&label=PyPI)](https://pypi.python.org/pypi/hickle)
[![PyPI - Python Versions](https://img.shields.io/pypi/pyversions/hickle.svg?logo=python&logoColor=white&label=Python)](https://pypi.python.org/pypi/hickle)
[![Travis CI - Build Status](https://img.shields.io/travis/com/telegraphic/hickle/master.svg?logo=travis%20ci&logoColor=white&label=Travis%20CI)](https://travis-ci.com/telegraphic/hickle)
[![AppVeyor - Build Status](https://img.shields.io/appveyor/ci/telegraphic/hickle/master.svg?logo=appveyor&logoColor=white&label=AppVeyor)](https://ci.appveyor.com/project/telegraphic/hickle)
[![CodeCov - Coverage Status](https://img.shields.io/codecov/c/github/telegraphic/hickle/master.svg?logo=codecov&logoColor=white&label=Coverage)](https://codecov.io/gh/telegraphic/hickle/branches/master)
[![JOSS Status](http://joss.theoj.org/papers/0c6638f84a1a574913ed7c6dd1051847/status.svg)](http://joss.theoj.org/papers/0c6638f84a1a574913ed7c6dd1051847)


Hickle
======

Hickle is an [HDF5](https://www.hdfgroup.org/solutions/hdf5/) based clone of `pickle`, with a twist: instead of serializing to a pickle file,
Hickle dumps to an HDF5 file (Hierarchical Data Format). It is designed to be a "drop-in" replacement for pickle (for common data objects), but is
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

Dumping custom objects
----------------------
Hickle provides several options to store objects of custom python classes. Objects of classes derived
from built in classes, numpy, scipy, pandas and astropy objects will be stored using the corresponding 
loader provided by hickle. Any other classes per default will be stored as binary pickle string.
Starting with version 4.x hickle offers the possibility to define dedicated loader functions for custom
classes and starting with hickle 5.x these can be collected in module, package and application specific
loader modules. 

```
class MyClass():
    def __init__(self):
        self.name = 'MyClass'
        self.value = 42
```

To create a loader for `MyClass` the `create_MyClass_dataset` and either the `load_MyClass` or the 
`MyClassContainer` class have to be defined. 

```
import hdf5
form hickle.helpters import no_compression


def create_MyClass_dataset(py_obj, h_group, name, **kwargs):
    """ 
    py_obj ..... the instance of MyClass to be dumped
    h_group .... the h5py.Group py_obj should be dumped into
    name ....... the name of the h5py.Dataset or h5py.Group representing py_obj
    **kwargs ... the compression keyword arguments passed to hickle.dump

    # if content of MyClass can be represented as single matrix, vector or scalar
    # values than created a dataset of approriate size. and either set its shape and 
    # dtype parameters # to the approriate size and tyoe . or directly pass the data
    # using the data parmameter
    ds = h_group.create_dataset(name,data = py_obj.value,**kwargs)

	## NOTE: if your class represents a scalar using empty tuple for shape
    ##       than kwargs have to be filtered by no_compression
    # ds = h_group.create_dataset(name,data = py_obj.value,shape=(),**no_compression(kwargs))

	# set addtional attributes providing additional specialisation of content
    ds.attrs['name'] = py_obj.name

    # when done return the new dataset object and an empty tuple or list
    return ds,()

def load_Myclass(h_node, base_type, py_obj_type):
    """
    h_node ........ the h5py.Dataset object containing the data of MyClass object to restore
    base_type ..... byte string naming the loader to be used for restoring MyClass object
    py_obj_type ... MyClass class or MyClass subclass object 
    """

    # py_obj_type should point to MyClass or any of its subclasses
    new_instance = py_obj_type()
    new_instance.name = h_node.attrs['name']
    new_instance.value = h_node[()]
    return new_instance
```

For dumping content of complex objects consisting of multiple sub-items which have to be
stored as individual h5py.Dataset or h5py.Group objects than define `create_MyClass_dataset`
using `create_group` method instead of `create_dataset` and define the corresponding
`MyClassContainer` class.

```
import h5py
from hickle.helpers import PyContainer

def create_MyClass_dataset(py_obj, h_group, name, **kwargs):
    """ 
    py_obj ..... the instance of MyClass to be dumped
    h_group .... the h5py.Group py_obj should be dumped into
    name ....... the name of the h5py.Dataset or h5py.Group representing py_obj
    **kwargs ... the compression keyword arguments passed to hickle.dump

    ds = h_group.create_group(name)

	# set addtional attributes providing additional specialisation of content
    ds.attrs['name'] = py_obj.name

    # when done return the new dataset object and a tuple, list or generator function
	# providing for all subitems a tuple or list describing containgin 
    #  name ..... the name to be used storing the subitem within the h5py.Group object
    #  item ..... the subitem object to be stored
    #  attrs .... dictionary included in attrs of created h5py.Group or h5py.Dataset
    #  kwargs ... the kwargs as passed to create_MyClass_dataset function
    return ds,(('name',py_obj.name,{},kwargs),('value',py_obj.value,{'the answer':True},kwargs))



class MyClassContainer(PyContainer):
    """
    Valid container classes must be derived from hickle.helpers.PyContainer class
    """

    def __init__(self,h5_attrs,base_type,object_type):
		"""
		h5_attrs ...... the attrs dictionary attached to the group representing MyClass
    	base_type ..... byte string naming the loader to be used for restoring MyClass object
    	py_obj_type ... MyClass class or MyClass subclass object 
		"""

		# the optional protected _content parameter of the PyContainer __init__
        # method can be used to change the data structure used to store
        # the subitems passed to the append method of the PyContainer class
        # per default it is set to []
        super().__init__(h5_attrs,base_type,object_type,_content = dict())

	def filter(self,h_parent): # optional overload
        """
		generator member functoin which can be overloaded to reorganize subitems
        of h_parent h5py.Group before beeing restored by hickle. Its default
        implementation simply yields from h_parent.items(). 
		"""
		yield from super().filter(h_parent)

	def append(self,name,item,h5_attrs): # optional overload
        """
		in case _content parameter was explicitly set or subitems should be sored 
        in specific order or have to be preprocessed before the next item is appended
        than this can be done before storing in self._content.

        name ....... the name identifying subitem item within the parent hdf5.Group
        item ....... the object representing the subitem
        h5_attrs ... attrs dictionary attached to h5py.Dataset, h5py.Group representing item
		"""
		self._content[name] = item

	def convert(self):
        """
		called by hickle when all sub items have been appended to MyClass PyContainer
		this method must be implemented by MyClass PyContainer.
		"""

    	# py_obj_type should point to MyClass or any of its subclasses
    	new_instance = py_obj_type()
		new_instance.__dict__.update(self._content)
		return new_instance
```

In a last step the loader for MyClass has to be registered with hickle. This is done by calling
`hickle.lookup.LoaderManager.register_class` method

```
from hickle.lookup import LoaderManager

# to register loader for object mapped to h5py.Dataset use
LoaderManager.register_class(
   MyClass,                # MyClass type object this loader handles
   b'MyClass',             # byte string representing the name of the loader 
   create_MyClass_Dataset, # the create dataset function defined in first example above
   load_MyClass,           # the load dataset function defined in first example above
   None,                   # usually None
   True,                   # Set to False to force explcit storage of MyClass instances in any case 
   'custom'                # Loader is only used when custom loaders are enabled on calling hickle.dump
)

# to register loader for object mapped to h5py.Group use
LoaderManager.register_class(
   MyClass,                # MyClass type object this loader handles
   b'MyClass',             # byte string representing the name of the loader 
   create_MyClass_Dataset, # the create dataset function defined in first example above
   None,                   # usually None
   MyClassContainer        # the PyContainer to be used to restore content of MyClass
   True,                   # Set to False to force explcit storage of MyClass instances in any case 
   None                    # if set to None loader is enabled unconditionally
)

# NOTE: in case content of MyClass instances may be mapped to h5py.Dataset or h5py.Group dependent upon
# their actual complexity than both types of loaders can be merged into one single
# using one common create_MyClass_dataset functoin and defining load_MyClass function and
# MyClassContainer class
```

For complex python modules, packages and applications defining several classes to be dumped and handled by 
hickle calling `hickle.lookup.LoaderManager.register_class` explicitly very quickly becomes tedious and
confusing. Therefore hickle offers from hickle 5.x on the possibility to collect all loaders for classes
and objects defined by your module, package or application within dedicated loader modules and install
them along with your module, package and application.

For packages and application packages the `load_MyPackage.py` loader module has to be stored within
`hickle_loaders` directory of the package directory (the first which contains a __init__.py file) and
should be structured as follows.

```
from hickle.helpers import PyContainer

## define below all create_MyClass_dataset load_MyClass functions and MyClassContainer classes
## of the loaders serving your module, package, application package or application

....

## the class_register table and the exclude_register table are required
## by hickle to properly load and apply your loaders
## each row in the class register table will corresponds to the parameters
## of LoaderManager.register_class and has to be specified in the same order
## as above

class_register = [
   [ MyClass,                # MyClass type object this loader handles
     b'MyClass',             # byte string representing the name of the loader 
     create_MyClass_Dataset, # the create dataset function defined in first example above
     load_MyClass,           # the load dataset function defined in first example above
     None,                   # usually None
     True,                   # Set to False to force explcit storage of MyClass instances in any case 
     'custom'                # Loader is only used when custom loaders are enabled on calling hickle.dump
   ],
   [ MyClass,                # MyClass type object this loader handles
     b'MyClass',             # byte string representing the name of the loader 
     create_MyClass_Dataset, # the create dataset function defined in first example above
     None,                   # usually None
     MyClassContainer        # the PyContainer to be used to restore content of MyClass
     True,                   # Set to False to force explcit storage of MyClass instances in any case 
     None                    # if set to None loader is enabled unconditionally
   ]
]

# used by hickle 4.x legacy loaders and other special loaders
# usually an empty list
exclude_register = []
```

For single file modules and application scripts the `load_MyModule.py` or `load_MyApp.py` files have to
be stored within the `hickle_loaders` directory located within the same directory as `MyModule.py` or
`My_App.py`. For further examples of more complex loaders and on how to store bytearrays and strings
such that they can be compressed when stored see default loader modules in  `hickle/loaders/` directory.


### Note: storing complex objects in HDF5 file
The HDF5 file format is designed to store several big matrices, images and vectors efficiently
and attache some metadata and to provide a convenient way access the data through a tree structure.
It is not designed like python pickle format for efficiently mapping the in memory object structure
to a file. Therefore mindlessly storing plenty of tiny objects and scalar values without combining
them into a single datataset will cause the HDF5 and thus the file created by hickle explode. File
sizes of several 10 GB are likely and possible when a pickle file would just need some 100 MB.
This can be prevented by `create_MyClass_dataset` method combining sub-items into bigger numpy arrays
or other data structures which can be mapped to `h5py.Datasets` and `load_MyClass` function and /or 
`MyClassContainer.convert` method restoring actual structure of the sub-items on load.

Recent changes
--------------

* June 2020: Major refactor to version 4, and removal of support for Python 2.
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

Installation guidelines
-----------------------

### Easy method
Install with `pip` by running `pip install hickle` from the command line.

#### Install on Windows 32 bit

Prebuilt Python wheels packages are available on PyPi until H5PY version 2.10 and Python 3.8.
Any newer versions have to be built and installed Manually.

1) Install h5py 2.10 with `pip` by running `pip install "h5py==2.10"` from the commandline

2) Install with `pip` by running `pip install hickle` form the command line

### Manual install

1. You should have Python 3.5 and above installed

2. Install hdf5
(Official page: http://www.hdfgroup.org/ftp/HDF5/current/src/unpacked/release_docs/INSTALL)
(Binary Downloads: https://portal.hdfgroup.org/display/support/Downloads)
__Note:__ On Windows 32 bit install prebuilt binary package for libhdf5 [1.10.4](https://portal.hdfgroup.org/display/support/HDF5+1.10.4), which is the latest version supporting 32 bit on Windows

3. Install h5py
(Official page: http://docs.h5py.org/en/latest/build.html)

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
