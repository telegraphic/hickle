"""
#lookup.py

This file contains all the mappings between hickle/HDF5 metadata and python types.
There are four dictionaries and one set that are populated here:

1) types_dict
types_dict: mapping between python types and dataset creation functions, e.g.
    types_dict = {
        list:        create_listlike_dataset,
        int:         create_python_dtype_dataset,
        np.ndarray:  create_np_array_dataset
        }

2) hkl_types_dict
hkl_types_dict: mapping between hickle metadata and dataset loading functions, e.g.
    hkl_types_dict = {
        "<type 'list'>"  : load_list_dataset,
        "<type 'tuple'>" : load_tuple_dataset
        }

3) container_types_dict
container_types_dict: mapping required to convert the PyContainer object in hickle.py
                      back into the required native type. PyContainer is required as
                      some iterable types are immutable (do not have an append() function).
                      Here is an example:
    container_types_dict = {
        "<type 'list'>": list,
        "<type 'tuple'>": tuple
        }

4) container_key_types_dict
container_key_types_dict: mapping specifically for converting hickled dict data back into
                          a dictionary with the same key type. While python dictionary keys
                          can be any hashable object, in HDF5 a unicode/string is required
                          for a dataset name. Example:
    container_key_types_dict = {
        "<type 'str'>": str,
        "<type 'unicode'>": unicode
        }

## Extending hickle to add support for other classes and types

The process to add new load/dump capabilities is as follows:

1) Create a file called load_[newstuff].py in loaders/
2) In the load_[newstuff].py file, define your create_dataset and load_dataset functions,
   along with all required mapping dictionaries.
3) Add an import call here, and populate the lookup dictionaries with update() calls:
    # Add loaders for [newstuff]
    try:
        from .loaders.load_[newstuff[ import types_dict as ns_types_dict
        from .loaders.load_[newstuff[ import hkl_types_dict as ns_hkl_types_dict
        types_dict.update(ns_types_dict)
        hkl_types_dict.update(ns_hkl_types_dict)
        ... (Add container_types_dict etc if required)
    except ImportError:
        raise
"""

from __future__ import absolute_import

from six import PY2
from ast import literal_eval
import numpy as np

def load_nothing(h_node):
    pass

types_dict = {}

hkl_types_dict = {}

types_not_to_sort = [b'dict', b'csr_matrix', b'csc_matrix', b'bsr_matrix']


container_types_dict = {
    b"<type 'list'>": list,
    b"<type 'tuple'>": tuple,
    b"<type 'set'>": set,
    b"<class 'list'>": list,
    b"<class 'tuple'>": tuple,
    b"<class 'set'>": set,
    b"csr_matrix":  None,
    b"csc_matrix": None,
    b"bsr_matrix": None
    }

# Technically, any hashable object can be used, for now sticking with built-in types
container_key_types_dict = {
    b"<type 'str'>": literal_eval,
    b"<type 'float'>": float,
    b"<type 'bool'>": bool,
    b"<type 'int'>": int,
    b"<type 'complex'>": complex,
    b"<type 'tuple'>": literal_eval,
    b"<class 'str'>": literal_eval,
    b"<class 'float'>": float,
    b"<class 'bool'>": bool,
    b"<class 'int'>": int,
    b"<class 'complex'>": complex,
    b"<class 'tuple'>": literal_eval
    }

if PY2:
    container_key_types_dict[b"<type 'unicode'>"] = literal_eval
    container_key_types_dict[b"<type 'long'>"] = long

# Add loaders for built-in python types
if PY2:
    from .loaders.load_python import types_dict as py_types_dict
    from .loaders.load_python import hkl_types_dict as py_hkl_types_dict
else:
    from .loaders.load_python3 import types_dict as py_types_dict
    from .loaders.load_python3 import hkl_types_dict as py_hkl_types_dict

types_dict.update(py_types_dict)
hkl_types_dict.update(py_hkl_types_dict)

from importlib import import_module
# This list holds all loaded loaders
loaded_loaders = []

# Add loaders for numpy types
from .loaders.load_numpy import types_dict as np_types_dict
from .loaders.load_numpy import hkl_types_dict as np_hkl_types_dict
from .loaders.load_numpy import check_is_numpy_array
types_dict.update(np_types_dict)
hkl_types_dict.update(np_hkl_types_dict)
import hickle.loaders
loaded_loaders.append(hickle.loaders.load_numpy)

#######################
## ND-ARRAY checking ##
#######################

ndarray_like_check_fns = {
    np.ndarray: check_is_numpy_array
}


def check_is_ndarray_like(py_obj):
    # Obtain the MRO of this object
    mro_list = py_obj.__class__.mro()

    # Create a function map
    func_map = map(ndarray_like_check_fns.get, mro_list)

    # Loop over the entire func_map until something else than None is found
    for func_item in func_map:
        if func_item is not None:
            return(func_item(py_obj))
    # If that did not happen, then py_obj is not ndarray_like
    else:
        return(False)


#######################
## loading optional  ##
#######################

def register_class(myclass_type, hkl_str, dump_function, load_function,
                   to_sort=True, ndarray_check_fn=None):
    """ Register a new hickle class.

    Args:
        myclass_type type(class): type of class
        hkl_str (str): String to write to HDF5 file to describe class
        dump_function (function def): function to write data to HDF5
        load_function (function def): function to load data from HDF5
        to_sort (bool): If the item is iterable, does it require sorting?
        ndarray_check_fn (function def): function to use to check if

    """
    types_dict.update({myclass_type: (dump_function, hkl_str)})
    hkl_types_dict.update({hkl_str: load_function})
    if not to_sort:
        types_not_to_sort.append(hkl_str)
    if ndarray_check_fn is not None:
        ndarray_like_check_fns[myclass_type] = ndarray_check_fn

def register_class_list(class_list):
    """ Register multiple classes in a list

    Args:
        class_list (list): A list, where each item is an argument to
                           the register_class() function.

    Notes: This just runs the code:
            for item in mylist:
                register_class(*item)
    """
    for class_item in class_list:
        register_class(*class_item)

def register_class_exclude(hkl_str_to_ignore):
    """ Tell loading funciton to ignore any HDF5 dataset with attribute 'type=XYZ'

    Args:
        hkl_str_to_ignore (str): attribute type=string to ignore and exclude from loading.
    """
    hkl_types_dict[hkl_str_to_ignore] = load_nothing

def register_exclude_list(exclude_list):
    """ Ignore HDF5 datasets with attribute type='XYZ' from loading

    ArgsL
        exclude_list (list): List of strings, which correspond to hdf5/hickle
                             type= attributes not to load.
    """
    for hkl_str in exclude_list:
        register_class_exclude(hkl_str)


# This function checks if an additional loader is required for given py_obj
def load_loader(py_obj):
    """
    Checks if given `py_obj` requires an additional loader to be handled
    properly and loads it if so.

    """

    # Obtain the MRO of this object
    if type(py_obj) is type:
        mro_list = py_obj.mro()
    else:
        mro_list = py_obj.__class__.mro()

    # Loop over the entire mro_list
    for mro_item in mro_list:
        # Obtain the package name of mro_item
        pkg_name = mro_item.__module__.split('.')[0]

        # Try to load a loader with this name
        try:
            loader = import_module('hickle.loaders.load_%s' % (pkg_name))
        # If such a loader does not exist, continue
        except (ImportError, NameError):
            pass
        # If such a loader does exist, register classes if not done before
        else:
            # Check if loader had been loaded before
            if loader not in loaded_loaders:
                register_class_list(loader.class_register)
                register_exclude_list(loader.exclude_register)
                loaded_loaders.append(loader)
