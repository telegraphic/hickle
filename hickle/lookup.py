"""
#lookup.py

This file contains all the mappings between hickle/HDF5 metadata and python
types.
There are three dictionaries that are populated here:

1) types_dict
Mapping between python types and dataset creation functions, e.g.
    types_dict = {
        list: (create_listlike_dataset, 'list'),
        int: (create_python_dtype_dataset, 'int'),
        np.ndarray: (create_np_array_dataset, 'ndarray'),
        }

2) hkl_types_dict
Mapping between hickle metadata and dataset loading functions, e.g.
    hkl_types_dict = {
        'list': load_list_dataset,
        'tuple': load_tuple_dataset
        }

3) dict_key_types_dict
Mapping specifically for converting hickled dict data back into a dictionary
with the same key type. While python dictionary keys can be any hashable
object, in HDF5 a unicode/string is required for a dataset name.

Example:
    dict_key_types_dict = {
        "<class 'str'>": literal_eval,
        "<class 'float'>": float
        }

## Extending hickle to add support for other classes and types

The process to add new load/dump capabilities is as follows:

1) Create a file called load_[newstuff].py in loaders/
2) In the load_[newstuff].py file, define your create_dataset and load_dataset
   functions, along with the 'class_register' and 'exclude_register' lists.

"""


# %% IMPORTS
# Built-in imports
from ast import literal_eval
from importlib import import_module
from inspect import isclass
from itertools import starmap


# %% GLOBALS
# Define dict of all acceptable types
types_dict = {}

# Define dict of all acceptable hickle types
hkl_types_dict = {}

# Define list of types that should never be sorted
types_not_to_sort = []

# Empty list of loaded loader names
loaded_loaders = []

# Define dict containing validation functions for ndarray-like objects
ndarray_like_check_fns = {}

# Define conversion dict of all acceptable dict key types
dict_key_types_dict = {
    b'str': literal_eval,
    b'float': float,
    b'bool': bool,
    b'int': int,
    b'complex': complex,
    b'tuple': literal_eval,
    b'NoneType': literal_eval,
    }


# %% FUNCTION DEFINITIONS
def load_nothing(h_node):
    pass


#####################
# ND-ARRAY checking #
#####################

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


#####################
# loading optional  #
#####################

# This function registers a class to be used by hickle
def register_class(myclass_type, hkl_str, dump_function, load_function,
                   ndarray_check_fn=None, to_sort=True):
    """ Register a new hickle class.

    Args:
        myclass_type type(class): type of class
        hkl_str (str): String to write to HDF5 file to describe class
        dump_function (function def): function to write data to HDF5
        load_function (function def): function to load data from HDF5
        ndarray_check_fn (function def): function to use to check if
        to_sort (bool): If the item is iterable, does it require sorting?

    """
    types_dict[myclass_type] = (dump_function, hkl_str)
    hkl_types_dict[hkl_str] = load_function
    if not to_sort:
        types_not_to_sort.append(hkl_str)
    if ndarray_check_fn is not None:
        ndarray_like_check_fns[myclass_type] = ndarray_check_fn


def register_class_exclude(hkl_str_to_ignore):
    """ Tell loading funciton to ignore any HDF5 dataset with attribute
    'type=XYZ'

    Args:
        hkl_str_to_ignore (str): attribute type=string to ignore and exclude
            from loading.
    """
    hkl_types_dict[hkl_str_to_ignore] = load_nothing


# This function checks if an additional loader is required for given py_obj
def load_loader(py_obj):
    """
    Checks if given `py_obj` requires an additional loader to be handled
    properly and loads it if so.

    """

    # Obtain the MRO of this object
    if isclass(py_obj):
        mro_list = py_obj.mro()
    else:
        mro_list = py_obj.__class__.mro()

    # Loop over the entire mro_list
    for mro_item in mro_list:
        # Check if mro_item can be found in types_dict and return if so
        if mro_item in types_dict:
            return

        # Obtain the package name of mro_item
        pkg_name = mro_item.__module__.split('.')[0]

        # Obtain the name of the associated loader
        loader_name = 'hickle.loaders.load_%s' % (pkg_name)

        # Check if this module is already loaded, and return if so
        if loader_name in loaded_loaders:
            return

        # Try to load a loader with this name
        try:
            loader = import_module(loader_name)
        # If any module is not found, catch error and check it
        except ImportError as error:
            # Check if the error was due to a package in loader not being found
            if 'hickle' not in error.args[0]:   # pragma: no cover
                # If so, reraise the error
                raise
        # If such a loader does exist, register classes and return
        else:
            list(starmap(register_class, loader.class_register))
            list(map(register_class_exclude, loader.exclude_register))
            loaded_loaders.append(loader_name)
            return
