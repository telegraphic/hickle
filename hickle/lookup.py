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

5) types_not_to_sort
type_not_to_sort is a list of hickle type attributes that may be hierarchical,
but don't require sorting by integer index.

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

def return_first(x):
    """ Return first element of a list """
    return x[0]

types_dict = {}

hkl_types_dict = {}

types_not_to_sort = {'dict', 'csr_matrix', 'csc_matrix', 'bsr_matrix'}

container_types_dict = {
    "<type 'list'>": list,
    "<type 'tuple'>": tuple,
    "<type 'set'>": set,
    "csr_matrix":  return_first,
    "csc_matrix": return_first,
    "bsr_matrix": return_first
    }

# Technically, any hashable object can be used, for now sticking with built-in types
container_key_types_dict = {
    "<type 'str'>": str,
    "<type 'unicode'>": unicode,
    "<type 'float'>": float,
    "<type 'bool'>": bool,
    "<type 'int'>": int,
    "<type 'long'>": long,
    "<type 'complex'>": complex
    }

# Add loaders for built-in python types
from .loaders.load_python import types_dict as py_types_dict
from .loaders.load_python import hkl_types_dict as py_hkl_types_dict

types_dict.update(py_types_dict)
hkl_types_dict.update(py_hkl_types_dict)

# Add loaders for numpy types
from .loaders.load_numpy import  types_dict as np_types_dict
from .loaders.load_numpy import  hkl_types_dict as np_hkl_types_dict
types_dict.update(np_types_dict)
hkl_types_dict.update(np_hkl_types_dict)

# Add loaders for astropy
try:
    from .loaders.load_astropy import types_dict as ap_types_dict
    from .loaders.load_astropy import hkl_types_dict as ap_hkl_types_dict
    types_dict.update(ap_types_dict)
    hkl_types_dict.update(ap_hkl_types_dict)

except ImportError:
    raise

