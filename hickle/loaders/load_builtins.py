# encoding: utf-8
"""
# load_python.py

Handlers for dumping and loading built-in python types.
NB: As these are for built-in types, they are critical to the functioning of
hickle.

"""


# %% IMPORTS
# Built-in imports
import warnings

# Package imports
import dill as pickle
import numpy as np

# hickle imports
from hickle.helpers import get_type_and_data


# %% CLASS DEFINITIONS
class SerializedWarning(UserWarning):
    """ An object type was not understood

    The data will be serialized using pickle.
    """
    pass


# %% FUNCTION DEFINITIONS
def get_py3_string_type(h_node):
    """ Helper function to return the python string type for items in a list.

    Notes:
        Py3 string handling is a bit funky and doesn't play too nicely with
        HDF5.
        We needed to add metadata to say if the strings in a list started off
        as bytes, string, etc. This helper loads

    """
    try:
        py_type = h_node.attrs["py3_string_type"]
        return py_type
    except Exception:
        return None


def create_listlike_dataset(py_obj, h_group, name, **kwargs):
    """ Dumper for list, set, tuple

    Args:
        py_obj: python object to dump; should be list-like
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the
            iterable.
    """

    obj = list(py_obj)

    # h5py does not handle Py3 'str' objects well. Need to catch this
    # Only need to check first element as this method
    # is only called if all elements have same dtype
    py3_str_type = None
    if type(obj[0]) in (str, bytes):
        py3_str_type = bytes(type(obj[0]).__name__, 'ascii')

    if type(obj[0]) is str:
        obj = [bytes(oo, 'utf8') for oo in obj]

    d = h_group.create_dataset(name, data=obj, **kwargs)

    # Need to add some metadata to aid in unpickling if it's a string type
    if py3_str_type is not None:
        d.attrs["py3_string_type"] = py3_str_type
    return(d)


def create_python_dtype_dataset(py_obj, h_group, name, **kwargs):
    """ dumps a python dtype object to h5py file

    Args:
        py_obj: python object to dump; should be a python type (int, float,
            bool etc)
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the
            iterable.
    """

    # Determine the subdtype of the given py_obj
    subdtype = bytes(str(type(py_obj)), 'ascii')

    # If py_obj is an integer and cannot be stored in 64-bits, convert to str
    if isinstance(py_obj, int) and (py_obj.bit_length() > 64):
        py_obj = bytes(str(py_obj), 'ascii')

    # kwarg compression etc does not work on scalars
    d = h_group.create_dataset(name, data=py_obj)
    d.attrs['python_subdtype'] = subdtype
    return(d)


def create_stringlike_dataset(py_obj, h_group, name, **kwargs):
    """ dumps a list object to h5py file

    Args:
        py_obj: python object to dump; should be string-like (unicode or
            string)
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the
            iterable.
    """
    d = h_group.create_dataset(name, data=py_obj, **kwargs)
    return(d)


def create_none_dataset(py_obj, h_group, name, **kwargs):
    """ Dump None type to file

    Args:
        py_obj: python object to dump; must be None object
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the
            iterable.
    """
    d = h_group.create_dataset(name, data=[0], **kwargs)
    return(d)


def create_pickled_dataset(py_obj, h_group, name, **kwargs):
    """ If no match is made, raise a warning

    Args:
        py_obj: python object to dump; default if item is not matched.
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the
            iterable.
    """
    pickled_obj = pickle.dumps(py_obj)
    d = h_group.create_dataset(name, data=np.array(pickled_obj))

    warnings.warn("%s type not understood, data have been serialized"
                  % (type(py_obj)), SerializedWarning)
    return(d)


def load_list_dataset(h_node):
    _, _, data = get_type_and_data(h_node)
    py3_str_type = get_py3_string_type(h_node)

    if py3_str_type == b'bytes':
        # Yuck. Convert numpy._bytes -> str -> bytes
        return [bytes(str(item, 'utf8'), 'utf8') for item in data]
    if py3_str_type == b'str':
        return [str(item, 'utf8') for item in data]
    else:
        return list(data)


def load_tuple_dataset(h_node):
    data = load_list_dataset(h_node)
    return tuple(data)


def load_set_dataset(h_node):
    data = load_list_dataset(h_node)
    return set(data)


def load_bytes_dataset(h_node):
    _, _, data = get_type_and_data(h_node)
    return bytes(data)


def load_string_dataset(h_node):
    _, _, data = get_type_and_data(h_node)
    return str(data)


def load_none_dataset(h_node):
    return None


def load_pickled_data(h_node):
    _, _, data = get_type_and_data(h_node)
    return pickle.loads(data)


def load_python_dtype_dataset(h_node):
    _, _, data = get_type_and_data(h_node)
    subtype = h_node.attrs["python_subdtype"]
    type_dict = {
        b"<class 'int'>": int,
        b"<class 'float'>": float,
        b"<class 'bool'>": bool,
        b"<class 'complex'>": complex,
    }

    tcast = type_dict.get(subtype)
    return tcast(data)


# %% REGISTERS
class_register = [
    [list, b"list", create_listlike_dataset, load_list_dataset],
    [tuple, b"tuple", create_listlike_dataset, load_tuple_dataset],
    [set, b"set", create_listlike_dataset, load_set_dataset],
    [bytes, b"bytes", create_stringlike_dataset, load_bytes_dataset],
    [str, b"string", create_stringlike_dataset, load_string_dataset],
    [int, b"python_dtype", create_python_dtype_dataset,
     load_python_dtype_dataset],
    [float, b"python_dtype", create_python_dtype_dataset,
     load_python_dtype_dataset],
    [complex, b"python_dtype", create_python_dtype_dataset,
     load_python_dtype_dataset],
    [bool, b"python_dtype", create_python_dtype_dataset,
     load_python_dtype_dataset],
    [type(None), b"none", create_none_dataset, load_none_dataset],
    [object, b"pickle", create_pickled_dataset, load_pickled_data]]

exclude_register = []
