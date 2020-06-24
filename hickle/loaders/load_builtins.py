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
    str_type = None
    if type(obj[0]) in (str, bytes):
        str_type = bytes(type(obj[0]).__name__, 'ascii')

    if type(obj[0]) is str:
        obj = [bytes(oo, 'utf8') for oo in obj]

    d = h_group.create_dataset(name, data=obj, **kwargs)

    # Need to add some metadata to aid in unpickling if it's a string type
    if str_type is not None:
        d.attrs["str_type"] = str_type
    return(d)


def create_scalar_dataset(py_obj, h_group, name, **kwargs):
    """ dumps a python dtype object to h5py file

    Args:
        py_obj: python object to dump; should be a scalar (int, float,
            bool, str, etc)
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the
            iterable.
    """

    # Make sure 'compression' is not in kwargs
    kwargs.pop('compression', None)

    # If py_obj is an integer and cannot be stored in 64-bits, convert to str
    if isinstance(py_obj, int) and (py_obj.bit_length() > 64):
        py_obj = bytes(str(py_obj), 'ascii')

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
    d = h_group.create_dataset(name, data=b'None', **kwargs)
    return(d)


def create_pickled_dataset(py_obj, h_group, name, reason=None, **kwargs):
    """ If no match is made, raise a warning

    Args:
        py_obj: python object to dump; default if item is not matched.
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the
            iterable.
    """
    reason_str = " (Reason: %s)" % (reason) if reason is not None else ""
    pickled_obj = pickle.dumps(py_obj)
    d = h_group.create_dataset(name, data=np.array(pickled_obj), **kwargs)

    warnings.warn("%r type not understood, data has been serialized%s"
                  % (py_obj.__class__.__name__, reason_str), SerializedWarning)
    return(d)


def load_list_dataset(h_node):
    _, _, data = get_type_and_data(h_node)
    str_type = h_node.attrs.get('str_type', None)

    if str_type == b'str':
        return(np.array(data, copy=False, dtype=str).tolist())
    else:
        return(data.tolist())


def load_tuple_dataset(h_node):
    data = load_list_dataset(h_node)
    return tuple(data)


def load_set_dataset(h_node):
    data = load_list_dataset(h_node)
    return set(data)


def load_none_dataset(h_node):
    return None


def load_pickled_data(h_node):
    _, _, data = get_type_and_data(h_node)
    return pickle.loads(data)


def load_scalar_dataset(h_node):
    _, base_type, data = get_type_and_data(h_node)

    if(base_type == b'int'):
        data = int(data)

    return(data)


# %% REGISTERS
class_register = [
    [list, b"list", create_listlike_dataset, load_list_dataset],
    [tuple, b"tuple", create_listlike_dataset, load_tuple_dataset],
    [set, b"set", create_listlike_dataset, load_set_dataset],
    [bytes, b"bytes", create_scalar_dataset, load_scalar_dataset],
    [str, b"str", create_scalar_dataset, load_scalar_dataset],
    [int, b"int", create_scalar_dataset, load_scalar_dataset],
    [float, b"float", create_scalar_dataset, load_scalar_dataset],
    [complex, b"complex", create_scalar_dataset, load_scalar_dataset],
    [bool, b"bool", create_scalar_dataset, load_scalar_dataset],
    [type(None), b"None", create_none_dataset, load_none_dataset],
    [object, b"pickle", create_pickled_dataset, load_pickled_data]]

exclude_register = []
