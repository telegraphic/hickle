# encoding: utf-8
"""
# load_python.py

Handlers for dumping and loading built-in python types.
NB: As these are for built-in types, they are critical to the functioning of hickle.

"""

import six
from hickle.helpers import get_type_and_data

try:
    from exceptions import Exception
except ImportError:
    pass        # above imports will fail in python3

try:
    ModuleNotFoundError  # This fails on Py3.5 and below
except NameError:
    ModuleNotFoundError = ImportError

import h5py as h5


def get_py3_string_type(h_node):
    """ Helper function to return the python string type for items in a list.

    Notes:
        Py3 string handling is a bit funky and doesn't play too nicely with HDF5.
        We needed to add metadata to say if the strings in a list started off as
        bytes, string, etc. This helper loads

    """
    try:
        py_type = h_node.attrs["py3_string_type"]
        return py_type
    except:
        return None

def create_listlike_dataset(py_obj, h_group, name, **kwargs):
    """ Dumper for list, set, tuple

    Args:
        py_obj: python object to dump; should be list-like
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.
    """
    obj = list(py_obj)

    # h5py does not handle Py3 'str' objects well. Need to catch this
    # Only need to check first element as this method
    # is only called if all elements have same dtype
    py3_str_type = None
    if type(obj[0]) in (str, bytes):
        py3_str_type = bytes(str(type(obj[0])), 'ascii')

    if type(obj[0]) is str:
        #print(py3_str_type)
        #print(obj, "HERE")
        obj = [bytes(oo, 'utf8') for oo in obj]
        #print(obj, "HERE")


    d = h_group.create_dataset(name, data=obj, **kwargs)

    # Need to add some metadata to aid in unpickling if it's a string type
    if py3_str_type is not None:
        d.attrs["py3_string_type"] = py3_str_type
    return(d)



def create_python_dtype_dataset(py_obj, h_group, name, **kwargs):
    """ dumps a python dtype object to h5py file

    Args:
        py_obj: python object to dump; should be a python type (int, float, bool etc)
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.
    """

    # Determine the subdtype of the given py_obj
    subdtype = bytes(str(type(py_obj)), 'ascii')

    # If py_obj is an integer and cannot be stored in 64-bits, convert to str
    if isinstance(py_obj, int) and (py_obj.bit_length() > 64):
        py_obj = str(py_obj)

    # kwarg compression etc does not work on scalars
    d = h_group.create_dataset(name, data=py_obj)     #, **kwargs)
    d.attrs['python_subdtype'] = subdtype
    return(d)


def create_stringlike_dataset(py_obj, h_group, name, **kwargs):
    """ dumps a list object to h5py file

    Args:
        py_obj: python object to dump; should be string-like (unicode or string)
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.
    """
    d = h_group.create_dataset(name, data=py_obj, **kwargs)
    return(d)

def create_none_dataset(py_obj, h_group, name, **kwargs):
    """ Dump None type to file

    Args:
        py_obj: python object to dump; must be None object
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.
    """
    d = h_group.create_dataset(name, data=[0], **kwargs)
    return(d)


def load_list_dataset(h_node):
    _, _, data = get_type_and_data(h_node)
    py3_str_type = get_py3_string_type(h_node)

    if py3_str_type == b"<class 'bytes'>":
        # Yuck. Convert numpy._bytes -> str -> bytes
        return [bytes(str(item, 'utf8'), 'utf8') for item in data]
    if py3_str_type == b"<class 'str'>":
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

def load_unicode_dataset(h_node):
    _, _, data = get_type_and_data(h_node)
    return unicode(data)

def load_none_dataset(h_node):
    return None

def load_pickled_data(h_node):
    _, _, data = get_type_and_data(h_node)
    try:
        import cPickle as pickle
    except ModuleNotFoundError:
        import pickle
    return pickle.loads(data)


def load_python_dtype_dataset(h_node):
    _, _, data = get_type_and_data(h_node)
    subtype = h_node.attrs["python_subdtype"]
    type_dict = {
        b"<class 'int'>": int,
        b"<class 'float'>": float,
        b"<class 'bool'>": bool,
        b"<class 'complex'>": complex
    }

    tcast = type_dict.get(subtype)
    return tcast(data)



types_dict = {
    list:        (create_listlike_dataset, b"<class 'list'>"),
    tuple:       (create_listlike_dataset, b"<class 'tuple'>"),
    set:         (create_listlike_dataset, b"<class 'set'>"),
    bytes:       (create_stringlike_dataset, b"bytes"),
    str:         (create_stringlike_dataset, b"string"),
    # bytearray:   (create_stringlike_dataset, b"bytes"),
    int:         (create_python_dtype_dataset, b"python_dtype"),
    float:       (create_python_dtype_dataset, b"python_dtype"),
    bool:        (create_python_dtype_dataset, b"python_dtype"),
    complex:     (create_python_dtype_dataset, b"python_dtype"),
    type(None):  (create_none_dataset, b"none"),
}

hkl_types_dict = {
    b"<class 'list'>"  : load_list_dataset,
    b"<class 'tuple'>" : load_tuple_dataset,
    b"<class 'set'>"   : load_set_dataset,
    b"bytes"           : load_bytes_dataset,
    b"python_dtype"   : load_python_dtype_dataset,
    b"string"         : load_string_dataset,
    b"pickle"         : load_pickled_data,
    b"none"           : load_none_dataset,
}
