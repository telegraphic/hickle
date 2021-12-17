# encoding: utf-8
"""
# load_python.py

Handlers for dumping and loading built-in python types.
NB: As these are for built-in types, they are critical to the functioning of hickle.

"""

import six
from ..helpers import get_type_and_data

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
        py_type = h_node.attrs["py3_string_type"][0]
        return py_type
    except:
        return None

def create_listlike_dataset(py_obj, h_group, call_id=0, **kwargs):
    """ Dumper for list, set, tuple

    Args:
        py_obj: python object to dump; should be list-like
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.
    """
    dtype = str(type(py_obj))
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


    d = h_group.create_dataset('data_%i' % call_id, data=obj, **kwargs)
    d.attrs["type"] = [bytes(dtype, 'ascii')]

    # Need to add some metadata to aid in unpickling if it's a string type
    if py3_str_type is not None:
        d.attrs["py3_string_type"] = [py3_str_type]



def create_python_dtype_dataset(py_obj, h_group, call_id=0, **kwargs):
    """ dumps a python dtype object to h5py file

    Args:
        py_obj: python object to dump; should be a python type (int, float, bool etc)
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.
    """
    # kwarg compression etc does not work on scalars
    d = h_group.create_dataset('data_%i' % call_id, data=py_obj,
                               dtype=type(py_obj))     #, **kwargs)
    d.attrs["type"] = [b'python_dtype']
    d.attrs['python_subdtype'] = bytes(str(type(py_obj)), 'ascii')


def create_stringlike_dataset(py_obj, h_group, call_id=0, **kwargs):
    """ dumps a list object to h5py file

    Args:
        py_obj: python object to dump; should be string-like (unicode or string)
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.
    """
    if isinstance(py_obj, bytes):
        d = h_group.create_dataset('data_%i' % call_id, data=[py_obj], **kwargs)
        d.attrs["type"] = [b'bytes']
    elif isinstance(py_obj, str):
        dt = h5.special_dtype(vlen=str)
        dset = h_group.create_dataset('data_%i' % call_id, shape=(1, ), dtype=dt, **kwargs)
        dset[0] = py_obj
        dset.attrs['type'] = [b'string']

def create_none_dataset(py_obj, h_group, call_id=0, **kwargs):
    """ Dump None type to file

    Args:
        py_obj: python object to dump; must be None object
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.
    """
    d = h_group.create_dataset('data_%i' % call_id, data=[0], **kwargs)
    d.attrs["type"] = [b'none']


def load_list_dataset(h_node):
    py_type, data = get_type_and_data(h_node)
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
    py_type, data = get_type_and_data(h_node)
    return bytes(data[0])

def load_string_dataset(h_node):
    py_type, data = get_type_and_data(h_node)
    return str(data[0])

def load_unicode_dataset(h_node):
    py_type, data = get_type_and_data(h_node)
    return unicode(data[0])

def load_none_dataset(h_node):
    return None

def load_pickled_data(h_node):
    py_type, data = get_type_and_data(h_node)
    import dill as pickle
    return pickle.loads(data[0])


def load_python_dtype_dataset(h_node):
    py_type, data = get_type_and_data(h_node)
    subtype = h_node.attrs["python_subdtype"]
    type_dict = {
        b"<class 'int'>": int,
        b"<class 'float'>": float,
        b"<class 'bool'>": bool,
        b"<class 'complex'>": complex,
        "<class 'int'>": int,
        "<class 'float'>": float,
        "<class 'bool'>": bool,
        "<class 'complex'>": complex
    }

    tcast = type_dict.get(subtype)
    return tcast(data)



types_dict = {
    list:        create_listlike_dataset,
    tuple:       create_listlike_dataset,
    set:         create_listlike_dataset,
    bytes:         create_stringlike_dataset,
    str:           create_stringlike_dataset,
    #bytearray:     create_stringlike_dataset,
    int:         create_python_dtype_dataset,
    float:       create_python_dtype_dataset,
    bool:        create_python_dtype_dataset,
    complex:     create_python_dtype_dataset,
    type(None):    create_none_dataset,
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
