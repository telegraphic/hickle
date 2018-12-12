# encoding: utf-8
"""
# load_python.py

Handlers for dumping and loading built-in python types.
NB: As these are for built-in types, they are critical to the functioning of hickle.

"""

from hickle.helpers import get_type_and_data

import sys
if sys.version_info.major == 3:
    unicode = type(str)
    str = type(bytes)
    long = type(int)
    NoneType = type(None)
else:
    from types import NoneType

import h5py as h5

def create_listlike_dataset(py_obj, h_group, call_id=0, **kwargs):
    """ Dumper for list, set, tuple

    Args:
        py_obj: python object to dump; should be list-like
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.
    """
    dtype = str(type(py_obj))
    obj = list(py_obj)
    d = h_group.create_dataset('data_%i' % call_id, data=obj, **kwargs)
    d.attrs["type"] = [dtype]


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
    d.attrs["type"] = ['python_dtype']
    d.attrs['python_subdtype'] = str(type(py_obj))


def create_stringlike_dataset(py_obj, h_group, call_id=0, **kwargs):
    """ dumps a list object to h5py file

    Args:
        py_obj: python object to dump; should be string-like (unicode or string)
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.
    """
    if isinstance(py_obj, str):
        d = h_group.create_dataset('data_%i' % call_id, data=[py_obj], **kwargs)
        d.attrs["type"] = ['string']
    else:
        dt = h5.special_dtype(vlen=unicode)
        dset = h_group.create_dataset('data_%i' % call_id, shape=(1, ), dtype=dt, **kwargs)
        dset[0] = py_obj
        dset.attrs['type'] = ['unicode']


def create_none_dataset(py_obj, h_group, call_id=0, **kwargs):
    """ Dump None type to file

    Args:
        py_obj: python object to dump; must be None object
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.
    """
    d = h_group.create_dataset('data_%i' % call_id, data=[0], **kwargs)
    d.attrs["type"] = ['none']


def load_list_dataset(h_node):
    py_type, data = get_type_and_data(h_node)
    return list(data)

def load_tuple_dataset(h_node):
    py_type, data = get_type_and_data(h_node)
    return tuple(data)

def load_set_dataset(h_node):
    py_type, data = get_type_and_data(h_node)
    return set(data)

def load_string_dataset(h_node):
    py_type, data = get_type_and_data(h_node)
    return str(data[0])

def load_unicode_dataset(h_node):
    py_type, data = get_type_and_data(h_node)
    return unicode(data[0])

def load_none_dataset(h_node):
    return None

def load_python_dtype_dataset(h_node):
    py_type, data = get_type_and_data(h_node)
    subtype = h_node.attrs["python_subdtype"]
    type_dict = {
        "<type 'int'>": int,
        "<type 'float'>": float,
        "<type 'long'>": long,
        "<type 'bool'>": bool,
        "<type 'complex'>": complex
    }
    tcast = type_dict.get(subtype)
    return tcast(data)

types_dict = {
    list:        create_listlike_dataset,
    tuple:       create_listlike_dataset,
    set:         create_listlike_dataset,
    str:         create_stringlike_dataset,
    unicode:     create_stringlike_dataset,
    int:         create_python_dtype_dataset,
    float:       create_python_dtype_dataset,
    long:        create_python_dtype_dataset,
    bool:        create_python_dtype_dataset,
    complex:     create_python_dtype_dataset,
    NoneType:    create_none_dataset,
}

hkl_types_dict = {
    "<type 'list'>"  : load_list_dataset,
    "<type 'tuple'>" : load_tuple_dataset,
    "<type 'set'>"   : load_set_dataset,
    "python_dtype"   : load_python_dtype_dataset,
    "string"         : load_string_dataset,
    "unicode"        : load_unicode_dataset,
    "none"           : load_none_dataset
}

