# encoding: utf-8
"""
# load_numpy.py

Utilities and dump / load handlers for handling numpy and scipy arrays

"""
import six
import numpy as np


from hickle.helpers import get_type_and_data


def check_is_numpy_array(py_obj):
    """ Check if a python object is a numpy array (masked or regular)

    Args:
        py_obj: python object to check whether it is a numpy array

    Returns
        is_numpy (bool): Returns True if it is a numpy array, else False if it isn't
    """

    is_numpy = type(py_obj) in (type(np.array([1])), type(np.ma.array([1])))

    return is_numpy


def create_np_scalar_dataset(py_obj, base_type, h_group, call_id=0, **kwargs):
    """ dumps an np dtype object to h5py file

    Args:
        py_obj: python object to dump; should be a numpy scalar, e.g. np.float16(1)
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.
    """

    # DO NOT PASS KWARGS TO SCALAR DATASETS!
    d = h_group.create_dataset('data_%i' % call_id, data=py_obj)  # **kwargs)
    d.attrs['base_type'] = b'np_scalar'

    if six.PY2:
        d.attrs["np_dtype"] = str(d.dtype)
    else:
        d.attrs["np_dtype"] = bytes(str(d.dtype), 'ascii')


def create_np_dtype(py_obj, base_type, h_group, call_id=0, **kwargs):
    """ dumps an np dtype object to h5py file

    Args:
        py_obj: python object to dump; should be a numpy scalar, e.g. np.float16(1)
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.
    """
    d = h_group.create_dataset('data_%i' % call_id, data=[str(py_obj)])
    d.attrs['base_type'] = b'np_dtype'


def create_np_array_dataset(py_obj, base_type, h_group, call_id=0, **kwargs):
    """ dumps an ndarray object to h5py file

    Args:
        py_obj: python object to dump; should be a numpy array or np.ma.array (masked)
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.
    """
    if isinstance(py_obj, type(np.ma.array([1]))):
        d = h_group.create_dataset('data_%i' % call_id, data=py_obj, **kwargs)
        #m = h_group.create_dataset('mask_%i' % call_id, data=py_obj.mask, **kwargs)
        m = h_group.create_dataset('data_%i_mask' % call_id, data=py_obj.mask, **kwargs)
        d.attrs['base_type'] = b'ndarray_masked_data'
        m.attrs['base_type'] = b'ndarray_masked_mask'
    else:
        d = h_group.create_dataset('data_%i' % call_id, data=py_obj, **kwargs)
        d.attrs['base_type'] = b'ndarray'




#######################
## Lookup dictionary ##
#######################

types_dict = {
    np.ndarray:  (create_np_array_dataset, b'ndarray'),
    np.ma.core.MaskedArray: (create_np_array_dataset, b"ndarray_masked_data"),
    np.float16:    (create_np_scalar_dataset, b'np_scalar'),
    np.float32:    (create_np_scalar_dataset, b'np_scalar'),
    np.float64:    (create_np_scalar_dataset, b'np_scalar'),
    np.int8:       (create_np_scalar_dataset, b'np_scalar'),
    np.int16:      (create_np_scalar_dataset, b'np_scalar'),
    np.int32:      (create_np_scalar_dataset, b'np_scalar'),
    np.int64:      (create_np_scalar_dataset, b'np_scalar'),
    np.uint8:      (create_np_scalar_dataset, b'np_scalar'),
    np.uint16:     (create_np_scalar_dataset, b'np_scalar'),
    np.uint32:     (create_np_scalar_dataset, b'np_scalar'),
    np.uint64:     (create_np_scalar_dataset, b'np_scalar'),
    np.complex64:  (create_np_scalar_dataset, b'np_scalar'),
    np.complex128: (create_np_scalar_dataset, b'np_scalar'),
    np.dtype:      (create_np_dtype, b'np_dtype')
}

def load_np_dtype_dataset(h_node):
    py_type, data = get_type_and_data(h_node)
    data = np.dtype(data)
    return data

def load_np_scalar_dataset(h_node):
    py_type, data = get_type_and_data(h_node)
    subtype = h_node.attrs["np_dtype"].decode('utf-8')
    data = getattr(np, subtype)(data)
    return data

def load_ndarray_dataset(h_node):
    py_type, data = get_type_and_data(h_node)
    return np.array(data, copy=False)

def load_ndarray_masked_dataset(h_node):
    py_type, data = get_type_and_data(h_node)
    try:
        mask_path = h_node.name + "_mask"
        h_root = h_node.parent
        mask = h_root.get(mask_path)[:]
    except IndexError:
        mask = h_root.get(mask_path)
    except ValueError:
        mask = h_root.get(mask_path)
    data = np.ma.array(data, mask=mask)
    return data

def load_nothing(h_hode):
    pass

hkl_types_dict = {
    b"np_dtype"            : load_np_dtype_dataset,
    b"np_scalar"           : load_np_scalar_dataset,
    b"ndarray"             : load_ndarray_dataset,
    b"numpy.ndarray"       : load_ndarray_dataset,
    b"ndarray_masked_data" : load_ndarray_masked_dataset,
    b"ndarray_masked_mask" : load_nothing        # Loaded automatically
}


