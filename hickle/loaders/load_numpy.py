# encoding: utf-8
"""
# load_numpy.py

Utilities and dump / load handlers for handling numpy and scipy arrays

"""
import numpy as np
import dill as pickle

from hickle.helpers import get_type_and_data


def check_is_numpy_array(py_obj):
    """ Check if a python object is a numpy array (masked or regular)

    Args:
        py_obj: python object to check whether it is a numpy array

    Returns
        is_numpy (bool): Returns True if it is a numpy array, else False if it isn't
    """

    return(isinstance(py_obj, np.ndarray))

def create_np_scalar_dataset(py_obj, h_group, name, **kwargs):
    """ dumps an np dtype object to h5py file

    Args:
        py_obj: python object to dump; should be a numpy scalar, e.g. np.float16(1)
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.
    """

    # DO NOT PASS KWARGS TO SCALAR DATASETS!
    d = h_group.create_dataset(name, data=py_obj)  # **kwargs)

    d.attrs["np_dtype"] = bytes(str(d.dtype), 'ascii')
    return(d)


def create_np_dtype(py_obj, h_group, name, **kwargs):
    """ dumps an np dtype object to h5py file

    Args:
        py_obj: python object to dump; should be a numpy scalar, e.g. np.float16(1)
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.
    """
    d = h_group.create_dataset(name, data=[str(py_obj)])
    return(d)


def create_np_array_dataset(py_obj, h_group, name, **kwargs):
    """ dumps an ndarray object to h5py file

    Args:
        py_obj: python object to dump; should be a numpy array or np.ma.array (masked)
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.
    """
    if isinstance(py_obj, np.ma.core.MaskedArray):
        d = h_group.create_dataset(name, data=py_obj, **kwargs)
        #m = h_group.create_dataset('mask_%i' % call_id, data=py_obj.mask, **kwargs)
        m = h_group.create_dataset('%s_mask' % name, data=py_obj.mask, **kwargs)
        m.attrs['type'] = np.array(pickle.dumps(py_obj.mask.__class__))
        m.attrs['base_type'] = b'ndarray_masked_mask'
    else:
        d = h_group.create_dataset(name, data=py_obj, **kwargs)
    return(d)




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
    _, _, data = get_type_and_data(h_node)
    data = np.dtype(data)
    return data

def load_np_scalar_dataset(h_node):
    _, _, data = get_type_and_data(h_node)
    subtype = h_node.attrs["np_dtype"].decode('utf-8')
    data = getattr(np, subtype)(data)
    return data

def load_ndarray_dataset(h_node):
    _, _, data = get_type_and_data(h_node)
    return np.array(data, copy=False)

def load_ndarray_masked_dataset(h_node):
    _, _, data = get_type_and_data(h_node)
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
