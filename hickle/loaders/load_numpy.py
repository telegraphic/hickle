# encoding: utf-8
"""
# load_numpy.py

Utilities and dump / load handlers for handling numpy and scipy arrays

"""
import six
import numpy as np

try:
    import scipy
    from scipy import sparse
    _has_scipy = True
except ImportError:
    _has_scipy = False

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


def create_np_scalar_dataset(py_obj, h_group, call_id=0, **kwargs):
    """ dumps an np dtype object to h5py file

    Args:
        py_obj: python object to dump; should be a numpy scalar, e.g. np.float16(1)
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.
    """

    # DO NOT PASS KWARGS TO SCALAR DATASETS!
    d = h_group.create_dataset('data_%i' % call_id, data=py_obj)  # **kwargs)
    d.attrs["type"] = [b'np_scalar']

    if six.PY2:
        d.attrs["np_dtype"] = str(d.dtype)
    else:
        d.attrs["np_dtype"] = bytes(str(d.dtype), 'ascii')


def create_np_dtype(py_obj, h_group, call_id=0, **kwargs):
    """ dumps an np dtype object to h5py file

    Args:
        py_obj: python object to dump; should be a numpy scalar, e.g. np.float16(1)
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.
    """
    d = h_group.create_dataset('data_%i' % call_id, data=[str(py_obj)])
    d.attrs["type"] = [b'np_dtype']


def create_np_array_dataset(py_obj, h_group, call_id=0, **kwargs):
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
        d.attrs["type"] = [b'ndarray_masked_data']
        m.attrs["type"] = [b'ndarray_masked_mask']
    else:
        d = h_group.create_dataset('data_%i' % call_id, data=py_obj, **kwargs)
        d.attrs["type"] = [b'ndarray']




#######################
## Lookup dictionary ##
#######################

types_dict = {
    np.ndarray:  create_np_array_dataset,
    np.ma.core.MaskedArray: create_np_array_dataset,
    np.float16:    create_np_scalar_dataset,
    np.float32:    create_np_scalar_dataset,
    np.float64:    create_np_scalar_dataset,
    np.int8:       create_np_scalar_dataset,
    np.int16:      create_np_scalar_dataset,
    np.int32:      create_np_scalar_dataset,
    np.int64:      create_np_scalar_dataset,
    np.uint8:      create_np_scalar_dataset,
    np.uint16:     create_np_scalar_dataset,
    np.uint32:     create_np_scalar_dataset,
    np.uint64:     create_np_scalar_dataset,
    np.complex64:  create_np_scalar_dataset,
    np.complex128: create_np_scalar_dataset,
    np.dtype:      create_np_dtype
}

def load_np_dtype_dataset(h_node):
    py_type, data = get_type_and_data(h_node)
    data = np.dtype(data[0])
    return data

def load_np_scalar_dataset(h_node):
    py_type, data = get_type_and_data(h_node)
    subtype = h_node.attrs["np_dtype"]
    data = np.array([data], dtype=subtype)[0]
    return data

def load_ndarray_dataset(h_node):
    py_type, data = get_type_and_data(h_node)
    return np.array(data)

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
    b"ndarray_masked_mask" : load_nothing        # Loaded autormatically
}


###########
## Scipy ##
###########

if _has_scipy:

    def check_is_scipy_sparse_array(py_obj):
        """ Check if a python object is a scipy sparse array

        Args:
            py_obj: python object to check whether it is a sparse array

        Returns
            is_numpy (bool): Returns True if it is a sparse array, else False if it isn't
        """
        t_csr = type(scipy.sparse.csr_matrix([0]))
        t_csc = type(scipy.sparse.csc_matrix([0]))
        t_bsr = type(scipy.sparse.bsr_matrix([0]))
        is_sparse = type(py_obj) in (t_csr, t_csc, t_bsr)

        return is_sparse


    def create_sparse_dataset(py_obj, h_group, call_id=0, **kwargs):
        """ dumps an sparse array to h5py file

        Args:
            py_obj: python object to dump; should be a numpy array or np.ma.array (masked)
            h_group (h5.File.group): group to dump data into.
            call_id (int): index to identify object's relative location in the iterable.
        """
        h_sparsegroup = h_group.create_group('data_%i' % call_id)
        data = h_sparsegroup.create_dataset('data', data=py_obj.data, **kwargs)
        indices = h_sparsegroup.create_dataset('indices', data=py_obj.indices, **kwargs)
        indptr = h_sparsegroup.create_dataset('indptr', data=py_obj.indptr, **kwargs)
        shape = h_sparsegroup.create_dataset('shape', data=py_obj.shape, **kwargs)

        if isinstance(py_obj, type(sparse.csr_matrix([0]))):
            type_str = 'csr'
        elif isinstance(py_obj, type(sparse.csc_matrix([0]))):
            type_str = 'csc'
        elif isinstance(py_obj, type(sparse.bsr_matrix([0]))):
            type_str = 'bsr'

        if six.PY2:
            h_sparsegroup.attrs["type"] = [b'%s_matrix' % type_str]
            data.attrs["type"]          = [b"%s_matrix_data" % type_str]
            indices.attrs["type"]       = [b"%s_matrix_indices" % type_str]
            indptr.attrs["type"]        = [b"%s_matrix_indptr" % type_str]
            shape.attrs["type"]         = [b"%s_matrix_shape" % type_str]
        else:
            h_sparsegroup.attrs["type"] = [bytes(str('%s_matrix' % type_str), 'ascii')]
            data.attrs["type"]          = [bytes(str("%s_matrix_data" % type_str), 'ascii')]
            indices.attrs["type"]       = [bytes(str("%s_matrix_indices" % type_str), 'ascii')]
            indptr.attrs["type"]        = [bytes(str("%s_matrix_indptr" % type_str), 'ascii')]
            shape.attrs["type"]         = [bytes(str("%s_matrix_shape" % type_str), 'ascii')]

    def load_sparse_matrix_data(h_node):

        py_type, data = get_type_and_data(h_node)
        h_root  = h_node.parent
        indices = h_root.get('indices')[:]
        indptr  = h_root.get('indptr')[:]
        shape   = h_root.get('shape')[:]

        if py_type == b'csc_matrix_data':
            smat = sparse.csc_matrix((data, indices, indptr), dtype=data.dtype, shape=shape)
        elif py_type == b'csr_matrix_data':
            smat = sparse.csr_matrix((data, indices, indptr), dtype=data.dtype, shape=shape)
        elif py_type == b'bsr_matrix_data':
            smat = sparse.bsr_matrix((data, indices, indptr), dtype=data.dtype, shape=shape)
        return smat



    types_dict[scipy.sparse.csr_matrix] = create_sparse_dataset
    types_dict[scipy.sparse.csc_matrix] = create_sparse_dataset
    types_dict[scipy.sparse.bsr_matrix] = create_sparse_dataset


    hkl_types_dict[b"csc_matrix_data"] = load_sparse_matrix_data
    hkl_types_dict[b"csr_matrix_data"] = load_sparse_matrix_data
    hkl_types_dict[b"bsr_matrix_data"] = load_sparse_matrix_data

    # Need to ignore things like csc_matrix_indices which are loaded automatically
    for mat_type in ('csr', 'csc', 'bsr'):
        for attrib in ('indices', 'indptr', 'shape'):
            if six.PY2:
                hkl_types_dict["%s_matrix_%s" %(mat_type, attrib)] = load_nothing
            else:
                hkl_types_dict[bytes(str("%s_matrix_%s" % (mat_type, attrib)), 'ascii')] = load_nothing
