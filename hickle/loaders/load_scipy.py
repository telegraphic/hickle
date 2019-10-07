import six
import scipy
import numpy as np
from scipy import sparse

try:
    import dill as pickle
except ImportError:
    try:
        import cPickle as pickle
    except ImportError:
        import pickle

import sys
if sys.version_info.major == 3:
    NoneType = type(None)
else:
    from types import NoneType

from hickle.helpers import get_type_and_data


def return_first(x):
    """ Return first element of a list """
    return x[0]


def check_is_scipy_sparse_array(py_obj):
    """ Check if a python object is a scipy sparse array

    Args:
        py_obj: python object to check whether it is a sparse array

    Returns
        is_numpy (bool): Returns True if it is a sparse array, else False if it isn't
    """
    t_csr = sparse.csr_matrix
    t_csc = sparse.csc_matrix
    t_bsr = sparse.bsr_matrix
    is_sparse = type(py_obj) in (t_csr, t_csc, t_bsr)

    return is_sparse


def create_sparse_dataset(py_obj, h_group, name, **kwargs):
    """ dumps an sparse array to h5py file

    Args:
        py_obj: python object to dump; should be a numpy array or np.ma.array (masked)
        h_group (h5.File.group): group to dump data into.
        call_id (int): index to identify object's relative location in the iterable.
    """
    h_sparsegroup = h_group.create_group(name)
    data = h_sparsegroup.create_dataset('data', data=py_obj.data, **kwargs)
    indices = h_sparsegroup.create_dataset('indices', data=py_obj.indices, **kwargs)
    indptr = h_sparsegroup.create_dataset('indptr', data=py_obj.indptr, **kwargs)
    shape = h_sparsegroup.create_dataset('shape', data=py_obj.shape, **kwargs)

    if isinstance(py_obj, sparse.csr_matrix):
        type_str = 'csr'
    elif isinstance(py_obj, sparse.csc_matrix):
        type_str = 'csc'
    elif isinstance(py_obj, sparse.bsr_matrix):
        type_str = 'bsr'

    h_sparsegroup.attrs['type'] = np.array(pickle.dumps(return_first))
    h_sparsegroup.attrs['base_type'] = ('%s_matrix' % type_str).encode('ascii')
    indices.attrs['type']               = np.array(pickle.dumps(NoneType))
    indices.attrs['base_type']       = ("%s_matrix_indices" % type_str).encode('ascii')
    indptr.attrs['type']               = np.array(pickle.dumps(NoneType))
    indptr.attrs['base_type']        = ("%s_matrix_indptr" % type_str).encode('ascii')
    shape.attrs['type']               = np.array(pickle.dumps(NoneType))
    shape.attrs['base_type']         = ("%s_matrix_shape" % type_str).encode('ascii')

    return(data)

def load_sparse_matrix_data(h_node):

    _, base_type, data = get_type_and_data(h_node)
    h_root  = h_node.parent
    indices = h_root.get('indices')[:]
    indptr  = h_root.get('indptr')[:]
    shape   = h_root.get('shape')[:]

    if base_type == b'csc_matrix_data':
        smat = sparse.csc_matrix((data, indices, indptr), dtype=data.dtype, shape=shape)
    elif base_type == b'csr_matrix_data':
        smat = sparse.csr_matrix((data, indices, indptr), dtype=data.dtype, shape=shape)
    elif base_type == b'bsr_matrix_data':
        smat = sparse.bsr_matrix((data, indices, indptr), dtype=data.dtype, shape=shape)
    return smat





class_register = [
    [scipy.sparse.csr_matrix, b'csr_matrix_data', create_sparse_dataset, load_sparse_matrix_data, False, check_is_scipy_sparse_array],
    [scipy.sparse.csc_matrix, b'csc_matrix_data', create_sparse_dataset, load_sparse_matrix_data, False, check_is_scipy_sparse_array],
    [scipy.sparse.bsr_matrix, b'bsr_matrix_data', create_sparse_dataset, load_sparse_matrix_data, False, check_is_scipy_sparse_array],
]

exclude_register = []

# Need to ignore things like csc_matrix_indices which are loaded automatically
for mat_type in ('csr', 'csc', 'bsr'):
    for attrib in ('indices', 'indptr', 'shape'):
        hkl_key = "%s_matrix_%s" % (mat_type, attrib)
        if not six.PY2:
            hkl_key = hkl_key.encode('ascii')
        exclude_register.append(hkl_key)
