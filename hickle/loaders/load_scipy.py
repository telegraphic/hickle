import six
import scipy
from scipy import sparse

from ..helpers import get_type_and_data

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
