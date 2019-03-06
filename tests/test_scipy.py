import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, bsr_matrix

import hickle
from hickle.loaders.load_scipy import check_is_scipy_sparse_array

from py.path import local

# Set the current working directory to the temporary directory
local.get_temproot().chdir()


def test_is_sparse():
    sm0 = csr_matrix((3, 4), dtype=np.int8)
    sm1 = csc_matrix((1, 2))

    assert check_is_scipy_sparse_array(sm0)
    assert check_is_scipy_sparse_array(sm1)


def test_sparse_matrix():
    sm0 = csr_matrix((3, 4), dtype=np.int8).toarray()

    row = np.array([0, 0, 1, 2, 2, 2])
    col = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6])
    sm1 = csr_matrix((data, (row, col)), shape=(3, 3))
    sm2 = csc_matrix((data, (row, col)), shape=(3, 3))

    indptr = np.array([0, 2, 3, 6])
    indices = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6]).repeat(4).reshape(6, 2, 2)
    sm3 = bsr_matrix((data,indices, indptr), shape=(6, 6))

    hickle.dump(sm1, 'test_sp.h5')
    sm1_h = hickle.load('test_sp.h5')
    hickle.dump(sm2, 'test_sp2.h5')
    sm2_h = hickle.load('test_sp2.h5')
    hickle.dump(sm3, 'test_sp3.h5')
    sm3_h = hickle.load('test_sp3.h5')

    assert isinstance(sm1_h, csr_matrix)
    assert isinstance(sm2_h, csc_matrix)
    assert isinstance(sm3_h, bsr_matrix)

    assert np.allclose(sm1_h.data, sm1.data)
    assert np.allclose(sm2_h.data, sm2.data)
    assert np.allclose(sm3_h.data, sm3.data)

    assert sm1_h. shape == sm1.shape
    assert sm2_h. shape == sm2.shape
    assert sm3_h. shape == sm3.shape


if __name__ == "__main__":
    test_sparse_matrix()
    test_is_sparse()