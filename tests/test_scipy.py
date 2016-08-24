import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
import os

from hickle import check_is_scipy_sparse_array
import hickle

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

    try:
        hickle.dump(sm1, 'test.h5')
        sm1_h = hickle.load('test.h5')
        hickle.dump(sm2, 'test2.h5')
        sm2_h = hickle.load('test2.h5')

        assert isinstance(sm1_h, csr_matrix)
        assert isinstance(sm2_h, csc_matrix)

        assert np.allclose(sm1_h.data, sm1.data)
        assert np.allclose(sm2_h.data, sm2.data)

    finally:
        #os.remove('test.h5')
        pass



if __name__ == "__main__":
    test_sparse_matrix()
    test_is_sparse()