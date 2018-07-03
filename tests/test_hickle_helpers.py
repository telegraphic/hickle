#! /usr/bin/env python
# encoding: utf-8
"""
# test_hickle_helpers.py

Unit tests for hickle module -- helper functions.

"""

import numpy as np
try:
    import scipy
    from scipy import sparse
    _has_scipy = True
except ImportError:
    _has_scipy = False

from hickle.helpers import check_is_hashable, check_is_iterable, check_iterable_item_type

from hickle.loaders.load_numpy import check_is_numpy_array 
if _has_scipy:
    from hickle.loaders.load_scipy import check_is_scipy_sparse_array



def test_check_is_iterable():
    assert check_is_iterable([1,2,3]) is True
    assert check_is_iterable(1) is False


def test_check_is_hashable():
    assert check_is_hashable(1) is True
    assert check_is_hashable([1,2,3]) is False


def test_check_iterable_item_type():
    assert check_iterable_item_type([1,2,3]) is int
    assert check_iterable_item_type([int(1), float(1)]) is False
    assert check_iterable_item_type([]) is False


def test_check_is_numpy_array():
    assert check_is_numpy_array(np.array([1,2,3])) is True
    assert check_is_numpy_array(np.ma.array([1,2,3])) is True
    assert check_is_numpy_array([1,2]) is False


def test_check_is_scipy_sparse_array():
    t_csr = scipy.sparse.csr_matrix([0])
    t_csc = scipy.sparse.csc_matrix([0])
    t_bsr = scipy.sparse.bsr_matrix([0])
    assert check_is_scipy_sparse_array(t_csr) is True
    assert check_is_scipy_sparse_array(t_csc) is True
    assert check_is_scipy_sparse_array(t_bsr) is True
    assert check_is_scipy_sparse_array(np.array([1])) is False

if __name__ == "__main__":
    test_check_is_hashable()
    test_check_is_iterable()
    test_check_is_numpy_array()
    test_check_iterable_item_type()
    if _has_scipy:
        test_check_is_scipy_sparse_array()