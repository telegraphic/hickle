#! /usr/bin/env python
# encoding: utf-8
"""
# test_hickle_helpers.py

Unit tests for hickle module -- helper functions.

"""

import numpy as np

from hickle.helpers import check_is_hashable, check_is_iterable, check_iterable_item_type

from hickle.loaders.load_numpy import check_is_numpy_array


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

if __name__ == "__main__":
    test_check_is_hashable()
    test_check_is_iterable()
    test_check_is_numpy_array()
    test_check_iterable_item_type()
