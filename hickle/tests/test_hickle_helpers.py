#! /usr/bin/env python
# encoding: utf-8
"""
# test_hickle_helpers.py

Unit tests for hickle module -- helper functions.

"""


# %% IMPORTS
# Package imports
import numpy as np

# hickle imports
from hickle.helpers import (
    check_is_hashable, check_is_iterable, check_iterable_item_type)
from hickle.loaders.load_numpy import check_is_numpy_array


# %% FUNCTION DEFINITIONS
def test_check_is_iterable():
    assert check_is_iterable([1, 2, 3])
    assert not check_is_iterable(1)


def test_check_is_hashable():
    assert check_is_hashable(1)
    assert not check_is_hashable([1, 2, 3])


def test_check_iterable_item_type():
    assert check_iterable_item_type([1, 2, 3]) is int
    assert not check_iterable_item_type([int(1), float(1)])
    assert not check_iterable_item_type([])


def test_check_is_numpy_array():
    assert check_is_numpy_array(np.array([1, 2, 3]))
    assert check_is_numpy_array(np.ma.array([1, 2, 3]))
    assert not check_is_numpy_array([1, 2])


# %% MAIN SCRIPT
if __name__ == "__main__":
    test_check_is_hashable()
    test_check_is_iterable()
    test_check_is_numpy_array()
    test_check_iterable_item_type()
