#! /usr/bin/env python
# encoding: utf-8
"""
# test_hickle.py

Unit tests for hickle module.

"""

import os
from hickle import *
import hickle
import hashlib
import time

import h5py
import numpy as np
from pprint import pprint

NESTED_DICT = {
    "level1_1": {
        "level2_1": [1, 2, 3],
        "level2_2": [4, 5, 6]
    },
    "level1_2": {
        "level2_1": [1, 2, 3],
        "level2_2": [4, 5, 6]
    },
    "level1_3": {
        "level2_1": {
            "level3_1": [1, 2, 3],
            "level3_2": [4, 5, 6]
        },
        "level2_2": [4, 5, 6]
    }
}

DUMP_CACHE = []             # Used in test_track_times()


def test_string():
    """ Dumping and loading a string """
    filename, mode = 'test.h5', 'w'
    string_obj = "The quick brown fox jumps over the lazy dog"
    dump(string_obj, filename, mode)
    string_hkl = load(filename)
    #print "Initial list:   %s"%list_obj
    #print "Unhickled data: %s"%list_hkl
    try:
        assert type(string_obj) == type(string_hkl) == str
        assert string_obj == string_hkl
        os.remove(filename)
    except AssertionError:
        os.remove(filename)
        raise


def test_list():
    """ Dumping and loading a list """
    filename, mode = 'test.h5', 'w'
    list_obj = [1, 2, 3, 4, 5]
    dump(list_obj, filename, mode)
    list_hkl = load(filename)
    #print "Initial list:   %s"%list_obj
    #print "Unhickled data: %s"%list_hkl
    try:
        assert type(list_obj) == type(list_hkl) == list
        assert list_obj == list_hkl
        import h5py
        a = h5py.File(filename)

        os.remove(filename)
    except AssertionError:
        print("ERR:", list_obj, list_hkl)
        import h5py
        os.remove(filename)
        raise


def test_set():
    """ Dumping and loading a list """
    filename, mode = 'test.h5', 'w'
    list_obj ={1.0, 0.0, 3.0, 4.5, 11.2}
    dump(list_obj, filename, mode)
    list_hkl = load(filename)
    #print "Initial list:   %s"%list_obj
    #print "Unhickled data: %s"%list_hkl
    try:
        assert type(list_obj) == type(list_hkl) == set
        assert list_obj == list_hkl
        os.remove(filename)
    except AssertionError:
        os.remove(filename)
        raise


def test_numpy():
    """ Dumping and loading numpy array """
    filename, mode = 'test.h5', 'w'
    dtypes = [b'float32', b'float64', b'complex64', b'complex128']
    
    for dt in dtypes:
        array_obj = np.ones(8, dtype=dt)
        dump(array_obj, filename, mode)
        array_hkl = load(filename)
    try:
        assert array_hkl.dtype == array_obj.dtype
        assert np.all((array_hkl, array_obj))
        os.remove(filename)
    except AssertionError:
        os.remove(filename)
        print(array_hkl)
        print(array_obj)
        raise



    
if __name__ == '__main__':
    """ Some tests and examples """
    #test_string()
    test_list()
    test_set()
    test_numpy()

    print("ALL TESTS PASSED!")