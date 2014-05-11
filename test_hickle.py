#! /usr/bin/env python
# encoding: utf-8
"""
test_hickle.py
===============

Unit tests for hickle module.

"""

import os
from hickle import *

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
        os.remove(filename)
    except AssertionError:
        os.remove(filename)
        raise

def test_set():
    """ Dumping and loading a list """
    filename, mode = 'test.h5', 'w'
    list_obj = set([1, 0, 3, 4.5, 11.2])
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
    dtypes = ['float32', 'float64', 'complex64', 'complex128']
    
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
        print array_hkl
        print array_obj
        raise

def test_dict():
    """ Test dictionary dumping and loading """
    filename, mode = 'test.h5', 'w'
    
    dd = {
        'name'   : 'Danny',
        'age'    : 28,
        'height' : 6.1,
        'dork'   : True,
        'nums'   : [1, 2, 3],
        'narr'   : np.array([1,2,3]),
        #'unic'   : u'dan[at]thetelegraphic.com'
    }
    
    
    dump(dd, filename, mode)
    dd_hkl = load(filename)
    
    for k in dd.keys():
        try:
            assert k in dd_hkl.keys()
            
            if type(dd[k]) is type(np.array([1])):
                assert np.all((dd[k], dd_hkl[k]))
            else:
                #assert dd_hkl[k] == dd[k]
                pass
            assert type(dd_hkl[k]) == type(dd[k])
        except AssertionError:
            print k
            print dd_hkl[k]
            print dd[k]
            print type(dd_hkl[k]), type(dd[k])
            os.remove(filename)
            raise
    os.remove(filename)

def test_compression():
    """ Test compression on datasets"""
    
    filename, mode = 'test.h5', 'w'
    dtypes = ['int32', 'float32', 'float64', 'complex64', 'complex128']
    
    comps = [None, 'gzip', 'lzf']
    
    for dt in dtypes:
        for cc in comps:
            array_obj = np.ones(32768, dtype=dt)
            dump(array_obj, filename, mode, compression=cc)
            print cc, os.path.getsize(filename)
            array_hkl = load(filename)
    try:
        assert array_hkl.dtype == array_obj.dtype
        assert np.all((array_hkl, array_obj))
        os.remove(filename)
    except AssertionError:
        os.remove(filename)
        print array_hkl
        print array_obj
        raise

if __name__ == '__main__':
  """ Some tests and examples"""
  test_list()
  test_set()
  test_numpy()
  test_dict()
  test_compression()
  