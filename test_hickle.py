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


def test_masked():
    """ Test masked numpy array """
    filename, mode = 'test.h5', 'w'    
    a = np.ma.array([1,2,3,4], dtype='float32', mask=[0,1,0,0])
    
    dump(a, filename, mode)
    a_hkl = load(filename)
    
    try:
        assert a_hkl.dtype == a.dtype
        assert np.all((a_hkl, a))
        os.remove(filename)
    except AssertionError:
        os.remove(filename)
        print a_hkl
        print a
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

def test_dict_int_key():
    """ Test for dictionaries with integer keys """
    filename, mode = 'test.h5', 'w'

    dd = {
        0: "test",
        1: "test2"
    }

    dump(dd, filename, mode)
    dd_hkl = load(filename)

    
    os.remove(filename)

def test_dict_nested():
    """ Test for dictionaries with integer keys """
    filename, mode = 'test.h5', 'w'

    dd = {
        "level1_1" : {
            "level2_1" : [1, 2, 3],
            "level2_2" : [4, 5, 6]
        },
        "level1_2" : {
            "level2_1" : [1, 2, 3],
            "level2_2" : [4, 5, 6]            
        },
        "level1_3" : {
            "level2_1" : {
                "level3_1" : [1, 2, 3],
                "level3_2" : [4, 5, 6]                     
            },
            "level2_2" : [4, 5, 6]            
        }
    }

    dump(dd, filename, mode)
    dd_hkl = load(filename)
    
    ll_hkl = dd_hkl["level1_3"]["level2_1"]["level3_1"]
    ll     = dd["level1_3"]["level2_1"]["level3_1"]
    assert ll == ll_hkl
    os.remove(filename)

def test_masked_dict():
    """ Test dictionaries with masked arrays """

    filename, mode = 'test.h5', 'w'

    dd = {
        "data"  : np.ma.array([1,2,3], mask=[True, False, False]),
        "data2" : np.array([1,2,3,4,5])
    }

    dump(dd, filename, mode)
    dd_hkl = load(filename)
    
    for k in dd.keys():
        try:
            assert k in dd_hkl.keys()
            if type(dd[k]) is type(np.array([1])):
                assert np.all((dd[k], dd_hkl[k]))
            elif type(dd[k]) is type(np.ma.array([1])):
                print dd[k].data
                print dd_hkl[k].data
                assert np.allclose(dd[k].data, dd_hkl[k].data)
                assert np.allclose(dd[k].mask, dd_hkl[k].mask)
                
            assert type(dd_hkl[k]) == type(dd[k])

        except AssertionError:
            print k
            print dd_hkl[k]
            print dd[k]
            print type(dd_hkl[k]), type(dd[k])
            os.remove(filename)
            raise
    os.remove(filename)

def test_nomatch():
    """ Test for dictionaries with integer keys """
    filename, mode = 'donotmakethisfile.h5', 'w'

    dd = Exception('Nothing to see here')
    no_match = False
    
    try:
        dump(dd, filename, mode)
    except NoMatchError:
        no_match = True
        print "PASS: No match exception raised!"
    assert no_match is True
    assert not os.path.isfile(filename)

if __name__ == '__main__':
  """ Some tests and examples"""
  test_masked_dict()
  test_list()
  test_set()
  test_numpy()
  test_dict()
  test_compression()
  test_masked()
  test_dict_int_key()
  test_dict_nested()
  test_nomatch()
  