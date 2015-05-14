#! /usr/bin/env python
# encoding: utf-8
"""
test_hickle.py
===============

Unit tests for hickle module.

"""

import os
from hickle import *
import unicodedata
import hashlib
import time

NESTED_DICT = {
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

def test_unicode():
    """ Dumping and loading a unicode string """
    filename, mode = 'test.h5', 'w'
    u = unichr(233) + unichr(0x0bf2) + unichr(3972) + unichr(6000)
    dump(u, filename, mode)
    u_hkl = load(filename)

    try:
        assert type(u) == type(u_hkl) == unicode
        assert u == u_hkl
        # For those interested, uncomment below to see what those codes are:
        # for i, c in enumerate(u_hkl):
        #     print i, '%04x' % ord(c), unicodedata.category(c),
        #     print unicodedata.name(c)
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

    dd = NESTED_DICT

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
    """ Test for non-supported data types """
    filename, mode = 'nomatch.h5', 'w'

    dd = Exception('Nothing to see here')
    no_match = False
    dump(dd, filename, mode)
    
    #dd_hkl = load(filename)
    dd_hkl = load(filename, safe=False)
    
    assert type(dd_hkl) == type(dd) == Exception
    os.remove(filename)

def test_np_float():
    """ Test for singular np dtypes """
    filename, mode = 'np_float.h5', 'w'    
    
    dtype_list = (np.float16, np.float32, np.float64, 
                  np.complex64, np.complex128,
                  np.int8, np.int16, np.int32, np.int64,
                  np.uint8, np.uint16, np.uint32, np.uint64)
                  
    for dt in dtype_list:
    
        dd = dt(1)
        dump(dd, filename, mode)
        dd_hkl = load(filename)  
        assert dd == dd_hkl
        assert dd.dtype == dd_hkl.dtype
        os.remove(filename)

    dd = {}
    for dt in dtype_list:
        dd[str(dt)] = dt(1.0)
    dump(dd, filename, mode)
    dd_hkl = load(filename)

    #print dd
    for dt in dtype_list:
        assert dd[str(dt)] == dd_hkl[str(dt)]

    os.remove(filename)

def md5sum(filename, blocksize=65536):
    """ Compute MD5 sum for a given file """
    hash = hashlib.md5()
    with open(filename, "r+b") as f:
        for block in iter(lambda: f.read(blocksize), ""):
            hash.update(block)
    return hash.hexdigest()

def caching_dump(obj, filename, mode, **kwargs):
    """ Save arguments of all dump calls"""
    dump_cache.append((obj, filename, mode, kwargs))
    return hickle_dump(obj, filename, mode, **kwargs)

def test_track_times():
    """ Verify that track_times = False produces identical files"""
    hashes = []
    for obj, filename, mode, kwargs in dump_cache:
        kwargs['track_times'] = False
        hickle_dump(obj, filename, mode, **kwargs)
        hashes.append(md5sum(filename))
        os.remove(filename)

    time.sleep(1)

    for hash1, (obj, filename, mode, kwargs) in zip(hashes, dump_cache):
        hickle_dump(obj, filename, mode, **kwargs)
        hash2 = md5sum(filename)
        print hash1, hash2
        try:
            assert hash1 == hash2
            os.remove(filename)
        except AssertionError:
            os.remove(filename)
            raise


def test_comp_kwargs():
    """ Test compression with some kwargs for shuffle and chunking """

    filename, mode = 'test.h5', 'w'
    dtypes = ['int32', 'float32', 'float64', 'complex64', 'complex128']

    comps = [None, 'gzip', 'lzf']
    chunks = [(100, 100), (250, 250)]
    shuffles = [True, False]
    scaleoffsets = [0, 1, 2]

    for dt in dtypes:
        for cc in comps:
            for ch in chunks:
                for sh in shuffles:
                    for so in scaleoffsets:
                        kwargs = {
                            'compression' : cc,
                            'dtype': dt,
                            'chunks': ch,
                            'shuffle': sh,
                            'scaleoffset': so
                        }
                        #array_obj = np.random.random_integers(low=-8192, high=8192, size=(1000, 1000)).astype(dt)
                        array_obj = NESTED_DICT
                        dump(array_obj, filename, mode, compression=cc)
                        print kwargs, os.path.getsize(filename)
                        array_hkl = load(filename)
    try:
        os.remove(filename)
    except AssertionError:
        os.remove(filename)
        print array_hkl
        print array_obj
        raise

def test_list_numpy():
    """ Test converting a list of numpy arrays """

    filename, mode = 'test.h5', 'w'

    a = np.ones(1024)
    b = np.zeros(1000)
    c = [a, b]

    dump(c, filename, mode)
    dd_hkl = load(filename)

    print dd_hkl

    assert isinstance(dd_hkl, list)
    assert isinstance(dd_hkl[0], np.ndarray)


    os.remove(filename)

def test_tuple_numpy():
    """ Test converting a list of numpy arrays """

    filename, mode = 'test.h5', 'w'

    a = np.ones(1024)
    b = np.zeros(1000)
    c = (a, b, a)

    dump(c, filename, mode)
    dd_hkl = load(filename)

    print dd_hkl

    assert isinstance(dd_hkl, tuple)
    assert isinstance(dd_hkl[0], np.ndarray)


    os.remove(filename)

def test_none():
    """ Test None type hickling """
    
    filename, mode = 'test.h5', 'w'

    a = None

    dump(a, filename, mode)
    dd_hkl = load(filename)
    print a
    print dd_hkl

    assert isinstance(dd_hkl, NoneType)

    os.remove(filename)
 
def test_dict_none():
     """ Test None type hickling """
    
     filename, mode = 'test.h5', 'w'

     a = {'a': 1, 'b' : None}

     dump(a, filename, mode)
     dd_hkl = load(filename)
     print a
     print dd_hkl

     assert isinstance(a['b'], NoneType)

     os.remove(filename)   
    
dump_cache = []
hickle_dump = dump
dump = caching_dump

if __name__ == '__main__':
  """ Some tests and examples"""
  test_dict_none()
  test_none()
  test_unicode()
  test_string()
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
  test_np_float()
  test_track_times()
  time.sleep(2)
  test_comp_kwargs()
  test_list_numpy()
  test_tuple_numpy()
  
  print "ALL TESTS PASSED!"
  