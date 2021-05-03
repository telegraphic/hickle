#! /usr/bin/env python
# encoding: utf-8
"""
# generate_legacy_4_0_0.py

Creates datastructure to be dumped to the hickle_4_0_0.hkl file.

When run as script under hickle 4.0.0 or hickle 4.0.1 it will
result in a valid legacy 4.0.0 file which can be used to tests
that later version are still capable loading hickle 4.0.0 format
files.

When imported by any of the tests the method generate_py_object
returns the same datastructure stored to the prior generated file.

"""
import sys
sys.path.insert(0,"../..")
import hickle
import numpy as np
import scipy
import scipy.sparse
import astropy
import collections
import dill as pickle
import os.path

def generate_py_object():
    """
    create a data structure covering all or at least the most obvious,
    prominent and most likely breaking differences between hickle 
    4.0.0/4.0.1 version and Versions > 4.1.0

    Returns:
        list object containing all the relevant data objects and the 
        filename of the file the data has been stored to or shall be
        stored to. 
    """
    scriptdir = os.path.split(__file__)[0]
    some_string = "this is some string to be dumped by hickle 4.0.0"
    some_bytes = b"this is the same in bytes instead of utf8"
    some_char_list = list(some_string)
    some_bytes_list = list(some_bytes)
    some_numbers = tuple(range(50))
    some_floats = tuple( float(f) for f in range(50))
    mixed = list( f for f in ( some_numbers[i//2] if i & 1 else some_floats[i//2] for i in range(100) ) )
    wordlist = ["hello","world","i","like","you"]
    byteslist = [ s.encode("ascii") for s in wordlist]
    mixus = [some_string,some_numbers,12,11]
    numpy_array = np.array([
        [
            0.8918443906408066, 0.5408942506873636, 0.43463333793335346, 0.21382281373491407,
            0.14580527098359963, 0.6869306139451369, 0.22954988509310692, 0.2833880251470392,
            0.8811201329390297, 0.4144190218983931, 0.06595369247674943
        ], [
            0.8724300029833221, 0.7173303189807705, 0.5721666862018427, 0.8535567654595188,
            0.5806566016388102, 0.9921250367638187, 0.07104048226766191, 0.47131100732975095,
            0.8006065068241431, 0.2804909335297441, 0.1968823602346148
        ], [
            0.0515177648326276, 0.1852582437284651, 0.22016412062225577, 0.6393104121476216,
            0.7751103631149562, 0.12810902186723572, 0.09634877693000932, 0.2388423061420949,
            0.5730001119950099, 0.1197268172277629, 0.11539619086292308
        ], [
            0.031751102230864414, 0.21672180477587166, 0.4366501648161476, 0.9549518596659471,
            0.42398684476912474, 0.04490851499559967, 0.7394234049135264, 0.7378312792413693,
            0.9808812550712923, 0.2488404519024885, 0.5158454824458993
        ], [
            0.07550969197984403, 0.08485317435746553, 0.15760274251917195, 0.18029979414515496,
            0.9501707036126847, 0.1723868250469468, 0.7951538687631865, 0.2546219217084682,
            0.9116518509985955, 0.6930255788272572, 0.9082828280630456
        ], [
            0.6712307672376565, 0.367223385378443, 0.9522931417348294, 0.714592360187415,
            0.18334824241062575, 0.9322238504996762, 0.3594776411821822, 0.6302097368268973,
            0.6281766915388312, 0.7114942437206809, 0.6977764481953693
        ], [
            0.9541502922560433, 0.47788295940203784, 0.6511716236981558, 0.4079446664375711,
            0.2747969334307605, 0.3571662787734283, 0.10235638316970186, 0.8567343897483571,
            0.6623468654315807, 0.21377047332104315, 0.860146852430476
        ]
    ])
    mask = np.array([
        [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],
        [0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0],
        [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1],
        [0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1]
    ])

    numpy_array_masked = np.ma.array(numpy_array, dtype='float32', mask=mask)
    plenty_dict = {
        "string":1,
        b'bytes':2,
        12:3,
        0.55:4,
        complex(1,4):5,
        (1,):6,
        tuple(mixus):7,
        ():8,
        '9':9,
        None:10,
        'a/b':11
    }
    odrdered_dict = collections.OrderedDict(((3, [3, 0.1]), (7, [5, 0.1]), (5, [3, 0.1])))
    
    row = np.array([0, 0, 1, 2, 2, 2])
    col = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6])
    csr_matrix = scipy.sparse.csr_matrix((data, (row, col)), shape=(3, 3))
    csc_matrix = scipy.sparse.csc_matrix((data, (row, col)), shape=(3, 3))
    
    indptr = np.array([0, 2, 3, 6])
    indices = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6]).repeat(4).reshape(6, 2, 2)
    bsr_matrix = scipy.sparse.bsr_matrix((data, indices, indptr), shape=(6, 6))
    numpy_string = np.array(some_string)
    numpy_bytes = np.array(some_bytes)
    numpy_wordlist = np.array(wordlist)
    numpy_dict = np.array({})
    
    return [
        some_string ,
        some_bytes ,
        some_char_list ,
        some_bytes_list ,
        some_numbers ,
        some_floats ,
        mixed ,
        wordlist ,
        byteslist ,
        mixus ,
        numpy_array ,
        mask ,
        numpy_array_masked ,
        plenty_dict ,
        odrdered_dict ,
        csr_matrix ,
        csc_matrix ,
        bsr_matrix ,
        numpy_string ,
        numpy_bytes ,
        numpy_wordlist ,
        numpy_dict
    ],os.path.join(scriptdir,"legacy_hkls","hickle_4.0.0.hkl") 

if __name__ == '__main__':
    # create the file by dumping using hickle but only if
    # the available hickle version is >= 4.0.0 and < 4.1.0
    hickle_version = hickle.__version__.split('.')
    if hickle_version[0] != 4 or hickle_version[1] > 0:
        raise RuntimeError("Shall be run using < 4.1 only")
    scriptdir = os.path.split(__file__)[0]
    now_dumping,testfile = generate_py_object()
    hickle.dump(now_dumping,testfile)
