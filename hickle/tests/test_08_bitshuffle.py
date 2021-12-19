import os
import numpy as np
import hickle as hkl
from hickle.hickle_bshuf import BitShuffleLz4


H5_FILE = "./test.h5"
array1 = np.random.rand(16, 1, 1048576)


def validator(label, array_A, array_B):
    print("validator: {} .....".format(label))
    
    # Check the two are the same file
    print(array_A.shape, array_B.shape)
    assert array_A.shape == array_B.shape
    print(array_A.dtype, array_B.dtype)
    assert array_A.dtype == array_B.dtype
    assert np.all((array_A, array_B))


def test_08_bitshuffle():    
    # Uncompressed.
    hkl.dump(array1, H5_FILE, mode="w")
    array2 = hkl.load(H5_FILE)
    validator("No compression", array1, array2)
    
    # Compress with gzip.
    hkl.dump(array1, "essai_gzipped.hkl", mode="w", compression="gzip")
    array2 = hkl.load(H5_FILE)
    validator("Gzip compression", array1, array2)
    
    # Compress with Bitsshuffle + LZ4.
    bsh_hkl = BitShuffleLz4()
    bsh_hkl.dump(array1, H5_FILE, mode="w")
    array2 = bsh_hkl.load(H5_FILE)
    validator("Bitshuffle + LZ4", array1, array2)

    # Cleanup.
    os.remove(H5_FILE)

test_08_bitshuffle()
