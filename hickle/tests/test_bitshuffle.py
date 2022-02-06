import os
import sys
import numpy as np
import pytest
import hickle as hkl
from hickle.hickle_bshuf import BitShuffleLz4


H5_FILE = "./test.h5"
array1 = np.random.rand(16, 1, 1048576)


def install_hdf5plugin():
    cmd = "python3 -m pip install -U hdf5plugin"
    exit_code = os.system(cmd)
    assert exit_code == 0


def uninstall_hdf5plugin():
    cmd = "python3 -m pip uninstall -y hdf5plugin"
    exit_code = os.system(cmd)
    assert exit_code == 0


def validator(label, array_A, array_B):
    print("validator: {} .....".format(label))
    
    # Check the two are the same file
    print(array_A.shape, array_B.shape)
    assert array_A.shape == array_B.shape
    print(array_A.dtype, array_B.dtype)
    assert array_A.dtype == array_B.dtype
    assert np.all((array_A, array_B))


@pytest.mark.no_compression
@pytest.mark.skipif(sys.platform == "win32" and sys.maxsize < 2**32, reason="no wheel for hdf5plugin available on windows 32 bit")
@pytest.mark.order(1)
def test_using_bitshuffle():
    # Uncompressed.
    hkl.dump(array1, H5_FILE, mode="w")
    array2 = hkl.load(H5_FILE)
    validator("No compression", array1, array2)
    
    # Compress with gzip.
    hkl.dump(array1, "essai_gzipped.hkl", mode="w", compression="gzip")
    array2 = hkl.load(H5_FILE)
    validator("Gzip compression", array1, array2)
    
    # Compress with Bitsshuffle + LZ4.
    install_hdf5plugin()
    bsh_hkl = BitShuffleLz4()
    bsh_hkl.dump(array1, H5_FILE, mode="w")
    array2 = bsh_hkl.load(H5_FILE)
    validator("Bitshuffle + LZ4", array1, array2)

    # Cleanup.
    os.remove(H5_FILE)
    
    # Remove bitshuffle for the remaining tests.
    uninstall_hdf5plugin()


if __name__ == "__main__":
    test_using_bitshuffle()
