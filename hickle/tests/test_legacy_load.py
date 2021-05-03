# %% IMPORTS
# Built-in imports
import glob
from os import path
import warnings
import pytest
import scipy.sparse
import numpy as np

# Package imports
import h5py

# hickle imports
import hickle as hkl
import dill as pickle


# %% FUNCTION DEFINITIONS
def test_legacy_load():
    dirpath = path.dirname(__file__)
    filelist = sorted(glob.glob(path.join(dirpath, 'legacy_hkls/*3_[0-9]_[0-9].hkl')))

    # Make all warnings show
    warnings.simplefilter("always")

    for filename in filelist:
        with pytest.warns(
            UserWarning,
            match = r"Input\s+argument\s+'file_obj'\s+appears\s+to\s+be\s+a\s+file\s+made"
                    r"\s+with\s+hickle\s+v3.\s+Using\s+legacy\s+load..."
        ):
            try:
                print(filename)
                a = hkl.load(filename,path='test')
            except Exception:
                with h5py.File(filename) as a:
                    print(a.attrs.items())
                    print(a.items())
                    for key, item in a.items():
                        print(item.attrs.items())
                    raise

@pytest.mark.no_compression
def test_4_0_0_load():
    """
    test that files created by hickle 4.0.x can be loaded by 
    hickle 4.1.x properly
    """
    dirpath = path.dirname(__file__)
    filelist = sorted(glob.glob(path.join(dirpath, 'legacy_hkls/*4.[0-9].[0-9].hkl')))
    from hickle.tests.generate_legacy_4_0_0 import generate_py_object
    compare_with,needs_compare = generate_py_object()
    # strange but without forcing garbage collection here h5py might produce 
    # strange assuming a race related RuntimeError when h5py file is closed by
    # hickle.load(). Unless observed in wildlife this is only triggered by fast successive
    # calls of h5py methods.
    import gc
    gc.collect()
    for filename in filelist:
        content = hkl.load(filename)
        if filename != needs_compare:
            continue
        for item_id,content_item,compare_item in ( (i,content[i],compare_with[i]) for i in range(len(compare_with)) ):
            if scipy.sparse.issparse(content_item):
                assert np.allclose(content_item.toarray(),compare_item.toarray())
                continue
            try:
                assert content_item == compare_item
            except ValueError:
                assert np.all(content_item == compare_item)

# %% MAIN SCRIPT
if __name__ == "__main__":
    test_legacy_load()
    test_4_0_0_load()
