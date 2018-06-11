import glob
import warnings
import hickle as hkl
import h5py
import six

def test_legacy_load():
    if six.PY2:
        filelist = sorted(glob.glob('legacy_hkls/*.hkl'))

        # Make all warnings show
        warnings.simplefilter("always")

        for filename in filelist:
            try:
                print(filename)
                a = hkl.load(filename)
            except:
                with h5py.File(filename) as a:
                    print(a.attrs.items())
                    print(a.items())
                    for key, item in a.items():
                        print(item.attrs.items())
                    raise
    else:
        print("Legacy loading only works in Py2. Sorry.")
        pass

if __name__ == "__main__":
    test_legacy_load()