import glob
import warnings
import hickle as hkl
import h5py


def test_legacy_load():
    filelist = sorted(glob.glob('legacy_hkls/*.hkl'))

    # Make all warnings show
    warnings.simplefilter("always")

    for filename in filelist:
        try:
            print(filename)
            a = hkl.load(filename)
        except Exception:
            with h5py.File(filename) as a:
                print(a.attrs.items())
                print(a.items())
                for key, item in a.items():
                    print(item.attrs.items())
                raise

if __name__ == "__main__":
    test_legacy_load()
