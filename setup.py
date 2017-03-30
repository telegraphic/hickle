# To increment version
# git tag x.y.z
# git push --tags
# python setup.py sdist upload
import sys
if sys.version_info[0] != 2:
    sys.stderr.write("This package only supports Python 2.\n")
    sys.exit(1)

from distutils.core import setup
setup(name = 'hickle',
      version = '2.0.5',
      description = 'Hickle - a HDF5 based version of pickle',
      author = 'Danny Price',
      author_email = 'dan@thetelegraphic.com',
      url = 'http://github.com/telegraphic/hickle',
      download_url='https://github.com/telegraphic/hickle/archive/2.0.5.tar.gz',
      platforms = 'Cross platform (Linux, Mac OSX, Windows)',
      keywords = ['pickle', 'hdf5', 'data storage', 'data export'],
      py_modules = ['hickle', 'hickle_legacy'],
      install_requires=['numpy', 'h5py', 'scipy']
      )
