# To increment version
# git tag x.y.z
# git push --tags
# python setup.py sdist upload
from distutils.core import setup
setup(name = 'hickle',
      version = '1.3.2',
      description = 'Hickle - a HDF5 based version of pickle',
      author = 'Danny Price',
      author_email = 'dan@thetelegraphic.com',
      url = 'http://github.com/telegraphic/hickle',
      download_url='https://github.com/telegraphic/hickle/archive/1.3.2.tar.gz',
      platforms = 'Cross platform (Linux, Mac OSX, Windows)',
      keywords = ['pickle', 'hdf5', 'data storage', 'data export'],
      py_modules = ['hickle'],
      install_requires=['numpy', 'h5py']
      )