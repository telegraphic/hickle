# To increment version
# Check you have ~/.pypirc filled in
# git tag x.y.z
# git push --tags
# python setup.py sdist upload
from setuptools import setup, find_packages

version = '3.3.0'
author  = 'Danny Price'

setup(name='hickle',
      version=version,
      description='Hickle - a HDF5 based version of pickle',
      author=author,
      author_email='dan@thetelegraphic.com',
      url='http://github.com/telegraphic/hickle',
      download_url='https://github.com/telegraphic/hickle/archive/%s.tar.gz' % version,
      platforms='Cross platform (Linux, Mac OSX, Windows)',
      keywords=['pickle', 'hdf5', 'data storage', 'data export'],
      #py_modules = ['hickle', 'hickle_legacy'],
      install_requires=['numpy', 'h5py'],
      extras_require={
            'astropy': ['astropy'],
            'scipy': ['scipy'],
            'pandas': ['pandas'],
      },
      python_requires='>=2.7',
      packages=find_packages(),
      zip_safe=False,
)
