from distutils.core import setup
setup(name = 'hickle',
      version = '1.0.2',
      description = 'Hickle - a HDF5 based version of pickle',
      author = 'Danny Price',
      author_email = 'danny.price@astro.ox.ac.uk',
      url = 'http://github.com/telegraphic/hickle',
      download_url='https://github.com/telegraphic/hickle/archive/1.0.2.tar.gz',
      platforms = 'Cross platform (Linux, Mac OSX, Windows)',
      keywords = ['pickle', 'hdf5', 'data storage', 'data export'],
      py_modules = ['hickle']
      )