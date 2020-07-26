# To increment version
# Check you have ~/.pypirc filled in
# git tag x.y.z
# git push && git push --tags
# rm -rf dist; python setup.py sdist bdist_wheel
# TEST: twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# twine upload dist/*

from codecs import open
import re

from setuptools import setup, find_packages
import sys

author = "Danny Price, Ellert van der Velden and contributors"

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", 'r') as fh:
    requirements = fh.read().splitlines()

with open("requirements_test.txt", 'r') as fh:
    test_requirements = fh.read().splitlines()

# Read the __version__.py file
with open('hickle/__version__.py', 'r') as f:
    vf = f.read()

# Obtain version from read-in __version__.py file
version = re.search(r"^_*version_* = ['\"]([^'\"]*)['\"]", vf, re.M).group(1)

setup(name='hickle',
      version=version,
      description='Hickle - an HDF5 based version of pickle',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author=author,
      author_email='dan@thetelegraphic.com',
      url='http://github.com/telegraphic/hickle',
      download_url=('https://github.com/telegraphic/hickle/archive/v%s.zip'
                    % (version)),
      platforms='Cross platform (Linux, Mac OSX, Windows)',
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved',
          'Natural Language :: English',
          'Operating System :: MacOS',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: Unix',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Topic :: Software Development :: Libraries :: Python Modules',
          'Topic :: Utilities',
          ],
      keywords=['pickle', 'hdf5', 'data storage', 'data export'],
      install_requires=requirements,
      tests_require=test_requirements,
      python_requires='>=3.5',
      packages=find_packages(),
      zip_safe=False,
)
