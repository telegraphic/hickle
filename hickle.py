# encoding: utf-8
"""
hickle.py
=============

Created by Danny Price and Jack Hickish on 2012-05-28.
Copyright (c) 2012 The University of Oxford. All rights reserved.

Hickle is a HDF5 based clone of Pickle. Instead of serializing to a pickle file,
Hickle dumps to a HDF5 file. It is designed to be as similar to pickle in usage as possible.

Notes
-----

Hickle has two main advantages over Pickle:
1) LARGE PICKLE HANDLING. Unpickling a large pickle is slow, as the Unpickler reads the entire pickle 
thing and loads it into memory. In comparison, HDF5 files are designed for large datasets. Things are 
only loaded when accessed. 

2) CROSS PLATFORM SUPPORT. Attempting to unpickle a pickle pickled on Windows on Linux and vice versa
is likely to fail with errors like "Insecure string pickle". HDF5 files will load fine, as long as
both machines have h5py installed.

TODO: Add support for chunking and compression

"""

import exceptions
import numpy as np
import h5py as h5

__version__ = "0.1"
__author__  = "Danny Price and Jack Hickish"

####################
## Error handling ##
####################

class FileError(exceptions.Exception):
  """ An exception raised if the file is fishy"""
  def __init__(self):
    return
  def __str__(self):
    print "Error: cannot open file. Please pass either a filename string, a file object, or a h5py.File"

class NoMatchError(exceptions.Exception):
  """ An exception raised if the object type is not understood (or supported)"""
  def __init__(self):
    return
  def __str__(self):
    print "Error: this type of python object cannot be converted into a hickle."

class ToDoError(exceptions.Exception):
  """ An exception raised for non-implemented functionality"""
  def __init__(self):
    return
  def __str__(self):
    print "Error: this functionality hasn't been implemented yet."

def fileOpener(file, mode='r'):
  """ A file opener helper function with some error handling. 
  
  This can open files through a file object, a h5py file, or just the filename.
  """
  # Were we handed a file object or just a file name string?
  if type(file) is file:
    filename, mode = file.name(), file.mode()
    file.close()
    h5f = h5.file(filename, mode)
  elif type(file) is h5._hl.files.File:
    h5f = file
  elif type(file) is str:
    filename = file
    h5f = h5.File(filename, mode)
  else:
    raise FileError
  
  return h5f


#############
## dumpers ##
#############

def dumpNdarray(obj, h5f):
  """ dumps an ndarray object to h5py file"""
  h5f.create_dataset('data', data=obj)

def dumpList(obj, h5f):
  """ dumps a list object to h5py file"""
  h5f.create_dataset('data', data=obj)

def dumpDict(obj, h5f=''):
  """ dumps a dictionary to h5py file """
  raise ToDoError

def noMatch(obj, h5f=''):
  """ If no match is made, raise an exception """
  raise NoMatchError

def dumperLookup(obj):
  """ What type of object are we trying to pickle?
   
  This is a python dictionary based equivalent of a case statement.
  It returns the correct helper function for a given data type.
  """
  t = type(obj)
  
  types = {
     list       : dumpList,
     np.ndarray : dumpNdarray,
     dict       : dumpDict
  }
  
  match = types.get(t, noMatch)
  return match

def dump(obj, file, mode='r'):
  """ Write a pickled representation of obj to the open file object file. 
  
  Parameters
  ----------
  obj: object
    python object o store in a Hickle
  file: file object, filename string, or h5py.File object
    file in which to store the object. A h5py.File or a filename is also acceptable.
  mode: string
    optional argument, 'r' (read only), 'w' (write) or 'a' (append). Ignored if file is a file object.
  """
  
  # Open the file
  h5f = fileOpener(file, mode)
  
  # Now dump to file
  dumper = dumperLookup(obj)
  print "dumping %s to file %s"%(type(obj), repr(h5f))
  dumper(obj, h5f)
  h5f.close()

  
def dumps(obj, file, mode='r'):
  """ Not sure how and whether to support this or not. """
  raise ToDoError


#############
## loaders ##
#############

def load(file):
  """ Load a hickle file and reconstruct a python object
  
  Parameters
  ----------
  file: file object, h5py.File, or filename string
  """
  
  h5f = fileOpener(file)
  data = h5f["data"][:]
  h5f.close()

  return data

def loadLarge(file):
  """ Load a large hickle file (returns the h5py object not the data) 

  Parameters
  ----------
  file: file object, h5py.File, or filename string  
  """

  h5f = fileOpener(file)
  return h5f

def loads(file):
  """ Not sure whether to support this or not. """
  raise ToDoError


##########
## Main ##
##########
  
if __name__ == '__main__':
  """ Some tests and examples"""
  
  # Dumping and loading a list
  filename, mode = 'test.h5', 'w'
  list_obj = [1, 2, 3, 4, 5]
  dump(list_obj, filename, mode)
  list_hkl = load(filename)
  print "Initial list: %s"%list_obj
  print "Unhickled data: %s"%list_hkl

  
  # Dumping and loading a numpy array
  filename, mode = 'test.h5', 'w'
  array_obj = np.array([1, 2, 3, 4, 5])
  dump(array_obj, filename, mode)
  array_hkl = load(filename)
  print "Initial array: %s"%array_obj
  print "Unhickled data: %s"%array_hkl
  
  