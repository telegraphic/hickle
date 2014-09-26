# encoding: utf-8
"""
hickle.py
=============

Created by Danny Price 2012-05-28.

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


"""

import os
import exceptions
import numpy as np
import h5py as h5

__version__ = "1.0.4"
__author__  = "Danny Price"

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

def dumpNdarray(obj, h5f, compression=None):
  """ dumps an ndarray object to h5py file"""
  h5f.create_dataset('data', data=obj, compression=compression)
  h5f.create_dataset('type', data=['ndarray'])
 
def dumpMasked(obj, h5f, compression=None):
  """ dumps an ndarray object to h5py file"""
  h5f.create_dataset('data', data=obj, compression=compression)
  h5f.create_dataset('mask', data=obj.mask, compression=compression)
  h5f.create_dataset('type', data=['masked'])

def dumpList(obj, h5f, compression=None):
  """ dumps a list object to h5py file"""
  h5f.create_dataset('data', data=obj, compression=compression)
  h5f.create_dataset('type', data=['list'])

def dumpSet(obj, h5f, compression=None):
  """ dumps a set object to h5py file"""
  obj = list(obj)
  h5f.create_dataset('data', data=obj, compression=compression)
  h5f.create_dataset('type', data=['set'])
 
def dumpString(obj, h5f, compression=None):
  """ dumps a list object to h5py file"""
  h5f.create_dataset('data', data=[obj], compression=compression)
  h5f.create_dataset('type', data=['string'])

def dumpUnicode(obj, h5f, compression=None):
  """ dumps a list object to h5py file"""
  dt = h5.special_dtype(vlen=unicode)
  ll = len(obj)
  dset = h5f.create_dataset('data', shape=(ll, ), compression=compression, dtype=dt)
  dset[:ll] = obj
  h5f.create_dataset('type', data=['unicode'])

def _dumpDict(dd, hgroup, compression=None):
  for key in dd:
    if type(dd[key]) in (str, int, float, unicode, bool):
        # Figure out type to be stored
        types = { str : 'str', int : 'int', float : 'float', 
                 unicode : 'unicode', bool : 'bool'}
        _key = types.get(type(dd[key]))
        
        # Store along with dtype info
        if _key == 'unicode':
            dd[key] = str(dd[key])

        hgroup.create_dataset("%s"%key, data=[dd[key]], compression=compression)
        hgroup.create_dataset("_%s"%key, data=[_key])
        
    elif type(dd[key]) in (type(np.array([1])), type(np.ma.array([1]))):
 
        if hasattr(dd[key], 'mask'):
            hgroup.create_dataset("_%s"%key, data=["masked"])
            hgroup.create_dataset("%s"%key, data=dd[key].data, compression=compression)
            hgroup.create_dataset("_%s_mask"%key, data=dd[key].mask, compression=compression)
        else:
            hgroup.create_dataset("_%s"%key, data=["ndarray"])
            hgroup.create_dataset("%s"%key, data=dd[key], compression=compression)
    
    elif type(dd[key]) is list:
        hgroup.create_dataset("%s"%key, data=dd[key], compression=compression)
        hgroup.create_dataset("_%s"%key, data=["list"])
    
    elif type(dd[key]) is set:
        hgroup.create_dataset("%s"%key, data=list(dd[key]), compression=compression)
        hgroup.create_dataset("_%s"%key, data=["set"])
    
    elif isinstance(dd[key], dict):
        new_group = hgroup.create_group("%s"%key)
        _dumpDict(dd[key], new_group, compression=compression)       
    
    else:
        print type(dd[key])
        raise NoMatchError

def dumpDict(obj, h5f='', compression=None):
  """ dumps a dictionary to h5py file """
  h5f.create_dataset('type', data=['dict'])
  hgroup = h5f.create_group('data')
  _dumpDict(obj, hgroup, compression=None)

def noMatch(obj, h5f, *args, **kwargs):
  """ If no match is made, raise an exception """
  import cPickle
  pickled_obj = cPickle.dumps(obj)
  h5f.create_dataset('type', data=['pickle'])
  h5f.create_dataset('data', data=[pickled_obj])
  
  print "Warning: %s type not understood, data have been serialized" % type(obj)
  #raise NoMatchError

def dumperLookup(obj):
  """ What type of object are we trying to pickle?
   
  This is a python dictionary based equivalent of a case statement.
  It returns the correct helper function for a given data type.
  """
  t = type(obj)
  
  types = {
     list       : dumpList,
     set        : dumpSet,
     dict       : dumpDict,
     str        : dumpString,
     unicode    : dumpUnicode,
     np.ndarray : dumpNdarray,
     np.ma.core.MaskedArray : dumpMasked,
  }
  
  match = types.get(t, noMatch)
  return match

def dump(obj, file, mode='w', compression=None):
  """ Write a pickled representation of obj to the open file object file. 
  
  Parameters
  ----------
  obj: object
    python object o store in a Hickle
  file: file object, filename string, or h5py.File object
    file in which to store the object. A h5py.File or a filename is also acceptable.
  mode: string
    optional argument, 'r' (read only), 'w' (write) or 'a' (append). Ignored if file is a file object.
  compression: str
    optional argument. Applies compression to dataset. Options: None, gzip, lzf (+ szip, if installed)
  """
  
  try:
      # See what kind of object to dump
      dumper = dumperLookup(obj)
      # Open the file
      h5f = fileOpener(file, mode)
      print "dumping %s to file %s"%(type(obj), repr(h5f))
      dumper(obj, h5f, compression)
      h5f.close()
  except NoMatchError:
      fname = h5f.filename
      h5f.close()
      try:
          os.remove(fname)
      except:
          print "Warning: dump failed. Could not remove %s" % fname
      finally:
          raise NoMatchError

#############
## loaders ##
#############

def load(file, safe=True):
  """ Load a hickle file and reconstruct a python object
  
  Parameters
  ----------
  file: file object, h5py.File, or filename string
  
  safe (bool): Disable automatic depickling of arbitrary python objects. 
  DO NOT set this to False unless the file is from a trusted source.
  (see http://www.cs.jhu.edu/~s/musings/pickle.html for an explanation)
  """
  
  try:
      h5f   = fileOpener(file)
      dtype = h5f["type"][0]
      
      if dtype == 'dict':
          group = h5f["data"]
          data = loadDict(group)
      elif dtype == 'pickle':
          data = loadPickle(h5f, safe)
      elif dtype == 'masked':
          data = np.ma.array(h5f["data"][:], mask=h5f["mask"][:])
      else:
          if dtype in ('string', 'unicode'):
              data  = h5f["data"][0]
          else:
              data  = h5f["data"][:]
  
          types = {
             'list'       : list,
             'set'        : set,
             'unicode'    : unicode,
             'string'     : str,
             'ndarray'    : loadNdarray,
          }
      
          mod = types.get(dtype, noMatch)
          data = mod(data) 
  finally:
      h5f.close()
  return data

def loadPickle(h5f, safe=True):
  """ Deserialize and load a pickled object within a hickle file
  
  WARNING: Pickle has 
  
  Parameters
  ----------
  h5f: h5py.File object
  
  safe (bool): Disable automatic depickling of arbitrary python objects. 
  DO NOT set this to False unless the file is from a trusted source.
  (see http://www.cs.jhu.edu/~s/musings/pickle.html for an explanation)
  """
  
  if not safe:
      import cPickle
      data = h5f["data"][:]
      data = cPickle.loads(data[0])
      return data
  else:
      print "\nWarning: Object is of an unknown type, and has not been loaded"
      print "         for security reasons (it could be malicious code). If"
      print "         you wish to continue, manually set safe=False\n"
      

def loadNdarray(arr):
    """ Load a numpy array """
    # Nothing to be done!
    return arr

def loadDict(group):
    """ Load dictionary """
    
    dd = {}
    for key in group.keys():
        if isinstance(group[key], h5._hl.group.Group):
            new_group = group[key]
            dd[key] = loadDict(new_group)
        elif not key.startswith("_"):
            _key = "_%s" % key
            #print _key, group[_key]
            if group[_key][0] in ('str', 'int', 'float', 'unicode', 'bool'):
                dd[key] = group[key][0]
            elif group[_key][0] == 'masked':
                key_ma     = "_%s_mask" % key
                dd[key] = np.ma.array(group[key][:], mask=group[key_ma])
            else:
                dd[key] = group[key][:]
            
            # Convert numpy constructs back to string
            dtype = group[_key][0]
            types = {'str' : str , 'int' : int, 'float' : float, 
                     'unicode' : unicode, 'bool' : bool, 'list' : list}
            try:
                mod = types.get(dtype)
                dd[key] = mod(dd[key])
            except:
                pass
    return dd

def loadLarge(file):
  """ Load a large hickle file (returns the h5py object not the data) 

  Parameters
  ----------
  file: file object, h5py.File, or filename string  
  """

  h5f = fileOpener(file)
  return h5f

  
  