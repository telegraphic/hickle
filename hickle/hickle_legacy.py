# encoding: utf-8
"""
# hickle_legacy.py

Created by Danny Price 2012-05-28.

Hickle is a HDF5 based clone of Pickle. Instead of serializing to a
pickle file, Hickle dumps to a HDF5 file. It is designed to be as similar
to pickle in usage as possible.

## Notes

This is a legacy handler, for hickle v1 files.
If V2 reading fails, this will be called as a fail-over.

"""

import os
import sys
import numpy as np
import h5py as h5

if sys.version_info.major == 3:
    NoneType = type(None)
else:
    from types import NoneType

__version__ = "1.3.0"
__author__ = "Danny Price"

####################
## Error handling ##
####################


class FileError(Exception):
    """ An exception raised if the file is fishy"""

    def __init__(self):
        return

    def __str__(self):
        print("Error: cannot open file. Please pass either a filename string, a file object, "
              "or a h5py.File")


class NoMatchError(Exception):
    """ An exception raised if the object type is not understood (or supported)"""

    def __init__(self):
        return

    def __str__(self):
        print("Error: this type of python object cannot be converted into a hickle.")


class ToDoError(Exception):
    """ An exception raised for non-implemented functionality"""

    def __init__(self):
        return

    def __str__(self):
        print("Error: this functionality hasn't been implemented yet.")


class H5GroupWrapper(h5.Group):
    def create_dataset(self, *args, **kwargs):
        kwargs['track_times'] = getattr(self, 'track_times', True)
        return super(H5GroupWrapper, self).create_dataset(*args, **kwargs)
    
    def create_group(self, *args, **kwargs):
        group = super(H5GroupWrapper, self).create_group(*args, **kwargs)
        group.__class__ = H5GroupWrapper
        group.track_times = getattr(self, 'track_times', True)
        return group


class H5FileWrapper(h5.File):
    def create_dataset(self, *args, **kwargs):
        kwargs['track_times'] = getattr(self, 'track_times', True)
        return super(H5FileWrapper, self).create_dataset(*args, **kwargs)

    def create_group(self, *args, **kwargs):
        group = super(H5FileWrapper, self).create_group(*args, **kwargs)
        group.__class__ = H5GroupWrapper
        group.track_times = getattr(self, 'track_times', True)
        return group


def file_opener(f, mode='r', track_times=True):
    """ A file opener helper function with some error handling.
  
  This can open files through a file object, a h5py file, or just the filename.
  """
    # Were we handed a file object or just a file name string?
    if isinstance(f, file):
        filename, mode = f.name, f.mode
        f.close()
        h5f = h5.File(filename, mode)

    elif isinstance(f, h5._hl.files.File):
        h5f = f
    elif isinstance(f, str):
        filename = f
        h5f = h5.File(filename, mode)
    else:
        raise FileError
   
    h5f.__class__ = H5FileWrapper
    h5f.track_times = track_times
    return h5f


#############
## dumpers ##
#############

def dump_ndarray(obj, h5f, **kwargs):
    """ dumps an ndarray object to h5py file"""
    h5f.create_dataset('data', data=obj, **kwargs)
    h5f.create_dataset('type', data=['ndarray'])


def dump_np_dtype(obj, h5f, **kwargs):
    """ dumps an np dtype object to h5py file"""
    h5f.create_dataset('data', data=obj)
    h5f.create_dataset('type', data=['np_dtype'])


def dump_np_dtype_dict(obj, h5f, **kwargs):
    """ dumps an np dtype object within a group"""
    h5f.create_dataset('data', data=obj)
    h5f.create_dataset('_data', data=['np_dtype'])


def dump_masked(obj, h5f, **kwargs):
    """ dumps an ndarray object to h5py file"""
    h5f.create_dataset('data', data=obj, **kwargs)
    h5f.create_dataset('mask', data=obj.mask, **kwargs)
    h5f.create_dataset('type', data=['masked'])


def dump_list(obj, h5f, **kwargs):
    """ dumps a list object to h5py file"""

    # Check if there are any numpy arrays in the list
    contains_numpy = any(isinstance(el, np.ndarray) for el in obj)

    if contains_numpy:
        _dump_list_np(obj, h5f, **kwargs)
    else:
        h5f.create_dataset('data', data=obj, **kwargs)
        h5f.create_dataset('type', data=['list'])


def _dump_list_np(obj, h5f, **kwargs):
    """ Dump a list of numpy objects to file """

    np_group = h5f.create_group('data')
    h5f.create_dataset('type', data=['np_list'])

    ii = 0
    for np_item in obj:
        np_group.create_dataset("%s" % ii, data=np_item, **kwargs)
        ii += 1


def dump_tuple(obj, h5f, **kwargs):
    """ dumps a list object to h5py file"""

    # Check if there are any numpy arrays in the list
    contains_numpy = any(isinstance(el, np.ndarray) for el in obj)

    if contains_numpy:
        _dump_tuple_np(obj, h5f, **kwargs)
    else:
        h5f.create_dataset('data', data=obj, **kwargs)
        h5f.create_dataset('type', data=['tuple'])


def _dump_tuple_np(obj, h5f, **kwargs):
    """ Dump a tuple of numpy objects to file """

    np_group = h5f.create_group('data')
    h5f.create_dataset('type', data=['np_tuple'])

    ii = 0
    for np_item in obj:
        np_group.create_dataset("%s" % ii, data=np_item, **kwargs)
        ii += 1


def dump_set(obj, h5f, **kwargs):
    """ dumps a set object to h5py file"""
    obj = list(obj)
    h5f.create_dataset('data', data=obj, **kwargs)
    h5f.create_dataset('type', data=['set'])


def dump_string(obj, h5f, **kwargs):
    """ dumps a list object to h5py file"""
    h5f.create_dataset('data', data=[obj], **kwargs)
    h5f.create_dataset('type', data=['string'])


def dump_none(obj, h5f, **kwargs):
    """ Dump None type to file """
    h5f.create_dataset('data', data=[0], **kwargs)
    h5f.create_dataset('type', data=['none'])


def dump_unicode(obj, h5f, **kwargs):
    """ dumps a list object to h5py file"""
    dt = h5.special_dtype(vlen=unicode)
    ll = len(obj)
    dset = h5f.create_dataset('data', shape=(ll, ), dtype=dt, **kwargs)
    dset[:ll] = obj
    h5f.create_dataset('type', data=['unicode'])


def _dump_dict(dd, hgroup, **kwargs):
    for key in dd:
        if type(dd[key]) in (str, int, float, unicode, bool):
            # Figure out type to be stored
            types = {str: 'str', int: 'int', float: 'float',
                     unicode: 'unicode', bool: 'bool', NoneType: 'none'}
            _key = types.get(type(dd[key]))

            # Store along with dtype info
            if _key == 'unicode':
                dd[key] = str(dd[key])

            hgroup.create_dataset("%s" % key, data=[dd[key]], **kwargs)
            hgroup.create_dataset("_%s" % key, data=[_key])

        elif type(dd[key]) in (type(np.array([1])), type(np.ma.array([1]))):

            if hasattr(dd[key], 'mask'):
                hgroup.create_dataset("_%s" % key, data=["masked"])
                hgroup.create_dataset("%s" % key, data=dd[key].data, **kwargs)
                hgroup.create_dataset("_%s_mask" % key, data=dd[key].mask, **kwargs)
            else:
                hgroup.create_dataset("_%s" % key, data=["ndarray"])
                hgroup.create_dataset("%s" % key, data=dd[key], **kwargs)

        elif type(dd[key]) is list:
            hgroup.create_dataset("%s" % key, data=dd[key], **kwargs)
            hgroup.create_dataset("_%s" % key, data=["list"])
            
        elif type(dd[key]) is tuple:
            hgroup.create_dataset("%s" % key, data=dd[key], **kwargs)
            hgroup.create_dataset("_%s" % key, data=["tuple"])

        elif type(dd[key]) is set:
            hgroup.create_dataset("%s" % key, data=list(dd[key]), **kwargs)
            hgroup.create_dataset("_%s" % key, data=["set"])

        elif isinstance(dd[key], dict):
            new_group = hgroup.create_group("%s" % key)
            _dump_dict(dd[key], new_group, **kwargs)
            
        elif type(dd[key]) is NoneType:
            hgroup.create_dataset("%s" % key, data=[0], **kwargs)
            hgroup.create_dataset("_%s" % key, data=["none"])
            
        else:
            if type(dd[key]).__module__ == np.__name__:
                #print type(dd[key])
                hgroup.create_dataset("%s" % key, data=dd[key])
                hgroup.create_dataset("_%s" % key, data=["np_dtype"])
                #new_group = hgroup.create_group("%s" % key)
                #dump_np_dtype_dict(dd[key], new_group)
            else:
                raise NoMatchError


def dump_dict(obj, h5f='', **kwargs):
    """ dumps a dictionary to h5py file """
    h5f.create_dataset('type', data=['dict'])
    hgroup = h5f.create_group('data')
    _dump_dict(obj, hgroup, **kwargs)


def no_match(obj, h5f, *args, **kwargs):
    """ If no match is made, raise an exception """
    try:
        import dill as cPickle
    except ImportError:
        import cPickle

    pickled_obj = cPickle.dumps(obj)
    h5f.create_dataset('type', data=['pickle'])
    h5f.create_dataset('data', data=[pickled_obj])

    print("Warning: %s type not understood, data have been serialized" % type(obj))
    #raise NoMatchError


def dumper_lookup(obj):
    """ What type of object are we trying to pickle?
   
  This is a python dictionary based equivalent of a case statement.
  It returns the correct helper function for a given data type.
  """
    t = type(obj)

    types = {
        list: dump_list,
        tuple: dump_tuple,
        set: dump_set,
        dict: dump_dict,
        str: dump_string,
        unicode: dump_unicode,
        NoneType: dump_none,
        np.ndarray: dump_ndarray,
        np.ma.core.MaskedArray: dump_masked,
        np.float16: dump_np_dtype,
        np.float32: dump_np_dtype,
        np.float64: dump_np_dtype,
        np.int8: dump_np_dtype,
        np.int16: dump_np_dtype,
        np.int32: dump_np_dtype,
        np.int64: dump_np_dtype,
        np.uint8: dump_np_dtype,
        np.uint16: dump_np_dtype,
        np.uint32: dump_np_dtype,
        np.uint64: dump_np_dtype,
        np.complex64: dump_np_dtype,
        np.complex128: dump_np_dtype,
    }

    match = types.get(t, no_match)
    return match


def dump(obj, file, mode='w', track_times=True, **kwargs):
    """ Write a pickled representation of obj to the open file object file.
  
  Parameters
  ----------
  obj: object
    python object o store in a Hickle
  file: file object, filename string, or h5py.File object
    file in which to store the object. A h5py.File or a filename is also acceptable.
  mode: string
    optional argument, 'r' (read only), 'w' (write) or 'a' (append). Ignored if file
    is a file object.
  compression: str
    optional argument. Applies compression to dataset. Options: None, gzip, lzf (+ szip,
    if installed)
  track_times: bool
    optional argument. If set to False, repeated hickling will produce identical files.
  """

    try:
        # See what kind of object to dump
        dumper = dumper_lookup(obj)
        # Open the file
        h5f = file_opener(file, mode, track_times)
        print("dumping %s to file %s" % (type(obj), repr(h5f)))
        dumper(obj, h5f, **kwargs)
        h5f.close()
    except NoMatchError:
        fname = h5f.filename
        h5f.close()
        try:
            os.remove(fname)
        except:
            print("Warning: dump failed. Could not remove %s" % fname)
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
        h5f = file_opener(file)
        dtype = h5f["type"][0]

        if dtype == 'dict':
            group = h5f["data"]
            data = load_dict(group)
        elif dtype == 'pickle':
            data = load_pickle(h5f, safe)
        elif dtype == 'np_list':
            group = h5f["data"]
            data = load_np_list(group)
        elif dtype == 'np_tuple':
            group = h5f["data"]
            data = load_np_tuple(group)
        elif dtype == 'masked':
            data = np.ma.array(h5f["data"][:], mask=h5f["mask"][:])
        elif dtype == 'none':
            data = None
        else:
            if dtype in ('string', 'unicode'):
                data = h5f["data"][0]
            else:
                try:
                    data = h5f["data"][:]
                except ValueError:
                    data = h5f["data"]
            types = {
                'list': list,
                'set': set,
                'unicode': unicode,
                'string': str,
                'ndarray': load_ndarray,
                'np_dtype': load_np_dtype
            }

            mod = types.get(dtype, no_match)
            data = mod(data)
    finally:
        if 'h5f' in locals():
            h5f.close()
    return data


def load_pickle(h5f, safe=True):
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
        try:
            import dill as cPickle
        except ImportError:
            import cPickle

        data = h5f["data"][:]
        data = cPickle.loads(data[0])
        return data
    else:
        print("\nWarning: Object is of an unknown type, and has not been loaded")
        print("         for security reasons (it could be malicious code). If")
        print("         you wish to continue, manually set safe=False\n")


def load_np_list(group):
    """ load a numpy list """
    np_list = []
    for key in sorted(group.keys()):
        data = group[key][:]
        np_list.append(data)
    return np_list


def load_np_tuple(group):
    """ load a tuple containing numpy arrays """
    return tuple(load_np_list(group))


def load_ndarray(arr):
    """ Load a numpy array """
    # Nothing to be done!
    return arr


def load_np_dtype(arr):
    """ Load a numpy array """
    # Just return first value
    return arr.value


def load_dict(group):
    """ Load dictionary """

    dd = {}
    for key in group.keys():
        if isinstance(group[key], h5._hl.group.Group):
            new_group = group[key]
            dd[key] = load_dict(new_group)
        elif not key.startswith("_"):
            _key = "_%s" % key

            if group[_key][0] == 'np_dtype':
                dd[key] = group[key].value
            elif group[_key][0] in ('str', 'int', 'float', 'unicode', 'bool'):
                dd[key] = group[key][0]
            elif group[_key][0] == 'masked':
                key_ma = "_%s_mask" % key
                dd[key] = np.ma.array(group[key][:], mask=group[key_ma])
            else:
                dd[key] = group[key][:]

            # Convert numpy constructs back to string
            dtype = group[_key][0]
            types = {'str': str, 'int': int, 'float': float,
                     'unicode': unicode, 'bool': bool, 'list': list, 'none' : NoneType}
            try:
                mod = types.get(dtype)
                if dtype == 'none':
                    dd[key] = None
                else:
                    dd[key] = mod(dd[key])
            except:
                pass
    return dd


def load_large(file):
    """ Load a large hickle file (returns the h5py object not the data)

  Parameters
  ----------
  file: file object, h5py.File, or filename string  
  """

    h5f = file_opener(file)
    return h5f
