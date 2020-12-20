# %% IMPORTS
# Built-in imports
import re
import operator
import typing
import types
import collections
import numbers
import h5py as h5
import os.path as os_path
from pathlib import Path


class FileError(Exception):
    """ An exception raised if the file is fishy """


class ClosedFileError(Exception):
    """ An exception raised if the file is fishy """


def not_io_base_like(f,*args):
    """
    creates function which can be used in replacement for
    IOBase.isreadable, IOBase.isseekable and IOBase.iswriteable
    methods in case f would not provide any of them.

    Parameters:
    ===========
        f (file or file like):
            file or file like object to which hickle shall
            dump data too.

        *args (tuple):
            tuple containg either 2 or 4 elements
            inices 0, 2 :
                name of methods to be tested
            odd indices  1, 3 :
                *args tuples passed to tested methods
                ( **kwargs not supported)
                ( optional if last item  in tuple)

    Returns:
    ========
        function to be called in replacement if any of IOBase.isreadable,
        IOBase.isseekable or IOBase.isreadable wold be not implemented

    Example:
    ========
        if not getattr(f,'read',not_io_base_like(f,'read',0))():
            raise ValueError("Not a reaable file or file like object")
    """
    def must_test():
        if not args: # pragma: nocover
            return False
        cmd = getattr(f,args[0],None)
        if not cmd:
            return False
        try:
            cmd(*args[1:2])
        except:
            return False
        if len(args) < 3:
            return True
        cmd = getattr(f,args[2],None)
        if not cmd:
            return False
        try:
            cmd(*args[3:4])
        except:
            return False
        return True
    return must_test

def file_opener(f, path, mode='r',filename = None):
    """
    A file opener helper function with some error handling.
    This can open files through a file object, an h5py file, or just the
    filename.

    Parameters
    ----------
    f : file object, str or :obj:`~h5py.Group` object
        File to open for dumping or loading purposes.
        If str, `file_obj` provides the path of the HDF5-file that must be
        used.
        If :obj:`~h5py.Group`, the group (or file) in an open
        HDF5-file that must be used.
    path : str
        Path within HDF5-file or group to dump to/load from.
    mode : str, optional
        Accepted values are 'r' (read only), 'w' (write; default) or 'a'
        (append).
        Ignored if file is a file object.

    """

    # Make sure that the given path always starts with '/'
    if not path.startswith('/'):
        path = "/%s" % path

    # Were we handed a file object or just a file name string?
    if isinstance(f, (str, Path)):
        return h5.File(f, mode[:(-1 if mode[-1] == 'b' else None)]),path,True
    if isinstance(f, h5.Group):
        if not f:
            raise ClosedFileError(
                "HDF5 file {}has been closed or h5py.Group or h5py.Dataset are not accessible. "
                "Please pass either a filename string, a pathlib.Path, a file or file like object, "
                "an opened h5py.File or h5py.Group or h5py.Dataset there outof.".format(
                    "'{}' ".format(filename) if isinstance(filename,(str,bytes)) and filename else ''
                )
            )
        base_path = f.name
        if not isinstance(f,h5.File):
            f = f.file
        if mode[0] == 'w' and f.mode != 'r+':
            raise FileError( "HDF5 file '{}' not opened for writing".format(f.filename))

        # Since this file was already open, do not close the file afterward
        return f,''.join((base_path,path.rstrip('/'))),False

    if not isinstance(filename,(str,bytes)):
        if filename is not None:
            raise ValueError("'filename' must be of type 'str' or 'bytes'")
        if isinstance(f,(tuple,list)) and len(f) > 1:
            f,filename = f[:2]
        elif isinstance(f,dict):
            f,filename = f['file'],f['name']
        else:
            filename = getattr(f,'filename',None)
            if filename is None:
                filename = getattr(f,'name',None)
                if filename is None:
                    filename = repr(f)
    if getattr(f,'closed',False):
        raise ClosedFileError(
            "HDF5 file {}has been closed or h5py.Group or h5py.Dataset are not accessible. "
            "Please pass either a filename string, a pathlib.Paht, a file or file like object, "
            "an opened h5py.File or h5py.Group or h5py.Dataset there outof.".format(
                "'{}' ".format(filename) if isinstance(filename,(str,bytes)) and filename else ''
            )
        )
    if (
        getattr(f,'readable',not_io_base_like(f,'read',0))() and
        getattr(f,'seekable',not_io_base_like(f,'seek',0,'tell'))()
    ):

        if len(mode) > 1 and mode[1] == '+':
            if not getattr(f,'writeable',not_io_base_like(f,'write',b''))():
                raise FileError(
                    "file '{}' not writable. Please pass either a filename string, "
                    "a pathlib.Path,  a file or file like object, "
                    "an opened h5py.File or h5py.Group or h5py.Dataset there outof.".format(filename)
                )
        if ( mode[0] != 'r' ):
            if mode[0] not in 'xwa':
                raise ValueError("invalid file mode must be one outof 'w','w+','x','x+','r','r+','a'. A trailing 'b' is ignored")
            if not getattr(f,'writeable',not_io_base_like(f,'write',b''))():
                raise FileError(
                    "file '{}' not writable. Please pass either a filename string, "
                    "a pathlib.Path, a file or file like object, "
                    "an opened h5py.File or h5py.Group or h5py.Dataset there outof.".format(filename)
                )
        return h5.File(
            f,
            mode[:( 1 if mode[0] == 'w' else ( -1 if mode[-1] == 'b' else None ) )],
            driver='fileobj',
            fileobj = f
        ), path, True

    if mode[0] == 'w' and getattr(f,'mode','')[:2] not in ('r+','w+','a','x+',''):
        raise FileError( "file or file like object '{}' not opened for reading and writing ".format(filename))
        
    raise FileError(
        "'file_obj' must be a valid path string, pahtlib.Path, h5py.File, h5py.Group, h5py.Dataset, file  or file like object'"
    )
