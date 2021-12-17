# encoding: utf-8
"""
#fileio.py

contains functions, classes and constants related to file management.
These functions may also be used by loader modules even though currently
no related use-case is known and storing dicts as independent files as
requested by @gpetty in issue #133 is better handled on hdf5 or h5py level
and not on hickle level.
"""

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

    Parameters
    ----------
    f (file or file like):
        file or file like object to which hickle shall
        dump data to.

    *args (tuple):
        list of one or more tuples containing the commands to be checked 
        in replacement tests for IOBase.isreadable, IOBase.isseekable or
        IOBase.iswriteable and the arguments required to perform the tests
        
        Note: **kwargs not supported

    Returns
    -------
        function to be called in replacement of any of not implemented
        IOBase.isreadable, IOBase.isseekable or IOBase.isreadable

    Example
    -------
        if not getattr(f, 'isreadable', not_io_base_like(f, 'read', 0))():
            raise ValueError("Not a readable file or file like object")
    """
    def must_test():
        if not args:
            return False
        for cmd,*call_args in args:
            cmd = getattr(f,cmd,None)
            if not cmd:
                return False
            try:
                cmd(*call_args)
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
    f (file, file-like, h5py.File, str, (file,str),{'file':file,'name':str} ):
        File to open for dumping or loading purposes.
        str:
            the path of the HDF5-file that must be used.
        `~h5py.Group`:
             the group (or file) in an open HDF5-file that must be used.
        file, file-like: 
            file or like object which provides `read`, `seek`, `tell` and write methods
        tuple:
            two element tuple with the first being the file or file like object
            to dump to and the second the filename to be used instead of 'filename'
            parameter
        dict:
            dictionary with 'file' and 'name' items

    path (str):
        Path within HDF5-file or group to dump to/load from.

    mode (str): optional
        string indicating how the file shall be opened. For details see Python `open`.
        
        Note: The 'b' flag is optional as all files are a and have to be opened in
            binary mode.
    
    filename (str): optional
        The name of the file. Ignored when f is `str` or `h5py.File` object.

    Returns
    -------
    tuple containing (file, path, closeflag)

    file (h5py.File):
        The h5py.File object the data is to be dumped to or loaded from

    path (str):
        Absolute path within HDF5-file or group to dump to/load from.

    closeflag:
        True .... file was opened by file_opener and must be closed by caller.
        False ... file shall not be closed by caller unless opened by caller

    Raises
    ------
    CloseFileError:
        If passed h5py.File, h5py.Group or h5py.Dataset object is not
        accessible. This in most cases indicates that underlying HDF5.File,
        file or file-like object has already been closed.

    FileError
        If passed file or file-like object is not opened for reading or
        in addition for writing in case mode corresponds to any
        of 'w', 'w+', 'x', 'x+' or a.

    ValueError:
        If anything else than str, bytes or None specified for filename
    """

    # Make sure that the given path always starts with '/'
    if not path.startswith('/'):
        path = "/%s" % path

    # Were we handed a file object or just a file name string?
    if isinstance(f, (str, Path)):
        return h5.File(f, mode.replace('b','')),path,True
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
        if f.mode == 'r' and ( mode[0] != 'r' or '+' in mode[1:] ):
            raise FileError( "HDF5 file '{}' not opened for writing".format(f.filename))

        # Since this file was already open, do not close the file afterward
        return f,''.join((base_path,path.rstrip('/'))),False

    # get the name of the file
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
            "Please pass either a filename string, a pathlib.Path, a file or file like object, "
            "an opened h5py.File or h5py.Group or h5py.Dataset there out of.".format(
                "'{}' ".format(filename) if isinstance(filename,(str,bytes)) and filename else ''
            )
        )
    # file and file-like object must be at least read and seekable. This means
    # they have as specified by IOBase provide read, seek and tell methods
    if (
        getattr(f,'readable',not_io_base_like(f,('read',0)))() and
        getattr(f,'seekable',not_io_base_like(f,('seek',0),('tell',)))()
    ):

        # if file is to be opened for cration, writing or appending check if file, file-like
        # object is writable or at least provides write method for writing binary data
        if mode[0] in 'xwa' or ( '+' in mode[1:] and mode[0] == 'r' ):
            if not getattr(f,'writeable',not_io_base_like(f,('write',b'')))():
                raise FileError(
                    "file '{}' not writable. Please pass either a filename string, "
                    "a pathlib.Path,  a file or file like object, "
                    "an opened h5py.File or h5py.Group or h5py.Dataset there out of.".format(filename)
                )
        elif mode[0] != 'r':
            raise ValueError(
                "invalid file mode must be one out of 'w','w+','x','x+','r','r+','a'. "
                "at max including a 'b' which will be ignored"
        )
        return h5.File(
            f,
            mode.replace('b','') if mode[0] == 'r'  else mode[0],
            driver='fileobj',
            fileobj = f
        ), path, True

    raise FileError(
        "'file_obj' must be a valid path string, pahtlib.Path, h5py.File, h5py.Group, "
        "h5py.Dataset, file  or file like object'"
    )
