# encoding: utf-8
"""
Plugin BitShuffleLz4

Based on hickle.dump and hdf5plugin

Hdf5plugin background information: https://github.com/kiyo-masui/bitshuffle
"""

import hickle


class BitShuffleLz4:

    
    def __init__(self):
        import hdf5plugin
        self.compression = hdf5plugin.Bitshuffle(nelems=0, lz4=True)["compression"]
        self.compression_opts = hdf5plugin.Bitshuffle(nelems=0, lz4=True)["compression_opts"]


    def dump(self, py_obj, file_obj, mode="w", path="/",*,filename = None, options = {}):
        """
        hickle dump + BitShuffle + LZ4
        
        See hickle.py::dump() for parameter documentation.
        """
        hickle.dump(py_obj, 
                    file_obj, 
                    mode=mode, 
                    path=path,
                    filename = filename, 
                    options = options,
                    compression = self.compression,
                    compression_opts = self.compression_opts)


    def load(self, file_obj, path="/", safe=True):
        """
        hickle load + BitShuffle + LZ4
        
        See hickle.py::load() for parameter documentation.
        """
        return hickle.load(file_obj,
                           path=path,
                           safe=safe)
