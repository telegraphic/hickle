
types_dict = {}

hkl_types_dict = {}

types_not_to_sort = ('dict', 'csr_matrix', 'csc_matrix', 'bsr_matrix')

def return_first(x):
    return x[0]

container_types_dict = {
    "<type 'list'>": list,
    "<type 'tuple'>": tuple,
    "<type 'set'>": set,
    "csr_matrix":  return_first,
    "csc_matrix": return_first,
    "bsr_matrix": return_first
    }

# Technically, any hashable object can be used, for now sticking with built-in types
container_key_types_dict = {
    "<type 'str'>": str,
    "<type 'unicode'>": unicode,
    "<type 'float'>": float,
    "<type 'bool'>": bool,
    "<type 'int'>": int,
    "<type 'long'>": long,
    "<type 'complex'>": complex
    }

# Add loaders for built-in python types
from .loaders.load_python import types_dict as py_types_dict
from .loaders.load_python import hkl_types_dict as py_hkl_types_dict

types_dict.update(py_types_dict)
hkl_types_dict.update(py_hkl_types_dict)

# Add loaders for numpy types
from .loaders.load_numpy import  types_dict as np_types_dict
from .loaders.load_numpy import  hkl_types_dict as np_hkl_types_dict
types_dict.update(np_types_dict)
hkl_types_dict.update(np_hkl_types_dict)

# Add loaders for astropy
try:
    from .loaders.load_astropy import types_dict as ap_types_dict
    from .loaders.load_astropy import hkl_types_dict as ap_hkl_types_dict
    types_dict.update(ap_types_dict)
    hkl_types_dict.update(ap_hkl_types_dict)

except ImportError:
    raise

