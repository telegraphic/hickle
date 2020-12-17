"""
# Generate_test_hickle.py

Helper to generate test hickle files for a given hickle version

Bash command to generate things:
> VER=2.1.0; pip uninstall hickle -y; pip install hickle==$VER; python generate_test_hickle.py $VER
"""
import hickle as hkl
import numpy as np
import sys

ver_str = sys.argv[1].replace('.', '_')

fn_out = 'hickle_%s.hkl' % ver_str

dd = {
    'dog_breed': b'Chihuahua',
    'age': 10,
    'height': 1.1,
    'nums': [1, 2, 3],
    'narr': np.array([1, 2, 3]),
}

print("Dumping %s..." % fn_out)
hkl.dump(dd, fn_out, path='test')

