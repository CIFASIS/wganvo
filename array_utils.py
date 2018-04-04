# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import numpy as np

def save(name, array, fmt='%1.6f'):
    np.savetxt(name + '.txt', array, delimiter=' ', fmt=fmt)

def load(name):
    return np.loadtxt(name, delimiter=' ')
    
def save_as_list(name, array, fmt='%1.6f'):
    np.savetxt(name + '.txt', array.ravel(), delimiter=' ', fmt=fmt)
    
def list_to_array(list):
    return np.array(list)
