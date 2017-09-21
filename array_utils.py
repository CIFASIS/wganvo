# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import numpy as np

def save(name, array):
    np.savetxt(name + '.txt', array, delimiter=' ', fmt='%1.6f')
    
def save_as_list(name, array):
    np.savetxt(name + '.txt', array.ravel(), delimiter=' ', fmt='%1.6f')
    
def list_to_array(list):
    return np.array(list)