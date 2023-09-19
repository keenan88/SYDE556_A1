# Import numpy and matplotlib -- you shouldn't need any other libraries
import numpy as np
import matplotlib.pyplot as plt
import Q1
from IPython import get_ipython


# https://en.wikipedia.org/wiki/Neural_coding
# https://en.wikipedia.org/wiki/Neural_decoding

get_ipython().magic('clear')
get_ipython().magic('reset -f')


Ns = [4, 8, 16, 32, 64, 128, 256, 512]
S = 41

x_linspace = np.linspace(-1, 1, S)

for N in Ns[0:1]:
    
    A = np.matrix(Q1.generate_tuning_curves(x_linspace, N))
    X = np.matrix(x_linspace)
    
    D = np.transpose(np.linalg.inv(A * np.transpose(A)) * A * np.transpose(X))
    
    A_noisy = A + np.random.normal(0, 0.2*np.amax(A), A.shape)






