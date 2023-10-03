# Import numpy and matplotlib -- you shouldn't need any other libraries
import numpy as np
import matplotlib.pyplot as plt
import Q1_1
from IPython import get_ipython


# https://en.wikipedia.org/wiki/Neural_coding
# https://en.wikipedia.org/wiki/Neural_decoding

get_ipython().magic('clear')
get_ipython().magic('reset -f')

np.random.seed(80)

def neurons_vs_error(Q_number, ro_scalar):
    
    Ns = [4, 8, 16, 32, 64, 128, 256, 512]
    S = 41

    noise_errors = []
    distortion_errors = []
    
    x_linspace = np.linspace(-1, 1, S)
    X = np.matrix(x_linspace)
    
    for N in Ns:
        
        noise_to_avg = []
        distortion_to_avg = []
        
        for i in range(10):
        
            A = np.matrix(Q1_1.generate_tuning_curves(x_linspace, N))
            ro = ro_scalar * np.amax(A)
            A_noisy = A + np.random.normal(0, ro, A.shape)
            
            regularizer = N*pow(ro, 2) * np.eye(N, N)
            D_reg = np.transpose(np.linalg.inv(A * A.T + regularizer) * A * np.transpose(X))
            
            x_fit = D_reg * A_noisy
            
            noise_error = np.sum(np.square(D_reg)) * pow(ro, 2) / 2
            distortion_error = np.sum(np.square(X - x_fit)) / 2
            
            noise_to_avg.append(noise_error)
            distortion_to_avg.append(distortion_error)
                        
            #Q1_1.plot_decoded_vs_ideal(x_linspace, x_fit, "1.2A)", "Noise and Regularization, " + str(N) + " Neurons")
            
        noise_errors.append(sum(noise_to_avg) / len(noise_to_avg))
        distortion_errors.append(sum(distortion_to_avg) / len(distortion_to_avg))
    
    plt.title(Q_number + "Errors vs Number of Neurons, \nRo Scalar = " + str(ro_scalar))
    plt.xlabel("Number of Neurons")
    plt.ylabel("Square Error")
    plt.ylim([1e-6, 1e1])
    plt.grid()
    plt.loglog(Ns, noise_errors, label="Error due to Noise")
    plt.loglog(Ns, distortion_errors, label="Error due to Distortion")
    plt.loglog(Ns, [1 / n for n in Ns], label="1/n")
    plt.loglog(Ns, [1 / (n*n) for n in Ns], label="1/n^2")
    
    plt.legend()
    plt.show()
    
    
# 1.2 A) [DONE]
print("1.2A)")
neurons_vs_error("1.2A) ", 0.1)

# 1.2B) [DONE]
print("1.2B)")
neurons_vs_error("1.2B) ", 0.01)

print("C) ")
# 1.2C) [DONE]
C= """

In both plots, error due to noise and distortion both appear to be proportional
to 1 / n. This is not necessarily expeceted, as it was shown in lecture
that error due to distortion tends to be proportaional to 1/n^2.
In both plots, error decreased as number of neurons increased,
but only approaches 0.

From both plots, we can see that distortion is a far greater source of error than noise.
However, if a ro scalar of 10 is used, it can be shown that there is no 
guarentee that distortion is a greater consideration than noise.

"""

print(C)


