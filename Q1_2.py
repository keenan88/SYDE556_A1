# Import numpy and matplotlib -- you shouldn't need any other libraries
import numpy as np
import matplotlib.pyplot as plt
import Q1_1
from IPython import get_ipython


# https://en.wikipedia.org/wiki/Neural_coding
# https://en.wikipedia.org/wiki/Neural_decoding

get_ipython().magic('clear')
get_ipython().magic('reset -f')

np.random.seed(18945)

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
    
    """
    plt.title(Q_number + "Errors vs Number of Neurons, \nRo Scalar = " + str(ro_scalar))
    plt.xlabel("Number of Neurons")
    plt.ylabel("Square Error")
    plt.grid()
    plt.plot(Ns, noise_errors, label="Error due to Noise")
    plt.plot(Ns, distortion_errors, label="Error due to Distortion")
    plt.plot(Ns, [1 / n for n in Ns], label="1/n")
    plt.plot(Ns, [1 / (n*n) for n in Ns], label="1/n^2")
    plt.xlim([0, 150])
    plt.legend()
    plt.show()
    """
    
# 1.2 A) [DONE]
print("1.2A)")
neurons_vs_error("1.2A) ", 0.1)

# 1.2B) [DONE]
print("1.2B)")
neurons_vs_error("1.2B) ", 0.01)

# 1.2C) [DONE]
"""

From the plot, we can see that distortion is a far greater source of error than noise.
We can also see that error due to distortion and noise both decrease roughly
in proprortion to 1/n, where n is the number of neurons.

Interestingly, we also see that distortion error changes with noise, not just noise error.

"""


