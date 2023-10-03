# Import numpy and matplotlib -- you shouldn't need any other libraries
import numpy as np
import matplotlib.pyplot as plt
from math import exp
from IPython import get_ipython
import Q1_1
from math import sqrt


get_ipython().magic('clear')
get_ipython().magic('reset -f')

Tref = 2 / 1000 # Converted to seconds
Trc = 20 / 1000 # Converted to seconds

def get_G_inv(a):
    J = 1 / (-exp((Tref - 1/a) / Trc) + 1)
    
    return J

def G_LIF(alpha, x, encoder, J_bias):
    
    J = alpha * x * encoder + J_bias
    
    if J > 1:
        G = 1 / (Tref - Trc * np.log(1 - 1/J))
    else:
        G = 0
        
    return G
    
def get_alpha(zeta, a_max):
    alpha = (get_G_inv(a_max) - 1) / (1 - zeta)
    
    return alpha
    
def get_J_bias(alpha, zeta):
    return 1 - alpha * zeta

def generate_LIF_tuning_curves(x_linspace, num_curves):
    
    tuning_curves = []
    
    
    for i in range(num_curves):
        
        a_max = np.random.uniform(100, 200)
        zeta = np.random.uniform(-0.95, 0.95)
        encoder = np.random.choice(np.array([-1, 1]))

        alpha = get_alpha(zeta, a_max)
        J_bias = get_J_bias(alpha, zeta)
        
        tuning_curve = []
        
        for x in x_linspace:
            a = G_LIF(alpha, x, encoder, J_bias)
            tuning_curve.append(a)
        
        tuning_curves.append(np.array(tuning_curve))

    return tuning_curves

def get_RMSE_matrix(mat1, mat2):
    return round(sqrt(np.mean(np.square(mat1 - mat2))), 3)

if __name__ == "__main__":
    np.random.seed(18945)
    
    N = 16
    S = 41
    x_linspace = np.linspace(-1, 1, S)
    
    # 1.3 B) [DONE]
    print("1.3B)")
    
    tuning_curves = generate_LIF_tuning_curves(x_linspace, N)
    
    for tuning_curve in tuning_curves:
        plt.plot(x_linspace, tuning_curve)

    plt.title("1.3B) 16 LIF Tuning Curves")
    plt.xlabel("Stimulus")
    plt.ylabel("Response")
    plt.xlim([-1, 1])
    plt.grid()
    plt.show() 
    
    # 1.3 C) [DONE]
    print("1.3C)")
    
    A = np.matrix(tuning_curves)
    X = np.matrix(x_linspace)
    
    # No Noise in A, D not regularized
    D = np.transpose(np.linalg.inv(A * np.transpose(A)) * A * np.transpose(X))
    X_hat = D*A
    Q1_1.plot_decoded_vs_ideal(x_linspace, X_hat, "1.3)", "No Noise No Regularization")
    
    rmse_no_noise_no_reg = get_RMSE_matrix(X, X_hat)
    print("RMSE without noise without regularization: ", rmse_no_noise_no_reg)
    
    # Noise in A, D not regularized
    ro = 0.2*np.amax(A)
    A_noisy = A + np.random.normal(0, ro, A.shape)
    X_hat_noisy = D * A_noisy
    Q1_1.plot_decoded_vs_ideal(x_linspace, X_hat_noisy, "1.3)", "Noise Without Regularization")
    
    rmse_with_noise_no_reg = get_RMSE_matrix(X, X_hat_noisy)
    print("RMSE with noise without regularization: ", rmse_with_noise_no_reg)
    
    #D regularized
    regularizer = N*pow(ro, 2) * np.eye(N, N)
    D_reg = np.transpose(np.linalg.inv(A * A.T + regularizer) * A * np.transpose(X))
    
    X_hat_reg = D_reg * A # No noise in A
    X_hat_noisy_reg = D_reg * A_noisy # Noise in A
    
    Q1_1.plot_decoded_vs_ideal(x_linspace, X_hat_reg, "1.3)", "Regularization without noise")
    Q1_1.plot_decoded_vs_ideal(x_linspace, X_hat_noisy_reg, "1.3)", "Regularization with noise")

    rmse_no_noise_with_reg = get_RMSE_matrix(X, X_hat_reg)
    print("RMSE no noise with regularization: ", rmse_no_noise_with_reg)
    
    rmse_with_noise_with_reg = get_RMSE_matrix(X, X_hat_noisy_reg)
    print("RMSE with noise with regularization: ", rmse_with_noise_with_reg)















