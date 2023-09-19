# Import numpy and matplotlib -- you shouldn't need any other libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

from IPython import get_ipython

# https://en.wikipedia.org/wiki/Neural_coding
# https://en.wikipedia.org/wiki/Neural_decoding

"""
noise = np.zeros(A.shape)

for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        ro = 0.2 * A[i, j]
        noise[i, j] = np.random.normal(0, ro)

A_noisy = A + noise
"""



def generate_tuning_curves(x_linspace, num_curves):
    
    Jth = 0
    tuning_curves = []
    alphas = []
    J_biases = []
    
    for i in range(num_curves):
        
        a_max = np.random.uniform(100, 200)
        zeta = round(np.random.uniform(-0.95, 0.95), 2)
        e_value = np.random.choice(np.array([-1, 1]))
        
        alpha = (a_max - Jth) / (1 - zeta)
        J_bias = Jth - zeta * alpha
        
        alphas.append(alpha)
        J_biases.append(J_bias)
        
        tuning_curve = alpha * e_value * x_linspace + J_bias
        tuning_curve[tuning_curve < 0] = 0
        
        tuning_curves.append(tuning_curve)
        
        plt.plot(x_linspace, tuning_curve)

    plt.show()

    return tuning_curves
    
def plot_decoded_vs_ideal(x_linspace, X_hat, decoded_label):
    
    plt.plot(x_linspace, X_hat.T, color = "red", label="Decoded, " + decoded_label)
    plt.plot(x_linspace, x_linspace, color = "black", label="Ideal (Actual)", linestyle="--")
    
    plt.title("Decoded Vs Ideal, " + decoded_label)
    plt.xlabel("Encoded x value")
    plt.ylabel("Stimulus")
    plt.xlim([-1, 1])
    plt.legend()
    plt.show()

    plt.title("Decoded Minus Ideal, " + decoded_label)
    plt.xlabel("Encoded x value")
    plt.ylabel("Difference in Stimulus")
    plt.xlim([-1, 1])
    plt.plot(x_linspace, X_hat.T - np.matrix(x_linspace).T)
    plt.show()
    
def get_RMSE(y_actual, y_predicted):
    rms = sqrt(mean_squared_error(y_actual, y_predicted))
    return rms

get_ipython().magic('clear')
get_ipython().magic('reset -f')

np.random.seed(18945)

if __name__ == "__main__":
    
    
    N = 16
    S = 41
    
    # B)
    
    x_linspace = np.linspace(-1, 1, S)
    
    tuning_curves = generate_tuning_curves(x_linspace, N)
    
    #C)
    
    A = np.matrix(tuning_curves)
    X = np.matrix(x_linspace)
    
    D = np.transpose(np.linalg.inv(A * np.transpose(A)) * A * np.transpose(X))
    
    
    #D)
    print("D)")
    X_hat = D*A
    
    plot_decoded_vs_ideal(x_linspace, X_hat, "No Noise")
    
    rmse = get_RMSE(np.asarray(X_hat)[0], x_linspace)
    print(rmse)
    print()
    
    #E)
    print("E)")
    ro = 0.2*np.amax(A)
    A_noisy = A + np.random.normal(0, ro, A.shape)
    
    
    X_hat_noisy = D * A_noisy
    
    plot_decoded_vs_ideal(x_linspace, X_hat_noisy, "Noise in A, No Regularization")
    rmse = get_RMSE(np.asarray(X_hat_noisy)[0], x_linspace)
    print(rmse)
    print()
    
    #F)
    print("F)")
    
    regularizer = S*pow(ro, 2) * np.eye(N, N)
    D_reg = np.transpose(np.linalg.inv(A * A.T + regularizer) * A * np.transpose(X))
    
    X_hat_reg = D_reg * A
    X_hat_noisy_reg = D_reg * A_noisy
    
    plot_decoded_vs_ideal(x_linspace, X_hat_reg, "No noise in A, Decoder Regularized")
    rmse = get_RMSE(np.asarray(X_hat_reg)[0], x_linspace)
    print(rmse)
    
    plot_decoded_vs_ideal(x_linspace, X_hat_noisy_reg, "Noise in A, Decoder Regularized")
    rmse = get_RMSE(np.asarray(X_hat_noisy_reg)[0], x_linspace)
    print(rmse)
    
    print()
    
    
    #G)
    print("G)")
    
    print()










