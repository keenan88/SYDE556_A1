# Import numpy and matplotlib -- you shouldn't need any other libraries
import numpy as np
import matplotlib.pyplot as plt
from math import exp
from IPython import get_ipython
import Q1


get_ipython().magic('clear')
get_ipython().magic('reset -f')

def generate_LIF_tuning_curves(x_linspace, num_curves):
    
    tuning_curves = []
    Tref_ms = 2
    Trc_ms = 20
    
    for i in range(num_curves):
        
        a_max = np.random.uniform(100, 200)
        zeta = round(np.random.uniform(-0.95, 0.95), 2)
        e_value = np.random.choice(np.array([-1, 1]))
        
        exponent = -1 / Trc_ms * (Tref_ms - 1/a_max)
        alpha = 1 / (1 - zeta) * (1 / (1 - exp(exponent)) - 1)
        J_bias = 1 - alpha * zeta
        
        J = alpha * x_linspace + J_bias
        tuning_curve = []
        
        for element in J:
            if element > 1:
                G = 1 / (Tref_ms - Trc_ms * np.log(1 - 1/element))
                tuning_curve.append(G)
            else:
                tuning_curve.append(0)
                
        tuning_curve = np.array(tuning_curve)
        tuning_curve *= e_value
        
        tuning_curves.append(tuning_curve)
        
        plt.plot(x_linspace, tuning_curve)

    plt.show()

    return tuning_curves


np.random.seed(18945)

N = 16
S = 41
x_linspace = np.linspace(-1, 1, S)

tuning_curves = generate_LIF_tuning_curves(x_linspace, N)
A = np.matrix(tuning_curves)
X = np.matrix(x_linspace)


# No Noise in A, D not regularized
D = np.transpose(np.linalg.inv(A * np.transpose(A)) * A * np.transpose(X))
X_hat = D*A
Q1.plot_decoded_vs_ideal(x_linspace, X_hat, "No Noise No Regularization")

# Noise in A, D not regularized
ro = 0.2*np.amax(A)
A_noisy = A + np.random.normal(0, ro, A.shape)
X_hat_noisy = D * A_noisy
Q1.plot_decoded_vs_ideal(x_linspace, X_hat_noisy, "Noise Without Regularization")

#D regularized
regularizer = S*pow(ro, 2) * np.eye(N, N)
D_reg = np.transpose(np.linalg.inv(A * A.T + regularizer) * A * np.transpose(X))

X_hat_reg = D_reg * A # No noise in A
X_hat_noisy_reg = D_reg * A_noisy # Noise in A

Q1.plot_decoded_vs_ideal(x_linspace, X_hat_reg, "Regularization without noise")
Q1.plot_decoded_vs_ideal(x_linspace, X_hat_noisy_reg, "Regularization with noise")
















