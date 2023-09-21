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

def G_rlu(alpha, x, encoder, J_bias):
    J = alpha * (x * encoder) + J_bias
    G = J if J >= 0 else 0
    return G
    

def generate_tuning_curves(x_linspace, num_curves):
    
    tuning_curves = []
    
    for i in range(num_curves):
        
        a_max = np.random.uniform(100, 200)
        zeta = round(np.random.uniform(-0.95, 0.95), 2)
        encoder = np.random.choice(np.array([-1, 1]))
        
        alpha = a_max / (1 - zeta)
        J_bias = a_max * zeta / (zeta - 1)
                
        # This is a = G[J] = Max(J, 0), where J = alpha + J_bias
        
        tuning_curve = []
        for x in x_linspace:
            a_value = G_rlu(alpha, x, encoder, J_bias)
            tuning_curve.append(a_value)
        
        tuning_curves.append(tuning_curve)

    return tuning_curves
    



def plot_decoded_vs_ideal(x_linspace, X_hat, question_label, decoded_label):
    
    plt.plot(x_linspace, X_hat.T, color = "red", label="Decoded Response")
    plt.plot(x_linspace, x_linspace, color = "black", label="Real Response", linestyle="--")
    
    plt.title(question_label + " Decoded Vs Ideal, " + decoded_label)
    plt.xlabel("Stimulus")
    plt.ylabel("Response")
    plt.xlim([-1, 1])
    plt.ylim([np.amin(X_hat), np.amax(X_hat)])
    plt.legend()
    plt.grid()
    plt.show()

    plt.grid()
    plt.title(question_label + " Decoded Minus Ideal, " + decoded_label)
    plt.xlabel("Stimulus")
    plt.ylabel("Decoded Stimulus - Actual Stimulus")
    plt.xlim([-1, 1])
    plt.ylim([np.amin(X_hat), np.amax(X_hat)])
    plt.plot(x_linspace, X_hat.T - np.matrix(x_linspace).T)
    plt.show()
    
def get_RMSE(y_actual, y_predicted):
    rms = round(sqrt(mean_squared_error(y_actual, y_predicted)), 2)
    return rms

get_ipython().magic('clear')
get_ipython().magic('reset -f')

np.random.seed(18945)

if __name__ == "__main__":
    
    
    N = 16
    S = 41
    
    # B) [DONE]
    print("B) ")
    
    x_linspace = np.linspace(-1, 1, S)
    
    tuning_curves = generate_tuning_curves(x_linspace, N)
    
    for tuning_curve in tuning_curves:
        plt.plot(x_linspace, tuning_curve)

    plt.title("1.1B) 16 RLU Tuning Curves")
    plt.xlabel("Stimulus")
    plt.ylabel("Response")
    plt.xlim([-1, 1])
    plt.show() # There is a slight elbow at the x axis when using delta_x = 0.05 on the plot. This is an artifact of the plot, not a feature of G[J] = max(0, J)
    print()
    
    #C) [DONE]
    print("C) ")
    
    A = np.matrix(tuning_curves)
    X = np.matrix(x_linspace)
    
    D = np.transpose(np.linalg.inv(A * np.transpose(A)) * A * np.transpose(X))
    print("Optimal Decoders:")
    print(D)
    print()
    
    #D) [DONE]
    print("D)")
    X_fit = D*A
    
    plot_decoded_vs_ideal(x_linspace, X_fit, "1D)", "No Noise")
    
    rmse_no_noise_no_reg = get_RMSE(np.asarray(X_fit)[0], x_linspace)
    print("RMSE no noise no regularization: ", rmse_no_noise_no_reg)
    print()
    
    
    
    #E) [DONE]
    print("E)")
    ro = 0.2*np.amax(A)
    A_noisy = A + np.random.normal(0, ro, A.shape)
    
    
    X_hat_noisy = D * A_noisy
    
    plot_decoded_vs_ideal(x_linspace, X_hat_noisy, "1E)","Noise in A, No Regularization")
    rmse_noisy_curves_no_reg = get_RMSE(np.asarray(X_hat_noisy)[0], x_linspace)
    print("RMSE Noisy Tuning Curves No Regularization: ", rmse_noisy_curves_no_reg)
    print()
    
    #F) [DONE]
    print("F)")
    
    regularizer = N*pow(ro, 2) * np.eye(N, N)
    D_reg = np.transpose(np.linalg.inv(A * A.T + regularizer) * A * np.transpose(X))
    
    X_hat_reg = D_reg * A
    X_hat_noisy_reg = D_reg * A_noisy
    
    plot_decoded_vs_ideal(x_linspace, X_hat_reg, "1F)", "No noise in A, Decoder Regularized")
    rmse_reg_non_noisy_curves_with_reg = get_RMSE(np.asarray(X_hat_reg)[0], x_linspace)
    print("RMSE Regularization on curves without noise: ", rmse_reg_non_noisy_curves_with_reg)
    
    plot_decoded_vs_ideal(x_linspace, X_hat_noisy_reg, "1F)", "Noise in A, Decoder Regularized")
    rmse_noisy_curves_with_reg = get_RMSE(np.asarray(X_hat_noisy_reg)[0], x_linspace)
    print("RMSE Regularization on noisy curves: ", rmse_noisy_curves_with_reg)
    
    print()
    
    
    #G) [DONE]
    print("G)")
    
    RMSE_matrix = [[rmse_no_noise_no_reg, rmse_noisy_curves_no_reg],
                   [rmse_reg_non_noisy_curves_with_reg, rmse_noisy_curves_with_reg]
                   ]
    print(RMSE_matrix)
    
    """
                                Col 1: No noise in curves   Col 2: Noise in Curves
    Row 1: No Regularization               0.19                       3519.63
    Row 2: Regularization                  0.02                        0.15
    
    The most immediate and obvious standout is that the RMSE when there is noise
    and no regularization is gargantuan compared to the other RMSEs.
    
    With regularization, the RMSE does of course get worse when noise is added,
    though not horribly so.
    
    With noise, the RMSE decreases drastically when regulariation is added.
    
    Without noise, regularization still helps decrease RMSE, though perhaps unintentionally,
    As error without noise is due to distortion.
    
    Noise can be added to A to create A_noisy = A + E
    
    Decoding A_noisy with a decoder calcuated with A is nonsensical, and yields huge RMSEs.
    
    Since A_noisy = A + E, the inversion calculation to calculated D naturally 
    yields a regularizer, N*ro^2*I, yielding D_noisy. Since D_noisy is calculated
    using the least squares calculation against A_noisy, it yields the best fit
    and a much lower RMSE, althought the system is still overdefined and has noticeable error.
    
    """
    
    
    print()










