# Import numpy and matplotlib -- you shouldn't need any other libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

from IPython import get_ipython

# https://en.wikipedia.org/wiki/Neural_coding
# https://en.wikipedia.org/wiki/Neural_decoding



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
        J_bias = zeta * a_max / (zeta - 1)
                
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
    plt.ylim([-1, 1])
    plt.legend()
    plt.grid()
    plt.show()

    plt.grid()
    plt.title(question_label + " Decoded Minus Ideal, " + decoded_label)
    plt.xlabel("Stimulus")
    plt.ylabel("Decoded Stimulus - Actual Stimulus")
    plt.xlim([-1, 1])
    plt.ylim([np.amin(X_hat), np.amax(X_hat)])
    plt.ylim([-1, 1])
    plt.plot(x_linspace, X_hat.T - np.matrix(x_linspace).T)
    plt.show()
    
def get_RMSE(y_actual, y_predicted):
    rms = round(sqrt(mean_squared_error(y_actual, y_predicted)), 2)
    return rms

get_ipython().magic('clear')
get_ipython().magic('reset -f')

np.random.seed(9)

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
    
    
    ro = 0.2*np.max(A)
        
    noise = np.zeros(A.shape)

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            #ro = 0.2 * A[i, j]
            noise[i, j] = np.random.normal(0, ro)

    A_noisy = np.copy(A)
    A_noisy += noise
    
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
    
    G = """
                                Col 1: No noise in curves   Col 2: Noise in Curves
    Row 1: No Regularization               0.00                        0.45
    Row 2: Regularization                  0.05                        0.18
    
    There is almost no error when there is no noise. This means that our
    decoder can accurately represent the stimulus with the number of neurons given.
    
    However, when noise is added, the RMSE increases significantly. This is to be
    expected, as the decoder is trying to decode curves that it was not made to decode,
    curves that are not the same shape as the curves used to make the decoder.
    
    Since neuron activity is in the range of 0 to 200, and the added noise is normal around 0 with stdev 
    of 40, any neural activity that recieves an increase in magnitude due to noise
    can become much larger, and since the decoder is just linearly scaling each
    neural activity at each point in time, any random spike in the neural activity
    any any point in time comes through in the summation of all neural activity
    for any point in time. Therefore, (as expected) there is a much greater RMSE when noise is added.   
    This lines up with visual inspection, which shows a decoded output that has 
    significant residuals compared to the expected output.
    
    Also as expected, adding regularization dereases the RMSE of the noisy curve.
    By adding regularization, each neuron gets an equal boost in consideration
    from the decoder, such that any neurons that would have been given a huge weight
    to accomodate for some specific noise blip are effectively "watered down"
    by the regularizer. In particular, the regularizer is done with an identity
    matrix because A * A.T will have each neurons activity multiplied by itself
    along the diagonal, and each neuron's activity multiplied by itself will be greater than
    a neuron's actvity multiplied by some other nueron's activity. This helps smooth out the fit, and since noise was 
    added normally with a mean of 0, the fit is smoothed back toward the 
    actual stimulus.

    The additional error observed when there is no noise but there is regularization
    is small. It is likely due to the regularizer beating out A * A.T,
    over smoothing the curve, and not allowing the necessary differences between
    decoder values to take place. Indeed, if the regularizer is scaled up by 
    1000, any regularizer curves nearly become y = 0, as neuron activity
    is dwarfed by the equalizing activity of the regularizer.
    
    """
    
    print(G)
    
    
    print()










