# Import numpy and matplotlib -- you shouldn't need any other libraries
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, pi, exp
from IPython import get_ipython
from scipy.optimize import curve_fit



Trc = 20 / 1000 # Given in ms, divide by 1000 to get s
Tref = 2 / 1000  # Given in ms, divide by 1000 to get s

def get_alpha(a_max):
    exponent = (Tref - 1/a_max) / Trc
    alpha = pow(1 - exp(exponent), -1) - 1
    
    return alpha

def get_LIF_3D(x, y, alpha, preferred_vector, J_bias):
    point = np.transpose([x, y])
    
    J = alpha * np.dot(point, preferred_vector) + J_bias
    
    if J <= 1:
        a = 0
    else:
        a = pow(Tref - Trc * np.log(1 - 1/J), -1)[0]
        
    return a
    
    

def generate_2D_tuning_curve(x_linspace, y_linspace):
    A = np.zeros((len(y_linspace), len(x_linspace)))
    
    preferred_angle_rads = -pi/4
    vector_mag = 1

    preferred_vector = np.array([
        [vector_mag * cos(preferred_angle_rads)], 
        [vector_mag * sin(preferred_angle_rads)]
    ])

    a_max = 100
    
    alpha = get_alpha(a_max)
    J_bias = 1
    
    x_idx = 0
    for x in x_linspace:
        y_idx = 0
        
        for y in y_linspace:
            
            a = get_LIF_3D(x, y, alpha, preferred_vector, J_bias)
            
            A[y_idx, x_idx] = a
            
            y_idx += 1
            
        x_idx += 1
        
    return A

def generate_2D_tuning_curve_unit_circle(x_linspace, y_linspace):
    A = []
    
    preferred_angle_rads = -pi/4
    vector_mag = 1

    preferred_vector = np.array([
        [vector_mag * cos(preferred_angle_rads)], 
        [vector_mag * sin(preferred_angle_rads)]
    ])

    a_max = 100
    
    alpha = get_alpha(a_max)
    J_bias = 1
    
    for x, y in zip(x_linspace, y_linspace):            
        a = get_LIF_3D(x, y, alpha, preferred_vector, J_bias)
        
        A.append(a)
        
    return A
            
            

def plot_3d(x_space, y_space, z_space, title):
    
    X, Y = np.meshgrid(x_space, y_space)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, z_space, cmap='viridis')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title(title)
    plt.show()
    
def cosFit(x, a, b, c, d):
    return (a * (np.cos(b * x +c))) + d
    
if __name__ == "__main__":
    
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
    
    np.random.seed(18945)
    
    # A) [DONE]
    print("A)")
    
    # Create a meshgrid of X and Y values
    x_linspace = np.linspace(-1, 1, 41)
    y_linspace = np.linspace(-1, 1, 41)
    points_2D = np.zeros((2, 41**2))
    
    idx = 0
    for x in x_linspace:
        for y in y_linspace:
            points_2D[0][idx] = x
            points_2D[1][idx] = y
            idx += 1
    
    A = generate_2D_tuning_curve(x_linspace, y_linspace)
    
    
    plot_3d(x_linspace, y_linspace, A, "3D tuning curve with zeta = (0, 0), a_max = 100Hz, \nTau_rc = 20, Tau_ref = 2")
    
    
    # B)
    print("B)")
    
    angles = np.linspace(0, 2*np.pi, 10000)
    
    x = np.cos(angles)
    y = np.sin(angles)
    
    tuning_curve_2D = generate_2D_tuning_curve_unit_circle(x, y)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(x, y, tuning_curve_2D, label="LIF neuron at unit circle")
    
    
    def func(thetas, a, b, c, d):

        return a + np.cos(b * thetas + c) + d

    popt, pcov = curve_fit(func, angles, tuning_curve_2D)
    
    ax.scatter(x, y, func(angles, *popt), label="fit")
    ax.legend()
    fig.show()
    
#    plt.plot(angles, func(angles, *popt), 'g--',

 #        label='Curve fit')
    
    
    #C)
    print("C)")







