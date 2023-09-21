# Import numpy and matplotlib -- you shouldn't need any other libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import cos, sin, pi, sqrt, exp
from IPython import get_ipython


    
    
get_ipython().magic('clear')
get_ipython().magic('reset -f')

np.random.seed(18945)

num_encoders = 100
dimension = 2

# A)
print("A)")
encoders = np.zeros((dimension, num_encoders))

for encoder_idx in range(num_encoders):
    angle = round(np.random.uniform(0, 2*pi), 2)
    
    encoders[0][encoder_idx] = cos(angle)
    encoders[1][encoder_idx] = sin(angle)

for col in range(encoders.shape[1]):
    plt.plot([0, encoders[0][col]], [0, encoders[1][col]])
    print(col)


plt.show()


# B)
print("B)")

    
x_linspace = np.linspace(-1, 1, 41)
y_linspace = np.linspace(-1, 1, 41)
points_2D = np.zeros((2, 41**2))

idx = 0
for x in x_linspace:
    for y in y_linspace:
        points_2D[0][idx] = x
        points_2D[1][idx] = y
        idx += 1

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


















