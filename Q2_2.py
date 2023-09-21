# Import numpy and matplotlib -- you shouldn't need any other libraries
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, pi, sqrt
from IPython import get_ipython
import Q2_1

def plot_unit_vectors(vectors, title, decoders = -1):
    fig, ax = plt.subplots()
    for col in range(vectors.shape[1]):
        ax.plot([0, vectors[0][col]], [0, vectors[1][col]])    
    
    if decoders == -1:
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
    else:
        ax.set_xlim([-1e-3, 1e-3])
        ax.set_ylim([-1e-3, 1e-3])
    
    ax.set_xlabel("x1")
    ax.set_xlabel("x2")
    ax.set_title(title)
    ax.grid()
    fig.show()

get_ipython().magic('clear')
get_ipython().magic('reset -f')

np.random.seed(18945)

N = 10
num_encoders = N
dimension = 2


# A)
print("A)")

encoders = np.zeros((dimension, num_encoders))

for encoder_idx in range(num_encoders):
    angle = round(np.random.uniform(0, 2*pi), 2)
    
    encoders[0][encoder_idx] = cos(angle)
    encoders[1][encoder_idx] = sin(angle)

plot_unit_vectors(encoders, "Encoder Unit Vectors")



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
alphas = []

Tref_ms = 2
Trc_ms = 20

for i in range(num_encoders):
    
    a_max = np.random.uniform(100, 200)
    zeta = round(np.random.uniform(-0.95, 0.95), 2)
    encoder = [[encoders[0][i]], [encoders[1][i]]]
    print(i)
    
    alpha = Q2_1.get_alpha(a_max)
    alphas.append(alpha)
    J_bias = 1
    
    A = np.zeros((len(y_linspace), len(x_linspace)))
    tuning_curve = []
    xy_as_col = []
    
    for x_idx in range(41):
        for y_idx in range(41):
            x = x_linspace[x_idx]
            y = y_linspace[y_idx]
    
            a = Q2_1.get_LIF_3D(x, y, alpha, encoder, J_bias)
            A[y_idx, x_idx] = a
            
            tuning_curve.append(a)
            xy_as_col.append([x, y])
            
    tuning_curves.append(tuning_curve)
            
#    Q2_1.plot_3d(x_linspace, y_linspace, A)

xy_as_col= np.matrix(xy_as_col).T
tuning_curves = np.matrix(tuning_curves)


A = tuning_curves
ro = 0.2 * np.amax(A)

regularizer = N*pow(ro, 2) * np.eye(N, N)
D_reg = np.transpose(np.linalg.inv(A * A.T + regularizer) * A * np.transpose(xy_as_col))

plot_unit_vectors(np.array(D_reg), "Decoders", 1)



#C)
print("C)")

"""
The decoder vectors are MUCH smaller than the encoder vectors, on the scale to 10^-3 
times smaller.

"""

#D)
print("D)")

num_test_pts = 20
directions = np.random.uniform(0, 2 * np.pi, num_test_pts)
magnitudes = np.random.uniform(0, 1, num_test_pts)

test_x = []
test_y = []
test_points_as_col = []

for direction, mag in zip(directions, magnitudes):
    x = cos(direction) * mag
    y = sin(direction) * mag
    test_x.append(x)
    test_y.append(y)
    test_points_as_col.append([x, y])

test_points_as_col = np.matrix(test_points_as_col).T


# Need to determine A matrix

A = np.zeros((num_encoders, num_test_pts))
for i in range(num_encoders):
    
    encoder = [[encoders[0][i]], [encoders[1][i]]]
    
    alpha = alphas[i]
    J_bias = 1
    
    sample_idx = 0
    for x, y in zip(test_x, test_y):
        a = Q2_1.get_LIF_3D(x, y, alpha, encoder, J_bias)
        A[i, sample_idx] = a
        sample_idx += 1
            
fit = D_reg * A
vectors = np.array(fit)

fig, ax = plt.subplots()

ax.scatter(vectors[0], vectors[1], color="red", label="Fit")
ax.scatter(test_x, test_y, color="black", label="Original Data")

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_xlabel("x1")
ax.set_xlabel("x2")
ax.set_title("Best Fit vs Actual Points")
plt.legend()
ax.grid()
fig.show()

rmse = np.sum(np.square(test_points_as_col - fit)) / (N * dimension)

print("RMSE: ", rmse)
print()


#E)
print("E)")

fit = np.matrix(encoders) * np.matrix(A)

vectors = np.array(fit)

for i in range(len(vectors[0])):
    x = vectors[0][i]
    y = vectors[1][i]
    magnitude = sqrt(x**2 + y**2)
    vectors[0][i] /= magnitude
    vectors[1][i] /= magnitude
    
for i in range(len(test_x)):
    x = test_x[i]
    y = test_y[i]
    
    magnitude = sqrt(x**2 + y**2)
    test_x[i] /= magnitude
    test_y[i] /= magnitude


test_points_as_col = np.matrix([test_x, test_y])

rmse = np.sum(np.square(test_points_as_col - vectors)) / (N * dimension)    
print("Angular RMSE for data decoded with encoders: ", rmse)

fig, ax = plt.subplots()

ax.scatter(vectors[0], vectors[1], color="red", label="Fit")
ax.scatter(test_x, test_y, color="black", label="Original Data")

ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("Best Fit vs Actual Points, Using decoders as encoders \n (Fit Scaled to unit circle)")
plt.legend()
ax.grid()
fig.show()








