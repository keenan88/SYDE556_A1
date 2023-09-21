# Import numpy and matplotlib -- you shouldn't need any other libraries
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, pi
from IPython import get_ipython

def generate_2D_tuning_curve(x_linspace, y_linspace):
    A = np.zeros((len(y_linspace), len(x_linspace)))
    
    preferred_angle_rads = -pi/4
    vector_mag = 1
    preferred_vector = np.array([
        [vector_mag * cos(preferred_angle_rads)], 
        [vector_mag * sin(preferred_angle_rads)]
    ])
    
    Tref = 2
    Trc = 20

    alpha = 100
    zeta = 0
    
    x_idx = 0
    for x in x_linspace:
        y_idx = 0
        
        for y in y_linspace:
            point = np.transpose([x, y])
            
            print(point)
            J = alpha * np.dot(point, preferred_vector) + zeta
            
            # a = G[J]
            if J <= 1:
                a = 0
            else:
                a = 1 / (Tref + Trc * (1 - 1/J))
                
            A[y_idx, x_idx] = a
            
            y_idx += 1
            
        x_idx += 1
        
    return A
            
            

def plot_3d(x_space, y_space, z_space):
    
    X, Y = np.meshgrid(x_space, y_space)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, z_space, cmap='viridis')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('3D Surface Plot')
    plt.show()
    
    
get_ipython().magic('clear')
get_ipython().magic('reset -f')

# A)
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

plot_3d(x_linspace, y_linspace, A)


# B)
print("B)")

angles = np.linspace(0, 2*np.pi, 10)

x = np.cos(angles)
y = np.sin(angles)

tuning_curve_2D = generate_2D_tuning_curve(x, y)

plot_3d(x, y, tuning_curve_2D)













