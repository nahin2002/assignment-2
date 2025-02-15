import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

# Define the range for x and y for function (a)
x_a = np.linspace(1, 7, 400)
y_a = np.linspace(1, 7, 400)
X_a, Y_a = np.meshgrid(x_a, y_a)

# Define the function f(x, y) = y^2 - 2y * cos(x)
Z_a = Y_a**2 - 2 * Y_a * np.cos(X_a)

# Define the range for x and y for function (b)
x_b = np.linspace(0, 2 * np.pi, 400)
y_b = np.linspace(0, 2 * np.pi, 400)
X_b, Y_b = np.meshgrid(x_b, y_b)

# Define the function g(x, y) = |sin(x) * sin(y)|
Z_b = np.abs(np.sin(X_b) * np.sin(Y_b))

# Create the 3D plot for function (a)
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X_a, Y_a, Z_a, cmap='viridis')
ax1.set_title('3D plot of f(x, y) = y^2 - 2y * cos(x)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x, y)')

# Create the 3D plot for function (b)
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X_b, Y_b, Z_b, cmap='viridis')
ax2.set_title('3D plot of g(x, y) = |sin(x) * sin(y)|')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('g(x, y)')

plt.tight_layout()
plt.show()