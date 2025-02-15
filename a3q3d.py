import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the symbolic variables x and y
x, y = sp.symbols('x y', real=True)

# Define the temperature function T(x, y)
T = 3 * x**2 * y

# Compute the gradient of T(x, y)
grad_T = sp.Matrix([T.diff(x), T.diff(y)])

# Evaluate the gradient at the point (-1, 3/2)
point = (-1, 3/2)
grad_T_at_point = grad_T.subs({x: point[0], y: point[1]})

# Define the direction vector
direction = sp.Matrix([-1, -1/2])
direction_normalized = direction / direction.norm()

# Compute the directional derivative
directional_derivative = grad_T.dot(direction_normalized)

# Evaluate the directional derivative at the point (-1, 3/2)
directional_derivative_at_point = directional_derivative.subs({x: point[0], y: point[1]}).evalf()

# Print the results
print(f"Gradient of T(x, y) at {point} is {grad_T_at_point}")
print(f"Directional derivative at {point} in the direction {direction} is {directional_derivative_at_point}")

# Plot the directional derivative over a surface
x_vals = np.linspace(-2, 0, 100)
y_vals = np.linspace(0, 2, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = np.zeros_like(X)


# Compute the directional derivative at each point on the grid
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = directional_derivative.subs({x: X[i, j], y: Y[i, j]}).evalf()


# Create a 3D plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')

# Set plot labels and title
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Directional Derivative')
ax.set_title('Directional Derivative of T(x, y)')

plt.show()