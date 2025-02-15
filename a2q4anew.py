import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 400)
y = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x, y)

Z1 = 4 * X**2 + Y**2

Z_values = [1, 4, 9, 16, 25, 36]

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
contour1 = plt.contour(X, Y, Z1, levels=Z_values, cmap='viridis')
plt.clabel(contour1, inline=True, fontsize=8)
plt.title('Contour plot of f(x, y) = 4x^2 + y^2')
plt.xlabel('x')
plt.ylabel('y')

x = np.linspace(-36, 36, 400)
y = np.linspace(-36, 36, 400)
X, Y = np.meshgrid(x, y)

plt.subplot(1, 2, 2)
for z in Z_values:
    Z2 = z**2 - X**2 - Y**2
    contour2 = plt.contour(X, Y, Z2, levels=[0], cmap='viridis')
    plt.clabel(contour2, inline=True, fontsize=8, fmt={0: f'{z}'})
plt.title('Contour plot of f(x, y, z) = z^2 - x^2 - y^2')
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.show()