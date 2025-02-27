import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 400)
y = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x, y)

f1 = 4 * X**2 + Y**2

z_levels=[1,4,9,16,25,36]
fig=plt.figure(figsize=(12,6))
ax=fig.add_subplot(121)
contour1 = plt.contour(X, Y, f1, levels=z_levels, cmap='viridis')
plt.clabel(contour1, inline=True, fontsize=8)
plt.title('Contour plot of f(x, y) = 4x^2 + y^2')
plt.xlabel('x')
plt.ylabel('y')

def level_surface(x,y,k):
    return np.sqrt(x**2+y**2+k)


alpha_values=np.linspace(0.3,0.8,len(z_levels))
ax=fig.add_subplot(122,projection='3d')

for k,alpha in zip(z_levels,alpha_values):
    Z=level_surface(X,Y,k)
    ax.plot_surface(X,Y,Z,alpha=alpha,cmap='coolwarm',edgecolor='none')
    ax.plot_surface(X,Y,-Z,alpha=alpha,cmap='coolwarm',edgecolor='none')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Level surfaces of f(x,y,z)=z^2-x^2-y^2')

plt.tight_layout()
plt.show()