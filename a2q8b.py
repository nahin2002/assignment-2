import numpy as np
import scipy as sp
import sympy as smp
import matplotlib.pyplot as plt

theta,phi=smp.symbols('theta phi',real=True)
x,y,z,G=smp.symbols('x y z G',cls=smp.Function,real=True)
x=x(phi,theta)
y=y(phi,theta)
z=z(phi,theta)
x=smp.sin(phi)*smp.cos(theta)
integrand=(x**2)*smp.sin(phi)

integrand_2=smp.lambdify([theta,phi],integrand)

surface_integral=sp.integrate.dblquad(integrand_2,0,np.pi,lambda theta:0,lambda theta:2*np.pi)[0]
print(surface_integral)

fig=plt.figure(figsize=(12,8))
ax=fig.add_subplot(111,projection='3d')

th=np.linspace(0,2*np.pi,100)
ph=np.linspace(0,np.pi,100)
x_sphere=np.outer(np.cos(th),np.sin(ph))
y_sphere=np.outer(np.sin(th),np.sin(ph))
z_sphere=np.outer(np.ones(np.size(th)),np.cos(ph))

x_squared = x_sphere**2

ax.plot_surface(x_sphere, y_sphere, z_sphere, facecolors=plt.cm.plasma(x_squared), rstride=1, cstride=1, alpha=0.8)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Unit Sphere with $x^2$ Color Map')

mappable = plt.cm.ScalarMappable(cmap='plasma')
mappable.set_array(x_squared)
plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=5, label='$x^2$')

plt.show()