import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as smp

x,y,z=smp.symbols('x y z',real=True)
r,theta=smp.symbols('r theta',real=True)
x=r*smp.cos(theta)+1
y=r*smp.sin(theta)
integ=smp.integrate(r,(z,0,3-r**2-2*r*smp.cos(theta)),(r,0,1),(theta,0,2*smp.pi)).evalf()
print(integ)

fig=plt.figure(figsize=(12,6))
ax=fig.add_subplot(111,projection='3d')

theta_vals=np.linspace(0,2*np.pi,100)
r_val=np.linspace(0,1,100)
R,THETA=np.meshgrid(r_val,theta_vals)
X=R*np.cos(THETA)+1
Y=R*np.sin(THETA)
Z=4-X**2-Y**2

ax.plot_surface(X,Y,Z,cmap='cool',alpha=0.7)

th=np.linspace(0,2*np.pi,100)
x2=np.cos(th)+1
y2=np.sin(th)
Z2=np.linspace(0,4,100)
for z in Z2:
    ax.plot(x2,y2,z,color='cyan',alpha=0.4) 

plt.xlabel('X')
plt.ylabel('Y')
ax.set_zlabel('Z')    
plt.show()