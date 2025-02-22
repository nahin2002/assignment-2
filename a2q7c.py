import numpy as np
import scipy as sp
import sympy as smp
import matplotlib.pyplot as plt
r,theta,z,h=smp.symbols('r theta z h' ,real=True)
rho=r**2
dV=r*smp.diff(r)*smp.diff(theta)*smp.diff(z)
mass_integral=smp.integrate(rho*dV,(r,0,r),(theta,0,2*smp.pi),(z,0,h)).doit().simplify()
print(f'the mass of the cylinder is: {mass_integral}\n')

#let
h=2
r=1
theta=np.linspace(0,2*np.pi,100)
x=r*np.cos(theta)
y=r*np.sin(theta)
Z1=np.linspace(0,h,100)

fig=plt.figure(figsize=(12,8))

ax=fig.add_subplot(111,projection='3d')

for z in Z1:
 ax.plot(x,y,z,color='cyan',alpha=0.5)

plt.xlabel('x')
plt.ylabel('y')
ax.set_zlabel('z') 

plt.show()