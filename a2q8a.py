import numpy as np
import scipy as sp
import sympy as smp
import matplotlib.pyplot as plt
x,y,theta,r=smp.symbols('x y theta r',real=True)
F1=smp.exp(x)-y**3
F2=smp.cos(y)+x**3
r_lower=0
r_upper=1
theta_lower=0
theta_upper=2*smp.pi
integrand=smp.diff(F2,x)-smp.diff(F1,y)
integrand_2=3*r**2
integrand_2=integrand_2*r*smp.diff(r)*smp.diff(theta)
integral=smp.integrate(integrand_2,(r,r_lower,r_upper),(theta,theta_lower,theta_upper)).doit().simplify()
print(f'the work done is: {integral}\n')

Theta=np.linspace(0,2*np.pi,100)
X=np.cos(Theta)
Y=np.sin(Theta)
plt.plot(X,Y,color='red')

x1=np.linspace(-1,1,20)
x2=np.linspace(-1,1,20)
X1,Y1=np.meshgrid(x1,x2)
U=np.exp(X1)-Y1**3
V=np.cos(Y1)+X1**3
plt.quiver(X1,Y1,U,V,alpha=0.5)

plt.xlabel('X')
plt.ylabel('Y')
plt.grid()
plt.show()
