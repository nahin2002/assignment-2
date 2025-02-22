import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import sympy as smp

f=lambda z,y,x:x*np.exp(-y)*np.cos(z)
anti_derivative_1=sp.integrate.tplquad(f,0,1, lambda x:0,lambda x:1-x**2,lambda x,y:3,lambda x,y:4-x**2-y**2)[0]
print(f'the values of integral 1 is: {anti_derivative_1}\n')

g=lambda y,x:(x*y)/np.sqrt(x**2+y**2+1)
anti_derivative_2=sp.integrate.dblquad(g,0,1,0,1)[0]
print(f'the values of integral 2 is: {anti_derivative_2}\n')

#b
x,y,z=smp.symbols('x y z',real=True)
z=smp.sqrt(4-x**2)
dzdx=z.diff(x)
dzdy=z.diff(y)
dzdz=1
integrand=smp.sqrt(dzdx**2+dzdy**2+dzdz**2)
integral_1=smp.integrate(integrand,(x,0,1),(y,0,4))
#print(integral_1)
#print(integrand)
integrand_1=smp.lambdify([y,x],integrand)
integral_2=sp.integrate.dblquad(integrand_1,0,1,0,4)[0] 
#y defined first so integral of y is calculated first and then x
#like integral{0-1}integral{0-4}( )dydx
print(f'the surface area is: {integral_2}')

