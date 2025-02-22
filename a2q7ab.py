import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
#a
f=lambda y,x:10-8*x**2-2*y**2
x_lower = 0
x_upper = 1
y_lower = 0
y_upper = 2
total_temperature=sp.integrate.dblquad(f,x_lower,x_upper,y_lower,y_upper)[0]
print(f'The total temperature is: {total_temperature}\n')
area=(x_upper-x_lower)*(y_upper-y_lower)
avg_temp=total_temperature/area
print(f'the average temperature of the rectangular portion is: {avg_temp} degrees Celsius')

#b
import sympy as smp
t=smp.symbols('t',real=True)
x,y,z,r,f=smp.symbols('x y z r f',cls=smp.Function,real=True)
x=x(t)
y=y(t)
z=z(t)
r=smp.Matrix([x,y,z])
f=f(x,y,z)
integrand=f*smp.diff(r,t).norm()
integrand=integrand.subs([(f,x*y+z**3),(x,smp.cos(t)),(y,smp.sin(t)),(z,t)]).doit().simplify()
integrand_2=smp.lambdify([t],integrand)
line_integral=sp.integrate.quad(integrand_2,0,np.pi)[0]
print(f'the values of the line integralis:{line_integral}\n')

th=np.linspace(0,np.pi,100)
x1=np.cos(th)
y1=np.sin(th)
z1=th

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

pl1=ax.plot(x1,y1,z1)

ax.scatter(1,0,0,label='(1,0,0)')
ax.scatter(-1,0,np.pi,label='(-1,0,3.1416)')
#ax.quiver(1,0,0,-1,0,np.pi, arrow_length_ratio=0.3)

plt.xlabel('x')
plt.ylabel('y')
ax.set_zlabel('z')
ax.set_title('Helix C and Line Integral')
ax.legend()

plt.show()
