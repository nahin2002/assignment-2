import sympy as smp
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

x,y,phi=smp.symbols('x y phi',real=True)
F1,F2,F=smp.symbols('F1 F2 F',cls=smp.Function,real=True)
F1=F1(x,y)
F2=F2(x,y)
F=smp.Matrix([F1,F2])
F1=smp.exp(y)
F2=x*smp.exp(y)
df1=smp.diff(F1,y)
df2=smp.diff(F2,x)

if df1==df2:
    print('the force field F is conservative \n')

PHI_x=F1
PHI_y=F2 

phi=smp.integrate(PHI_x,x)
C_y=phi+smp.Symbol('C')
C=smp.diff(C_y,y)-PHI_y
phi=phi+C
print(f'the potential function is: {phi.simplify()}\n')

phi_A=phi.subs([(x,1),(y,0)])
phi_B=phi.subs([(x,-1),(y,0)])
Work_done=phi_B-phi_A
print(f'work done: {Work_done}\n')

fig=plt.figure(figsize=(10,8))

x=np.linspace(-2,2,20)
y=np.linspace(-2,2,20)
X,Y=np.meshgrid(x,y)
U=np.exp(Y)
V=X*np.exp(Y)
plt.quiver(X,Y,U,V,color='blue',alpha=0.5)

theta=np.linspace(0,np.pi,100)
x1=np.cos(theta)
x2=np.sin(theta)
plt.plot(x1,x2,color='green')
plt.scatter(1,0,color='red',label='Start(1,0)')
plt.scatter(-1,0,color='purple',label='End(-1,0)')
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Force Field and Semicircular Path')
plt.legend()
plt.grid()
plt.show()