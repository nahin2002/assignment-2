import sympy as smp
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sympy.physics.vector import ReferenceFrame
from sympy.physics.vector import curl
from mpl_toolkits.mplot3d import Axes3D

x,y,z,r,t=smp.symbols('x y z r t',real=True)
R=ReferenceFrame('R')
F=2*R[2]*R.x+3*R[0]*R.y+5*R[1]*R.z
C=curl(F,R)
print(f'curl of F is: {C}\n')

curl_F=smp.Matrix([5,2,3])

F_k=smp.Matrix([0, 6*smp.cos(t), 10*smp.sin(t)])
rr=smp.Matrix([2*smp.cos(t),2*smp.sin(t),0])
dr=rr.diff(t)
integrand_1=F_k[0]*rr.diff(t)[0]+F_k[1]*rr.diff(t)[1]+F_k[2]*rr.diff(t)[2]
line_integral=smp.integrate(integrand_1,(t,0,2*smp.pi)).doit().simplify()
print(f'The line integral is: {line_integral}\n')

z=4-x**2-y**2
normal_vec=smp.Matrix([-z.diff(x),-z.diff(y),1])
F_dot_n=curl_F.dot(normal_vec)
integrand_2=r*(F_dot_n.subs([(x,r*smp.cos(t)),(y,r*smp.sin(t))]))
surface_integral=smp.integrate(integrand_2,(r,0,2),(t,0,2*smp.pi)).doit().simplify()
print(f'The surface integral is: {surface_integral}\n')

if line_integral-surface_integral==0:
    print('Since the line and surface integrals are equal to each other,\n Stokes Theorem is verified.')
else:
    print('Stokes theorem is not verified')

fig=plt.figure(figsize=(12,8))
ax=fig.add_subplot(111,projection='3d')
x=np.linspace(-2,2,100)
y=np.linspace(-2,2,100)
X,Y=np.meshgrid(x,y)
Z=4-X**2-Y**2
Z[Z<0]=np.nan
ax.plot_surface(X,Y,Z,cmap='viridis',alpha=0.5)


theta = np.linspace(0, 2*np.pi, 100)
x_circle = 2 * np.cos(theta)
y_circle = 2 * np.sin(theta)
z_circle = np.zeros_like(theta)
ax.plot(x_circle, y_circle, z_circle, color='blue')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Paraboloid and Boundary Circle')

plt.show()