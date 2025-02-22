import numpy as np
import sympy as smp
import scipy as sp
import matplotlib.pyplot as plt

x,y,r,theta,z=smp.symbols('x y r theta z',real=True)
F1=x**3
F2=y**3
F3=z**2

div_F=smp.diff(F1,x)+smp.diff(F2,y)+smp.diff(F3,z)
div_F=3*r**2+2*z
print(div_F)
integrand=div_F*r
integrand_2=smp.lambdify([r,theta,z],integrand)
flux=sp.integrate.tplquad(integrand_2,0,2,0,2*np.pi,0,3)[0]
print(f'The outward flux of the vector field across the region is {flux} \n')



fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

th=np.linspace(0,2*np.pi,100)
z_vals=np.linspace(0,2,100)
x_cyl=3*np.cos(th)
y_cyl=3*np.sin(th)
for z1 in z_vals:
 ax.plot(x_cyl,y_cyl,z1,color='red',alpha=1)

x_plane=np.linspace(-3,3,100)
y_plane=np.linspace(-3,3,100)
x_plane,y_plane=np.meshgrid(x_plane,y_plane)
z_top=np.full_like(x_plane,2)
z_bottom=np.full_like(x_plane,0)

ax.plot(x_plane,y_plane,z_top,color='pink',alpha=0.5)
ax.plot(x_plane,y_plane,z_bottom,color='pink',alpha=0.5)

x11,y11,z11=0,0,2
F11=F1.subs([(x,x11),(y,y11),(z,z11)])
F21=F2.subs([(x,x11),(y,y11),(z,z11)])
F31=F3.subs([(x,x11),(y,y11),(z,z11)])

ax.quiver(x11,y11,z11,F11,F21,F31,length=0.1,color='blue')

x22,y22,z22=3,0,1
F12=F1.subs([(x,x22),(y,y22),(z,z22)])
F22=F2.subs([(x,x22),(y,y22),(z,z22)])
F32=F3.subs([(x,x22),(y,y22),(z,z22)])

ax.quiver(x22,y22,z22,F12,F22,F32,length=0.05,color='blue')

x33,y33,z33=-3,0,1
F1_func=smp.lambdify([x,y,z],F1)
F2_func=smp.lambdify([x,y,z],F2)
F3_func=smp.lambdify([x,y,z],F3)

u1=F1_func(x33,y33,z33)
v1=F2_func(x33,y33,z33)
w1=F3_func(x33,y33,z33)

ax.quiver(x33,y33,z33,u1,v1,w1,length=0.05,color='blue')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Region Enclosed by Cylinder and Planes with Vector Field Quivers')
ax.set_zlim(0,2)
plt.show()

plt.show()
