import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sympy import *
from sympy.vector import CoordSys3D

x,y,z,t=symbols('x y z t',real=True)
f=x**2+4*y**2+z**2
p=(1,2,1)

#a
fx=diff(f,x)
fy=diff(f,y)
fz=diff(f,z)

grad_f=[fx,fy,fz]
nvec=[fx.subs({x:p[0],y:p[1],z:p[2]}) for fx in grad_f]
d=np.dot(nvec,p)
tangent_plane=simplify(Eq(np.dot(nvec,[x,y,z]),d))
tangent_plane_z=solve(tangent_plane,z)[0]
plane_z_func=lambdify((x,y),tangent_plane_z,'numpy')

print(f'The equation of the tangent plane to the surface f(x,y,z)={f} at the point {p} is: {tangent_plane}')

#b
normal_line_at_p=[p[i]+t*nvec[i] for i in range(len(p))]
print(f'The parametric equation of the normal line to the ellipsoid at the point {p}: ')
print(f'x={normal_line_at_p[0]},y={normal_line_at_p[1]},z={normal_line_at_p[2]}')
nend=[comp.subs(t,1) for comp in normal_line_at_p]

def mag(r):
    return simplify(sqrt(np.dot(r,r)))

def find_angle_between(n1,n2):
    n1_mag=mag(n1)
    n2_mag=mag(n2)
    angle=float(acos(np.dot(n1,n2)/(n1_mag*n2_mag)))
    return angle

angle_with_xy_plane=find_angle_between(nvec,[0,0,1])
print(f'The angle betweeen the tangent plane at {p} adn the xy plane is {np.rad2deg(angle_with_xy_plane):.3f}degrees.')
a=np.sqrt(18)
b=np.sqrt(18/4)
c=np.sqrt(18)

u=np.linspace(0,2*np.pi,100)
v=np.linspace(0,np.pi,100)
xe=a*np.outer(np.cos(u),np.sin(v))
ye=b*np.outer(np.sin(u),np.sin(v))
ze=c*np.outer(np.ones_like(u),np.cos(v))

fig=plt.figure(figsize=(12,9))
ax=fig.add_subplot(111,projection='3d')
ax.plot_surface(xe,ye,ze,cmap='jet',edgecolor='k',alpha=0.7,label='Ellipsoid')

xval=np.linspace(-5,5,100)
yval=np.linspace(-3,3,100)
X,Y=np.meshgrid(xval,yval)
Z=plane_z_func(X,Y)
ax.plot_surface(X,Y,Z,color='m',alpha=0.5,label='Tangent plane at (1,2,1)')
ax.set_zlim(-c,c)

ax.scatter(p[0],p[1],p[2],color='b',s=300,label='point(1,2,1)')
ax.quiver(p[0],p[1],p[2],nend[0],nend[1],nend[2],length=2,normalize=True,color='b',label='Normal line at (1,2,1)')

ax.legend(loc='best')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title(r"Ellipsoid $x^2+ 4y^2 +z^2 =18$")
ax.view_init(elev=18.18,azim=5)


plt.show()


