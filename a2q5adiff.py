import numpy as np
import sympy as smp
import matplotlib.pyplot as plt
x,y,z,t=smp.symbols('x y z t',real=True)

f=x**2+4*y**2+z**2-18

grad_f=smp.Matrix([f.diff(x),f.diff(y),f.diff(z)])
p=(1,2,1)
n_vec=grad_f.subs({x:p[0],y:p[1],z:p[2]})
tangent_plane=n_vec[0]*(x-p[0])+n_vec[1]*(y-p[1])+n_vec[2]*(z-p[2])
t_plane=tangent_plane.simplify()
print(t_plane)
z=18-x-8*y

normal_eqn=smp.Matrix([p[0]+t*n_vec[0],p[1]+t*n_vec[1],p[2]+t*n_vec[2]])
print(normal_eqn)

unit_vec=smp.Matrix([0,0,1])
dot_prod=n_vec.dot(unit_vec)
mag_n1=smp.sqrt(n_vec.dot(n_vec))
mag_n2=smp.sqrt(unit_vec.dot(unit_vec))
theta=smp.acos((dot_prod)/(mag_n1*mag_n2))
print(smp.deg(theta).evalf())

x1=np.linspace(-3,3,100)
y1=np.linspace(-3,3,100)
X,Y=np.meshgrid(x1,y1)
Z=18-X-8*Y
fig=plt.figure(figsize=(12,8))
ax=fig.add_subplot(111,projection='3d')
ax.plot_surface(X,Y,Z,alpha=0.5)

t1=np.linspace(-3,3,100)
xx=2*t1+1
yy=16*t1+2
zz=2*t1+1
ax.plot(xx,yy,zz)

u=np.linspace(0,2*np.pi,100)
v=np.linspace(0,np.pi,100)
"""
U,V=np.meshgrid(u,v)
X=np.sqrt(18)*np.cos(U)*np.sin(V)
Y=np.sqrt(18/4)*np.sin(U)*np.sin(V)
Z=np.sqrt(18)*np.cos(V)
ax.plot_surface(X,Y,Z,alpha=0.6)
"""
X=np.outer(np.cos(u),np.sin(v))*np.sqrt(18)
Y=np.outer(np.sin(u),np.sin(v))*np.sqrt(18/4)
Z=np.outer(np.ones(np.size(u)),np.cos(v))*np.sqrt(18)
ax.plot_surface(X,Y,Z,alpha=0.5)
ax.scatter(p[0],p[1],p[2])
ax.set_zlim(-np.sqrt(18),np.sqrt(18))
ax.view_init(elev=18.18,azim=5)
plt.show()