import sympy as smp
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

t,u=smp.symbols('t u',real=True)

def r(t):
 r=smp.Matrix([smp.cos(t),smp.sin(t),t])
 return np.array(r)

def r_prime(t):
 v=smp.diff(r,t)
 return np.array(v)



def arc_length(t_val):
 s_v=smp.sqrt((smp.cos(t))**2+(smp.sin(t))**2+1)
 s_v=s_v.subs(t,t_val)
 return smp.integrate(s_v,(u,0,t))




def t_from_s(s):
  return s/np.sqrt(2)

t_initial=0
s_final=10
t_final=t_from_s(s_final)

r_initial=r(t_initial)
print(r_initial)
r_final=r(t_final)
print(r_final)

t_value=np.linspace(0,t_final,500)
helix=np.array([r(t) for t in t_value])

fig=plt.figure(figsize=(10,6))
ax=fig.add_subplot(111,projection='3d')
ax.plot(helix[:,0],helix[:,1],helix[:,2],label='Helix: r(t) = (cos(t), sin(t), t)')

ax.scatter(1,0,0, color='red', label='Initial position (1, 0, 0)')
ax.scatter(0.70535,0.70886,7.07107,color='blue',label='final position after 10 units')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()


