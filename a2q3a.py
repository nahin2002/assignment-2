import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

t = sp.symbols('t',real=True)

r=sp.Matrix([sp.exp(t), sp.exp(t)*sp.cos(t), sp.exp(t)*sp.sin(t)])
print(r)
r_prime=r.diff(t)
r_prime2=r.diff(t,2)
print(r_prime2)
r_prime3=r.diff(t,3)
print(r_prime3)

def Tangent(r,tval):
 T=r/(r.norm())
 return T.subs(t,tval).evalf()

def Normal(r,tval):
 T=r/(r.norm())
 N=T.diff(t)
 N=N/N.norm()
 return N.subs(t,tval).evalf()

def Binormal(r,tval):
 T=r/(r.norm())
 N=T.diff(t)
 N=N/N.norm()
 B=T.cross(N)
 return B.subs(t,tval).evalf()

def curvature(r1,r2,tval):
 kappa=(r1.cross(r2)).norm()/(r1.norm()**3)
 return kappa.subs(t,tval).evalf()

def torsion(r1,r2,r3,tval):
 tao= r1.dot(r2.cross(r3))/(r1.cross(r2)).norm()**2
 return tao.subs(t,tval).evalf()

print(f'unit tangent vector at t=0 is : {Tangent(r_prime,0)} \n')
print(f'unit normal vector at t=0 is : {Normal(r_prime,0)} \n')
print(f'binormal vector at t=0 is : {Binormal(r_prime,0)} \n')
print(f'curvature at t=0 is : {curvature(r_prime,r_prime2,0)} \n')
print(f'torsion at t=0 is : {torsion(r_prime,r_prime2,r_prime3,0)} \n')

w=sp.Matrix([2*sp.cos(t),3*sp.sin(t),0])
w_prime=sp.diff(w,t)
w_prime2=sp.diff(w,t,2)
w_prime3=sp.diff(w,t,3)
print(f'for w: \n')

# At t=0
print(f'unit tangent vector at t=0 is : {Tangent(w_prime,0)} \n')
print(f'unit normal vector at t=0 is : {Normal(w_prime,0)} \n')
print(f'binormal vector at t=0 is : {Binormal(w_prime,0)} \n')
print(f'curvature at t=0 is : {curvature(w_prime,w_prime2,0)} \n')
print(f'torsion at t=0 is : {torsion(w_prime,w_prime2,w_prime3,0)} \n')

#t=2*pi
print(f'unit tangent vector at t=2*pi is : {Tangent(w_prime,2*sp.pi)} \n')
print(f'unit normal vector at t=2*pi is : {Normal(w_prime,2*sp.pi)} \n')
print(f'binormal vector at t=2*pi is : {Binormal(w_prime,2*sp.pi)} \n')
print(f'curvature at t=2*pi is : {curvature(w_prime,w_prime2,2*sp.pi)} \n')
print(f'torsion at t=2*pi is : {torsion(w_prime,w_prime2,w_prime3,2*sp.pi)} \n')

t_values = np.linspace(0, 2 * np.pi, 400)

kappa1_values=[]
for t_val in t_values:
 kappa1_values.append(curvature(r_prime,r_prime2,t_val)) 

kappa2_values =[]
for t_val in t_values:
 kappa2_values.append(curvature(w_prime,w_prime2,t_val)) 

plt.figure(figsize=(10, 6))
plt.plot(t_values, kappa1_values, label='Curvature of r1(t)')
plt.plot(t_values, kappa2_values, label='Curvature of r2(t)')
plt.xlabel('t')
plt.ylabel('Curvature κ(t)')
plt.title('Curvature of the Curves')
plt.legend()
plt.grid(True)
plt.show()












