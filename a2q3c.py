import sympy as sp

t=sp.symbols('t',real=True)

x=sp.cos(t)
y=sp.sin(t)
z=sp.tan(t)

w=sp.sqrt((x**2)+(y**2)+(z**2))

w_diff=w.diff(t)

print(w_diff)
t0=sp.pi/4
print(w_diff.subs(t,t0).evalf())
