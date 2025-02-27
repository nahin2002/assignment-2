import sympy as sp
x, y = sp.symbols('x y',real=True)
f = y**2 * sp.cos(x - y)
f_x = f.diff(x)
f_y = f.diff(y)
f_xx = f_x.diff(x)
f_yy = f_y.diff(y)
f_xy = f_x.diff(y)
f_yx = f_y.diff(x)
laplace_eq = f_xx + f_yy

if laplace_eq==0:
    print('laplace eqn satisfied')
else:
    print('laplace eqn not satisfied')

# Check Cauchy-Riemann equations
u = sp.re(f)
v = sp.im(f)

if u.diff(x)==v.diff(y) and u.diff(y)==-v.diff(x):
    print('Cauchy-Riemann conditions  satisfied')
else:
    print('Cauchy Riemann conditons not satisfied')

if f_xy==f_yx:
    print('fxy=fyx')
else:
    print('fxy != fyx')
