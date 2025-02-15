import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
x,y=sp.symbols('x y',real=True)

f1=4*x*y-x**4-y**4
f1_x=sp.diff(f1,x)
f1_y=sp.diff(f1,y)
f1_xx=sp.diff(f1_x,x)
f1_xy=sp.diff(f1_x,y)
f1_yy=sp.diff(f1_y,y)

crit_points_1=sp.solve([f1_x,f1_y],(x,y))

f2 = 4*x**2*sp.exp(y) - 2*x**4 - sp.exp(4*y)
f2_x = sp.diff(f2, x)
f2_y = sp.diff(f2, y)
f2_xx = sp.diff(f2_x, x)
f2_yy = sp.diff(f2_y, y)
f2_xy = sp.diff(f2_x, y)

crit_points_2 = sp.solve([f2_x, f2_y], (x, y))

def classification(fxx,fyy,fxy,cp):
    H=fxx*fyy-fxy**2
    for point in cp:
        c1=H.subs({x: point[0],y: point[1]})
        if c1>0 and fxx.subs({x: point[0],y: point[1]})>0:
            print(f'relative minima at {point}')
        elif c1>0 and fxx.subs({x: point[0],y: point[1]})<0:
            print(f'relative maxima at {point}')
        elif c1<0:
            print(f'saddle point at {point}')
        elif c1==0:
           print('no conclusion')
        else:
           print('invalid')

print('for f1: ')
classification(f1_xx,f1_yy,f1_xy,crit_points_1) 
print('\n for f2: ')
classification(f2_xx,f2_xy,f2_yy,crit_points_2)           

x_vals = np.linspace(-2, 2, 100)
y_vals = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z1 = 4*X*Y - X**4 - Y**4
Z2 = 4*X**2*np.exp(Y) - 2*X**4 - np.exp(4*Y)

fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z1, cmap='viridis', alpha=0.8)
ax1.set_title("Surface plot of f(x,y) = 4xy - x^4 - y^4")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("f(x, y)")

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, Z2, cmap='plasma', alpha=0.8)
ax2.set_title("Surface plot of f(x,y) = 4x^2 e^y - 2x^4 - e^{4y}")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("f(x, y)")

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax1 = axes[0]
ax1.contour(X, Y, Z1, levels=30, cmap="viridis")


for point in crit_points_1:
 ax1.plot(point[0], point[1], 'ro')

ax1.set_title("Contour plot of f(x,y) = 4xy - x^4 - y^4")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

ax2 = axes[1]
ax2.contour(X, Y, Z2, levels=30, cmap="plasma")

for point in crit_points_2:
 ax2.plot(point[0], point[1], 'ro')

ax2.set_title("Contour plot of f(x,y) = 4x^2 e^y - 2x^4 - e^{4y}")
ax2.set_xlabel("x")
ax2.set_ylabel("y")

plt.tight_layout()
plt.show()

