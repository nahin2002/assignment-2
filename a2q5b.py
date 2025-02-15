import sympy as sp

# Define the symbolic variables
x, y, z, λ = sp.symbols('x y z λ')

# Define the temperature function T(x, y, z)
T = 8 * x**2 + 4 * y * z - 16 * z + 600

# Define the constraint function g(x, y, z) representing the ellipsoid
g = 4 * x**2 + y**2 + 4 * z**2 - 16

# Define the Lagrange function L(x, y, z, λ)
L = T + λ * g

# Compute the partial derivatives of L
L_x = sp.diff(L, x)
L_y = sp.diff(L, y)
L_z = sp.diff(L, z)
L_λ = sp.diff(L, λ)

# Solve the system of equations L_x = 0, L_y = 0, L_z = 0, L_λ = 0
solutions = sp.solve([L_x, L_y, L_z, L_λ], (x, y, z, λ))

# Find the hottest point by evaluating T at the solutions
hottest_point = None
max_temperature = -sp.oo

for sol in solutions:
    temp = T.subs({x: sol[0], y: sol[1], z: sol[2]})
    if temp > max_temperature:
        max_temperature = temp
        hottest_point = sol

# Print the results
print(f"Hottest point: {hottest_point}")
print(f"Maximum temperature: {max_temperature}")