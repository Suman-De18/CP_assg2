import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

def fun(x, y):
    y1, y2 = y
    dy1_dx = y2
    if y1.any()!= 0:
        dy2_dx = y2*np.cos(x) - y1*np.log(y1)
    else:
        dy2_dx = y2*np.cos(x)
    return [dy1_dx, dy2_dx]

def bc(ya, yb):
    return [ya[0] - 1, yb[0] - np.exp(1)]

# Initial guess for the solution
x = np.linspace(0, np.pi/2, 100)
y_guess = np.zeros((2, x.size))

# Solve the boundary value problem
sol = solve_bvp(fun, bc, x, y_guess)

# Plot the solution
x_plot = np.linspace(0, np.pi/2, 100)
y_plot = sol.sol(x_plot)[0]

plt.figure(figsize=(8, 6))
plt.plot(x_plot, y_plot, label='y(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solution of the Boundary Value Problem')
plt.legend()
plt.grid(True)
plt.show()
