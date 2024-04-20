import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the differential equation
def fun(t, y):
    return np.cos(2*t) + np.sin(3*t)

# Define the initial condition
y0 = [1]

# Define the time span
t_span = [0, 1]

# Solve the differential equation
sol = solve_ivp(fun, t_span, y0, t_eval=np.linspace(0, 1, 100))

# Plot the solution
plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label='y(t)', color='b')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Solution of y\' = cos(2t) + sin(3t)')
plt.grid(True)
plt.legend()
plt.show()
plt.savefig()