import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the differential equation
def fun(t, y):
    return t * np.exp(3*t) - 2*y

# Define the initial condition
y0 = [0]


t_span = [0, 1]

# Solve the differential equation
sol = solve_ivp(fun, t_span, y0, t_eval=np.linspace(0, 1, 100))

# Plot the solution
plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label='y(t)', color='b')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Solution of y\' = te^{3t} - 2y')
plt.grid(True)
plt.legend()
plt.show()