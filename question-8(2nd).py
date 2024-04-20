import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the differential equation
def fun(t, y):
    return 1 - (t - y)**2

# Define the initial condition
y0 = [1]

# Define the time span
t_span = [2, 3]

# Solve the differential equation
sol = solve_ivp(fun, t_span, y0, t_eval=np.linspace(2, 3, 100))

# Plot the solution
plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label='y(t)', color='b')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Solution of y\' = 1 - (t - y)^2')
plt.grid(True)
plt.legend()
plt.show()
