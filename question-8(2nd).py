import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the differential equation
def fun(t, y):
    return 1 - (t - y)**2

# Define the initial condition
y0 = [1]

t_span = [2, 3]

# Solve the differential equation
sol = solve_ivp(fun, t_span, y0, t_eval=np.linspace(2, 3, 100))

plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label='y(t)', color='yellow')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Solution of y\' = 1 - (t - y)^2')
plt.grid(True)
plt.legend()

# Define the function
def f(t):
    return (1 - 3*t + t**2) / (-3 + t)

# Generate values for t
t_values = np.linspace(2, 3, 100)  # t ranges from 2 to 3

# Calculate function values
y_values = f(t_values)

plt.plot(t_values, y_values,linestyle='dotted', label=r'$\frac{{1 - 3t + t^2}}{{-3 + t}}$')
plt.grid(True)
plt.legend()
plt.show()