import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the differential equation
def fun(t, y):
    return np.cos(2*t) + np.sin(3*t)

# Define the initial condition
y0 = [1]

t_span = [0, 1]

# Solve the differential equation
sol = solve_ivp(fun, t_span, y0, t_eval=np.linspace(0, 1, 100))

plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label='y(t)', color='yellow')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Solution of y\' = cos(2t) + sin(3t)')
plt.grid(True)
plt.legend()

# Define the function
def f(t):
    return (1/6) * (8 - 2 * np.cos(3*t) + 3 * np.sin(2*t))

# Generate values for t
t_values = np.linspace(0, 1, 100)  # t ranges from 0 to 1

# Calculate function values
y_values = f(t_values)

plt.plot(t_values, y_values,linestyle='dotted', label=r'$\frac{1}{6}(8 - 2 \cos(3t) + 3 \sin(2t))$')
plt.grid(True)
plt.legend()
plt.show()
