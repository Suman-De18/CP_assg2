import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the differential equation
def fun(t, y):
    return 1 + y / t

# Define the initial condition
y0 = [2]

t_span = [1, 2]

# Solve the differential equation
sol = solve_ivp(fun, t_span, y0, t_eval=np.linspace(1, 2, 100))

plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label='y(t)', color='yellow')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Solution of y\' = 1 + y/t')
plt.grid(True)
plt.legend()

# Define the function
def f(t):
    return 2*t + t*np.log(t)

# Generate values for t
t_values = np.linspace(1, 2, 100)  # t ranges from 1 to 2

# Calculate function values
y_values = f(t_values)

plt.plot(t_values, y_values,linestyle='dotted', label=r'$2t + t \log(t)$')
plt.grid(True)
plt.legend()
plt.show()


