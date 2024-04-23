import numpy as np
import matplotlib.pyplot as plt

# Define the differential equations
def f(t, y, v):
    return v

def g(t, y, v):
    return (2*t*v - 2*y + t**3 * np.log(t)) / t**2

# Exact solution
def exact_solution(t):
    return 7*t/4 + (t**3/2) * np.log(t) - 3/4 * t**3

# Initial conditions
t0 = 1
y0 = 1
v0 = 0

# Step size
h = 0.001

# Number of steps
n = int((2 - t0) / h)

# Initialize arrays to store t, y, and v
t_values = np.zeros(n+1)
y_values = np.zeros(n+1)
v_values = np.zeros(n+1)

# Initial values
t_values[0] = t0
y_values[0] = y0
v_values[0] = v0

# Euler's method
for i in range(n):
    t = t_values[i]
    y = y_values[i]
    v = v_values[i]
    
    y_prime = f(t, y, v)
    v_prime = g(t, y, v)
    
    y_values[i+1] = y + h * y_prime
    v_values[i+1] = v + h * v_prime
    t_values[i+1] = t + h

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(t_values, y_values, label='Euler\'s method',linestyle='--',color='black')
plt.plot(t_values, exact_solution(t_values), label='Exact solution', color='yellow')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Euler\'s Method vs Exact Solution')
plt.legend()
plt.grid(True)
plt.show()
