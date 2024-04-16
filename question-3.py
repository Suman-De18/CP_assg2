import numpy as np
import matplotlib.pyplot as plt

# Define the system of first-order ODEs
def system(x, y):
    y1, y2 = y
    dy1_dx = y2
    dy2_dx = 2*y2 - y1 + x*np.exp(x) - x
    return [dy1_dx, dy2_dx]

# Initial conditions
x0 = 0
y0 = [0, 0]

# Step size
h = 0.1

# Number of steps
n_steps = int((1 - x0) / h)

# Arrays to store x, y1, and y2 values
x_values = np.zeros(n_steps + 1)
y1_values = np.zeros(n_steps + 1)
y2_values = np.zeros(n_steps + 1)

# Initial values
x_values[0] = x0
y1_values[0], y2_values[0] = y0

# 4th-order Runge-Kutta method
for i in range(n_steps):
    x = x_values[i]
    y = [y1_values[i], y2_values[i]]
    
    k1 = np.array(system(x, y))
    k2 = np.array(system(x + 0.5*h, y + 0.5*h*k1))
    k3 = np.array(system(x + 0.5*h, y + 0.5*h*k2))
    k4 = np.array(system(x + h, y + h*k3))
    
    y_next = y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    
    x_values[i + 1] = x + h
    y1_values[i + 1] = y_next[0]
    y2_values[i + 1] = y_next[1]

# Plot the solution
plt.figure(figsize=(10, 6))
plt.plot(x_values, y1_values, label='y(x)', color='blue')
plt.plot(x_values, y2_values, label="y'(x)", color='red')
plt.title('Solution using 4th-order Runge-Kutta Method')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
print(y1_values)
