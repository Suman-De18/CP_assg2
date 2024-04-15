import numpy as np
import matplotlib.pyplot as plt

# Define the function dy/dx = -20(y - x)^2 + 2x
def dy_dx(x, y):
    return -20 * (y - x)**2 + 2 * x

# Initial condition
x0 = 0
y0 = 1 / 3

# Step size
h = 0.01

# Number of steps
n_steps = int((1 - x0) / h)

# Arrays to store x and y values
x_values = np.zeros(n_steps + 1)
y_values = np.zeros(n_steps + 1)

# Initial values
x_values[0] = x0
y_values[0] = y0

# Backward Euler's method
for i in range(n_steps):
    x = x_values[i]
    y = y_values[i]
    
    # Calculate y_{n+1} using backward Euler's method
    y_next = (y + 20 * h * x**2 - 2 * h * x) / (1 + 20 * h)
    
    # Update x and y values
    x_values[i + 1] = x + h
    y_values[i + 1] = y_next

# Plot the solution
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, label='Numerical solution', color='blue')
plt.title('Numerical Solution using Backward Euler\'s Method')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
