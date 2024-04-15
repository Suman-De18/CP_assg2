import numpy as np
import matplotlib.pyplot as plt



# Initial condition
t0 = 1
y0 = 1

# Step size
h = 0.1

# Number of steps
n_steps = int((2 - t0) / h)

# Arrays to store t and y values
t_values = np.zeros(n_steps + 1)
y_values = np.zeros(n_steps + 1)
absolute_errors = np.zeros(n_steps + 1)
relative_errors = np.zeros(n_steps + 1)

# Initial values
t_values[0] = t0
y_values[0] = y0

# Euler's method
for i in range(n_steps):
    t = t_values[i]
    y = y_values[i]
    
    # Calculate y_{n+1} using Euler's method
    y_next = y + h * (y / t - (y / t)**2)
    
    # Update t and y values
    t_values[i + 1] = t + h
    y_values[i + 1] = y_next
    
    # Calculate exact solution for error calculation
    exact_solution = t_values[i + 1] / (1 + np.log(t_values[i + 1]))
    
    # Calculate absolute and relative errors
    absolute_errors[i + 1] = np.abs(y_next - exact_solution)
    relative_errors[i + 1] = absolute_errors[i + 1] / np.abs(exact_solution)

# Plot the numerical solution
plt.figure(figsize=(10, 6))
plt.plot(t_values, y_values, label='Numerical solution', color='blue')
plt.title('Numerical Solution using Euler\'s Method')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# Compute and print the absolute and relative errors
print("absolute error at each mesh point:",absolute_errors)
print("relative error at each mesh point:",relative_errors)
print(f"Maximum absolute error: {np.max(absolute_errors):.6f}")
print(f"Maximum relative error: {np.max(relative_errors):.6f}")
