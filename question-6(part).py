import numpy as np
import matplotlib.pyplot as plt

# Constants
g = 10
t_final = 10
N = 1000
dt = t_final / N
t = np.linspace(0, t_final, N)
# Function to solve the ODE for given initial velocity v0
def solve_ode(v0):
    x = np.zeros(N)
    v = np.zeros(N)
    
    x[0] = 0
    v[0] = v0
    
    for i in range(1, N):
        v[i] = v[i-1] - g * dt
        x[i] = x[i-1] + v[i] * dt
        
    return x

# Initial velocities to try
v0_values = np.linspace(-100, 100, 1000)  # Adjust the range based on the problem

# Find the initial velocity that results in x=0 at t=10
x_vs_t = [solve_ode(v0) for v0 in v0_values]
index_zero = np.argmin(np.abs([x[-1] for x in x_vs_t]))

v0_solution = v0_values[index_zero]
x_solution = x_vs_t[index_zero]

# Print the solution
print(f"Initial velocity v0 that satisfies x=0 at t=10: {v0_solution}")

# Plot x vs t for the solution
plt.figure(figsize=(10, 5))
plt.plot(t, x_solution, label=f'Position x vs t with v0={v0_solution}')
plt.xlabel('Time t')
plt.ylabel('Position x')
plt.title('Position x as a function of Time t')
plt.legend()
plt.grid(True)
plt.show()
