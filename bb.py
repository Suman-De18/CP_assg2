import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the ODE
def ode(t, y):
    dydt = [y[1], -10]
    return dydt

# Exact solution
def exact_solution(t):
    return -5 * t**2 + 50 * t

# Time points
t_span = [0, 10]

# Initial conditions range
initial_conditions = np.linspace(0, 10, 100)

# Errors
errors = []

# Solve the ODE for each initial condition
for init in initial_conditions:
    sol = solve_ivp(ode, t_span, [0, init], t_eval=np.linspace(0, 10, 100))
    error = np.linalg.norm(sol.y[0] - exact_solution(sol.t))
    errors.append(error)

# Find the index of the minimum error
min_error_index = np.argmin(errors)
optimal_init = initial_conditions[min_error_index]

print(f"Optimal initial condition: {optimal_init}")

# Solve the ODE with the optimal initial condition
optimal_sol = solve_ivp(ode, t_span, [0, optimal_init], t_eval=np.linspace(0, 10, 100))

# Plot the solution
plt.figure(figsize=(8, 6))
plt.plot(optimal_sol.t, optimal_sol.y[0], label='Computed Solution', color='green')
plt.plot(optimal_sol.t, exact_solution(optimal_sol.t), label='Exact Solution', color='blue')
plt.title('Computed vs Exact Solution')
plt.xlabel('Time (t)')
plt.ylabel('x(t)')
plt.grid(True)
plt.legend()
plt.show()
