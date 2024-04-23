import numpy as np
import matplotlib.pyplot as plt

def system(t, u):
    u1, u2, u3 = u
    du1 = u1 + 2*u2 - 2*u3 + np.exp(-t)
    du2 = u2 + u3 - 2*np.exp(-t)
    du3 = u1 + 2*u2 + np.exp(-t)
    return np.array([du1, du2, du3])

def rk4_system(system, t0, u0, h, n):
    t = np.zeros(n)
    u = np.zeros((n, len(u0)))
    t[0] = t0
    u[0] = u0
    
    for i in range(n-1):
        k1 = h * system(t[i], u[i])
        k2 = h * system(t[i] + h/2, u[i] + k1/2)
        k3 = h * system(t[i] + h/2, u[i] + k2/2)
        k4 = h * system(t[i] + h, u[i] + k3)
        
        u[i+1] = u[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
        t[i+1] = t[i] + h
        
    return t, u

# Initial conditions
t0 = 0
u0 = np.array([3, -1, 1])

# Step size and number of steps
h = 0.01
n = int(1 / h) + 1

# Solve using RK4
t, u = rk4_system(system, t0, u0, h, n)

# Plot the results
plt.figure(figsize=(12, 8))
plt.plot(t, u[:, 0], label='u1(t)', color='blue')
plt.plot(t, u[:, 1], label='u2(t)', color='red')
plt.plot(t, u[:, 2], label='u3(t)', color='green')
plt.title('Solution of the system of ODEs using RK4')
plt.xlabel('t')
plt.ylabel('u')
plt.grid(True)
plt.legend()
plt.xlim(0, 1)
plt.show()
