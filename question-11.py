import numpy as np
import matplotlib.pyplot as plt
def f(t, x):
    return 1 / (x**2 + t**2)

def rk4(f, t0, x0, h, n):
    t = np.zeros(n)
    x = np.zeros(n)
    t[0] = t0
    x[0] = x0
    
    for i in range(n-1):
        k1 = h * f(t[i], x[i])
        k2 = h * f(t[i] + h/2, x[i] + k1/2)
        k3 = h * f(t[i] + h/2, x[i] + k2/2)
        k4 = h * f(t[i] + h, x[i] + k3)
        
        x[i+1] = x[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
        t[i+1] = t[i] + h
        
        # Check if t[i+1] exceeds the target time
        if t[i+1] >= 3.5e6:
            break
        
    return t[:i+2], x[:i+2]

# Initial conditions
t0 = 0
x0 = 1

# Step size
h = 1e3  # Adjusted to cover the required time span
n = int(3.5e6 / h) + 1

# Solve using RK4
t, x = rk4(f, t0, x0, h, n)

# Print the value of x at t = 3.5e6
print(f"x({t[-1]}) = {x[-1]}")

plt.plot(t,x,)
plt.title('Solution of dx/dt = 1/(x^2 + t^2)')
plt.xlabel("t")
plt.ylabel('x(t)')
plt.grid(True)

plt.show()