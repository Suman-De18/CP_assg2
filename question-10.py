import numpy as np
import matplotlib.pyplot as plt

def rk4_step(f, t, y, h):
    k1 = h * f(t, y)
    k2 = h * f(t + 0.5*h, y + 0.5*k1)
    k3 = h * f(t + 0.5*h, y + 0.5*k2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6.0

def adaptive_rk4(f, t0, y0, t_end, tol):
    t = [t0]
    y = [y0]
    h = 0.01
    while t[-1] < t_end:
        y1 = rk4_step(f, t[-1], y[-1], h)
        y2 = rk4_step(f, t[-1], y[-1], h/2)
        y2 = rk4_step(f, t[-1] + h/2, y2, h/2)
        
        error = np.abs(y2 - y1)
        if error < tol:
            y.append(y2)
            t.append(t[-1] + h)
        h *= 0.5 * (tol / error)**0.2
    
    return np.array(t), np.array(y)

def f(t, y):
    return (y**2 + y) / t

t0 = 1.0
y0 = -2.0
t_end = 3.0
tol = 1e-4

t, y = adaptive_rk4(f, t0, y0, t_end, tol)

plt.figure(figsize=(10, 6))
plt.plot(t, y, label='Numerical Solution', marker='o')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Solution of y\' = (y^2 + y) / t using Adaptive RK4')
plt.grid(True)
plt.legend()
plt.show()