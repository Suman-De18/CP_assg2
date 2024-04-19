def equation_to_solve(a):
    t = 10
    return -500 + a * t
import numpy as np
import matplotlib.pyplot as plt

# Initial interval [a_low, a_high]
a_low = 0
a_high = 60
v0=np.zeros(6)
# Maximum number of iterations
max_iterations = 6

for i in range(max_iterations):
    a_mid = (a_low + a_high) / 2
    
    f_low = equation_to_solve(a_low)
    f_mid = equation_to_solve(a_mid)
    
    # Update interval
    if f_low * f_mid < 0:
        a_high = a_mid
    else:
        a_low = a_mid
    v0[i]=a_mid

t0=0
x0=0
g=10
h=0.1
n_steps=int((10 - x0) / h)
t = np.zeros((n_steps + 1,6))
x = np.zeros((n_steps + 1,6))
v=np.zeros((n_steps+1,6))

# Initial values
x[0] = x0
t[0] = t0
for j in range(6):
    v[0]=v0[j]
    for i in range(n_steps):
       t[i+1,j]=t[i,j]+h
       v[i+1,j]=v[i,j] - h*g
       x[i+1,j]=x[i,j]+h*v[i,j]
       
plt.figure(figsize=(10, 6))
colors = ['b', 'g', 'r', 'c', 'm','yellow']
for j in range(6):
    plt.plot(t[:,j],x[:,j],label=f'initial velocity : {v0[j]}',color=colors[j])
plt.title('Shooting Method')
plt.xlabel('t')
plt.ylabel('x')
plt.grid(True)

# Define the exact solution
t_exact=np.linspace(0,10,100)
x_exact = -5 * t_exact**2 + 50 * t_exact

# Plot the exact solution

plt.plot(t_exact, x_exact, label='Exact Solution (initial velocity=50)', color='black',linestyle='--')
plt.legend()
plt.show()
plt.savefig('shooting_method.jpg')


