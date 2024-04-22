import numpy as np
import matplotlib.pyplot as plt

g=10
tf=10
n_steps=100
t=np.linspace(0,tf,n_steps+1)
h=tf/n_steps
candidate_sol=[]
x= np.zeros(n_steps+1)

x_exact=np.zeros(n_steps+1)
x_exact = -5 * t**2 + 50 * t

for i in range(10000):
    x_last=x.copy()
    candidate_sol.append(x.copy())
    x[1:-1]=0.5*(x[2:] + x[:-2])+0.5*g*h**2 #(x''=(x(t+h)+x(t-h)-2x(t))/h^2)

iteration=[4000,5000,6000,7000,8000]
for j in iteration:
    plt.plot(t,candidate_sol[j],label=f'candidate sol at iteration:{j}',linestyle='--')
plt.plot(t,candidate_sol[-1],label='Numerical sol',linestyle=':',color='black')
plt.plot(t, x_exact, label='Exact Solution', linewidth=2,color='yellow')
plt.grid(True)
plt.legend()
plt.xlabel("t")
plt.ylabel("x(t)")
plt.title('Relaxation method')
plt.show()