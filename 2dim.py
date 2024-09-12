import numpy as np
import matplotlib.pyplot as plt
from numba import njit

#%% functions 

@njit
def solve_tridiag(lower, diag, upper, right, k1=0, k2=0, m1=0, m2=0):
    # (N-1) equations
    # boundary conditions:
    # y[0] = k1*y[1] + m1
    # y[N] = k2*y[N-1] + m2
    
    N = diag.size + 1
    a = np.empty(N)
    b = np.empty(N)
    c = np.empty(N)
    f = np.empty(N)
    a[1:N] = lower
    b[1:N] = upper
    c[1:N] = diag
    f[1:N] = right
    alpha = np.empty(N+1)
    beta  = np.empty(N+1)
    alpha[1] = k1
    beta[1] = m1
    for j in range(1, N):
        alpha[j+1] = -b[j] / (alpha[j]*a[j] + c[j])
        beta[j+1] = (f[j] - a[j]*beta[j]) / (a[j]*alpha[j] + c[j])
    
    y = np.empty(N+1)
    y[N] = (k2*beta[N] + m2) / (1 - k2*alpha[N])
    for j in range(N-1, -1, -1):
        y[j] = alpha[j+1]*y[j+1] + beta[j+1]
    return y


@njit
def straight(a2, T, u0, c, L1, L2, N1, N2, M):
    
    h1 = L1/N1
    h2 = L2/N2
    tau = T/M
    gamma1 = a2*tau/(2*h1**2)
    gamma2 = a2*tau/(2*h2**2)
    
    u = np.empty((M + 1, N1 + 1, N2 + 1))
    u[0] = np.copy(u0)
    for n in range(1, M+1):
        u_prev = np.copy(u[n - 1])
        for j in range(1, N2):
            right = u_prev[:,j]*(1 - 2*gamma2) + gamma2*(u_prev[:,j-1] + u_prev[:,j+1])
            right = right[1:-1]
            lower = np.ones(N1 - 1)*(-gamma1)
            upper = lower
            diag = np.ones(N1 - 1)*(2*gamma1 + 1 + 0.5*T/M*c[1:-1,j])
            u[n][:, j] = solve_tridiag(lower, diag, upper, right)
        u_prev = np.copy(u[n])
        for i in range(1, N1):
            right = u_prev[i,:]*(1 - 2*gamma1 - 0.5*T/M*c[i,:]) + gamma1*(u_prev[i+1,:] + u_prev[i-1,:])
            right = right[1:-1]
            lower = np.ones(N2 - 1)*(-gamma2)
            upper = lower
            diag = np.ones(N2 - 1)*(2*gamma2 + 1)
            u[n][i, :] = solve_tridiag(lower, diag, upper, right)
            
    return u


@njit
def inverse(a2, T, eta, eta_d, u1, u1_laplace, c, L1, L2, N1, N2, M, NB): #eta, u1 - arrays

    def B(a2, T, eta_d, f, c, L1, L2, N1, N2, M):
        y = straight(a2, T, f, c, L1, L2, N1, N2, M)
        y = np.ascontiguousarray(y.transpose((1,2,0)))
        y = np.ascontiguousarray(y.reshape(-1, M+1))
        eta_d = np.ascontiguousarray(eta_d)
        integral = np.dot(np.ascontiguousarray(y[:, 1:-1]), eta_d[1:-1])
        integral += 0.5*(y[:, 0]*eta_d[0] + y[:, -1]*eta_d[-1])
        integral *= T/M
        integral = np.ascontiguousarray(integral).reshape(N1+1, N2+1)
        y = y.reshape(N1+1, N2+1, M+1)
        y = y.transpose((2, 0, 1))
        return eta[-1]*y[-1] - integral
    
    f = -a2*u1_laplace + c*u1
    u0 = np.empty((NB + 1, N1 + 1, N2 + 1))
    u0[0] = f/eta[0]
    for k in range(NB):
        print(f'{k} iter')
        b = B(a2, T, eta_d, u0[k], c, L1, L2, N1, N2, M)
        u0[k + 1] = (f + b)/eta[0]
    return u0


#%% experiments

a2 = 16/25
T = 2
L1 = np.pi
L2 = 2*L1
N1 = 100
N2 = 100
M = 100
NB = 10

x = np.linspace(0, L1, N1+1)
y = np.linspace(0, L2, N2+1)
xgrid, ygrid = np.meshgrid(x, y, indexing='ij')
t = np.linspace(0, T, M + 1)

eta = 1 + t**2
eta_d = 2*t

c = lambda x, y: x + y
c = c(xgrid, ygrid)

x = xgrid
y = ygrid
u0_true = x**6*y*(L1-x)*(L2-y) + x*y*(L1-x)**6*(L2-y) - \
          x*y**4*(L1-x)*(L2-y) - x*y*(L1-x)*(L2-y)**4

u = straight(a2, T, u0_true, c, L1, L2, N1, N2, M)
u1 = (np.dot(u[1:-1].transpose((1,2,0)), eta[1:-1]) + 0.5*(u[0]*eta[0] + u[-1]*eta[-1]))*T/M
u1_xx = np.vstack((np.zeros((1,N2+1)), np.diff(u1, 2, axis=0), np.zeros((1,N2+1))))/(L1/N1)**2
u1_yy = np.hstack((np.zeros((N1+1,1)), np.diff(u1, 2, axis=1), np.zeros((N1+1,1))))/(L2/N2)**2
u1_laplace = u1_xx + u1_yy

u0_ret = inverse(a2, T, eta, eta_d, u1, u1_laplace, c, L1, L2, N1, N2, M, NB)

#%% plot errors

errors = np.max(np.max(np.abs(u0_ret - u0_true), axis=1), axis=1)
plt.xlabel('Число итераций')
plt.ylabel('Величина ошибки')
plt.grid(True)
plt.plot(errors, 'o-')


#%% plot surface

x = xgrid
y = ygrid
u0_true = x**6*y*(L1-x)*(L2-y) + x*y*(L1-x)**6*(L2-y) - \
          x*y**4*(L1-x)*(L2-y) - x*y*(L1-x)*(L2-y)**4

fig = plt.figure(figsize=(8,8))
axes = fig.add_subplot(projection='3d')
axes.plot_surface(xgrid, ygrid, u0_true, cmap='inferno')
plt.show()

#%% Carson transform H(p)

p = np.linspace(0, 1.7, 1000)
plt.grid(True)
plt.xlabel('p')
plt.ylabel('H(p)')
plt.plot(p, (p+1-np.exp(-4*p)*(5*p+1))/p, c='green')
