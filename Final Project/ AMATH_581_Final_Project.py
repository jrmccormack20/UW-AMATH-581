# AMATH 581 Final Project
# Jonathan McCormack 
# 10DEC2025
import numpy as np
from scipy.integrate import solve_ivp
import time
import copy
import plotly.graph_objects as go

# Set the initial conditions
L = 2*np.pi # Length of Spatial Domain
n = 16 # Number of internal points
tspan = [0, 4]; dt = 0.5 # Set time endpoints and time step
A = np.array([-1, -1, -1])
B = -A

# Set Time Grid
t_eval = np.arange(tspan[0], tspan[1] + dt, dt)

# Set Spatial Mesh Grid
x = np.linspace(-L/2, L/2, n+1)[:n]
[X, Y, Z] = np.meshgrid(x, x, x)

# Set Spectral Mesh Grid
k = 2*np.pi*np.fft.fftfreq(n, d=L/n)
k[0] = 1e-6
[KX, KY, KZ] = np.meshgrid(k, k, k)
K = KX**2 + KY**2 + KZ**2

# Define Spectral Method functions
# Set Lattice Matix (unique to Gross-Pitaevskii system)
Lattice = (A[0]*np.sin(X)**2 + B[0]) * (A[1]*np.sin(Y)**2 + B[1]) * (A[2]*np.sin(Z)**2 + B[2])

# Define FFT equation
def fftrhs(t, psihatvec, K, Lattice):
    
    psihat = psihatvec.reshape((n, n, n), order = 'F')
    psi = np.fft.ifftn(psihat)
    nonlinearhat = np.fft.fftn((np.abs(psi)**2)*psi - Lattice*psi)

    rhs = (-1j*(0.5*K*psihat + nonlinearhat)).reshape(n**3, order = 'F')
    return rhs

# Evaluate solve_ivp with spectral methods
#---------------
# Part A: Initial Condition:  ψ(x, y, z) = cos(x)*cos(y)*cos(z)
tic = time.time()

# Set Initial Conditions
psi0 = np.cos(X)*np.cos(Y)*np.cos(Z)
psi0hat = np.fft.fftn(psi0)
psi0hatvec = psi0hat.reshape(n**3, order = 'F')

# Evaluate Initial Conditions
sol1 = solve_ivp(fftrhs, (t_eval[0], t_eval[-1]), psi0hatvec, t_eval = t_eval, args = [K, Lattice])

toc = time.time()
print(toc-tic)

# Store real and imaginary results 
A1 = copy.deepcopy(np.real(sol1.y.T))
print(np.shape(A1))

A2 = copy.deepcopy(np.imag(sol1.y.T))
print(np.shape(A2))
del psi0, psi0hat, psi0hatvec
#---------------

#---------------
# Part B: Initial Condition:  ψ(x, y, z) = sin(x) sin(y) sin(z)
tic = time.time()

# Set Initial Conditions
psi0 = np.sin(X)*np.sin(Y)*np.sin(Z)
psi0hat = np.fft.fftn(psi0)
psi0hatvec = psi0hat.reshape(n**3, order = 'F')

# Evaluate Initial Conditions
sol2 = solve_ivp(fftrhs, (t_eval[0], t_eval[-1]), psi0hatvec, t_eval = t_eval, args = [K, Lattice])

toc = time.time()
print(toc-tic)

# Store real and imaginary results 
A3 = copy.deepcopy(np.real(sol2.y.T))
print(np.shape(A3))

A4 = copy.deepcopy(np.imag(sol2.y.T))
print(np.shape(A4))
del psi0, psi0hat, psi0hatvec
#---------------

del tspan, dt, A, B, k, KX, KY, KZ, K, Lattice, tic, toc

# Plot Isosurfaces for Part A: Initial Condition:  ψ(x, y, z) = cos(x)*cos(y)*cos(z)
for i in range(len(t_eval)):
    image = (np.abs(np.fft.ifftn(sol1.y.T[i,:].reshape(n,n,n, order = 'F'))))**2
    fig = go.Figure(data=go.Isosurface(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), value=image.flatten(), 
                                       opacity=0.5, isomin=0.25, isomax=1, surface_count=20))
    fig.update_layout(title_text=f"t={t_eval[i]}")
    fig.write_image(f"cos{t_eval[i]}.png")

# Plot Isosurfaces for Part B: Initial Condition:  ψ(x, y, z) = sin(x)*sin(y)*sin(z)
for i in range(len(t_eval)):
    image = (np.abs(np.fft.ifftn(sol2.y.T[i,:].reshape(n,n,n, order = 'F'))))**2
    fig = go.Figure(data=go.Isosurface(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), value=image.flatten(), 
                                       opacity=0.5, isomin=0.25, isomax=1, surface_count=20))
    fig.update_layout(title_text=f"t={t_eval[i]}")
    fig.write_image(f"sin{t_eval[i]}.png")

