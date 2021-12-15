import numpy as np
from numpy.fft import fft, ifft
from mpi4py import MPI
from ic import ic
from exact_sol import exact_sol
from Euler import expEuler, impEuler
from pfasst2 import pfasst
from int_nodes import int_nodes
from differential_operators import differentialA
from resolved_run import single_SDC
from spectral_int_matrix import spectral_int_matrix
from sweep import fine_sweep
#import matplotlib
#import matplotlib.pyplot as plt
#matplotlib.use('TkAgg') 
from scipy.linalg import expm
import time
import warnings
warnings.filterwarnings('ignore')


begin_program = time.time()

# Initialization of processes
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# number of ranks/processors
N = size

# number of intermediate points (fine and coarse level)
Mf = 5
Mc = 3

# Choose an ODE which shall be solved ('heat', 'heat_forced', 'Burgers' or 'advdif')
typeODE = "Burgers"

# function for initial condition - typeODE determines function for initial condition and spatial domain
if typeODE == 'heat':
    func = 'sin'
    a = -np.pi
    b = np.pi
    
elif typeODE == 'heat_forced':
    func = 'sin_heat'
    a = -0.5
    b = 0.5
    
elif typeODE == "Burgers":
    func = 'exp'
    a = 0
    b = 1
    
elif typeODE == "Burgers_poly":
    func = 'poly'
    a = 0
    b = 1
    
elif typeODE == 'advdif':
    func = 'sin_advdif'
    a = -0.5
    b = 0.5


# number of grid points are chosen so that nxf, and nxc, respectively, is a power of two
nxf = 512
nxc = 256
dxf = (b-a)/nxf
dxc = (b-a)/nxc
xf = np.linspace(a, b - dxf, num=nxf)
xc = np.zeros(nxc)

# every second point on the spatial fine level represents a point on the spatial coarse level
for k in range(0, nxc):
    xc[k] = xf[2*k]
    
# time interval and subintervals
T = 0.08
dt = T/N

t = dt * np.arange(0, N + 1)
t_solve = np.zeros(Mf)
nt = np.shape(t)[0]

tf = np.zeros(N * Mf)
tc = np.zeros(N * Mc)

ntf = np.shape(tf)[0]
ntc = np.shape(tc)[0]

# fine and coarse collocation nodes
for l in range(0, nt-1):
    tf[Mf*l:Mf*l + Mf] = int_nodes(t[l], t[l+1], Mf)

for l in range(0, nt-1):
    tc[Mc*l:Mc*l + Mc] = int_nodes(t[l], t[l+1], Mc)

dtf = np.zeros(Mf)
dtc = np.zeros(Mc)

ndtf = np.shape(dtf)[0]
ndtc = np.shape(dtc)[0]

# determination of the fine and coarse time steps
for j in range(0, Mf-1):
    dtf[j] = tf[j+1]-tf[j]

for j in range(0, Mc-1):
    dtc[j] = tc[j+1] - tc[j]

    
dtf2 = dtf[0:Mf-1]
dtc2 = dtc[0:Mc-1]

# diffusion coefficient and advection speed
nu = 0.005
v = 1

# initial condition - choose between "exp", "sin" or poly"
u0f, L = ic(xf, func, nu)
u0c, L = ic(xc, func, nu)

# number of coarse SDC sweeps per PFASST iteration
nG = 1

# number of PFASST iterations
K = 6

# Prediction to find a better initial condition
prediction_on = True

if rank == 0:
    print()
    print('        -- PFASST --')
    print()
    print('Working with %2i processes...' % size)
    print()
    print('   Equation:', typeODE)
    print('   End time:', T)
    print('   Number of coarse SDC sweeps: %2i' % nG)
    print('   Number of PFASST iterations: %2i' % K)
    if K+1 > 0:
        print('   Number of serial fine SDC iterations: %2i' % K) 
    print()

# PFASST output
if typeODE == 'heat' or typeODE == 'heat_forced' or typeODE == 'Burgers' or typeODE == 'advdif' or typeODE == 'Burgers_poly':
    AIf, AIc, res, uf_M, uc_M = pfasst(comm, dt, dtc2, dtf2, func, K, L, nG, nxc, nxf, nu, Mc, Mf, prediction_on, rank, size, T, tc, tf, typeODE, u0c, u0f, v, xc, xf)
    
# exact solution
if typeODE == 'heat':
    u_exactf = ifft(expm(t[rank+1]*AIf).dot(fft(u0f)))
    u_exactc = ifft(expm(t[rank+1]*AIc).dot(fft(u0c)))
    
elif typeODE == 'Burgers' or typeODE == 'Burgers_poly' or typeODE == 'advdif' or typeODE == 'heat_forced':
    u_exactf = exact_sol(func, nu, t[rank+1], xf)
    u_exactc = exact_sol(func, nu, t[rank+1], xc)
        

# Lmax error of the different processes
if K > 0:
    print('   Rank', rank, ": Error -- %12.8e" % (max(abs(uf_M - u_exactf))), "|| Residual -- %12.8e" % (res[-1]))
    print()
    
else:
    print('   Rank', rank, ": Error -- %12.8e" % (max(abs(uf_M - u_exactf))))
    print()
    
end_program = time.time() 
if rank == (size-1):
    print('   Run-time of program:', round((end_program - begin_program)/60, 6), 'Minutes')
    print()

# Plot
#if rank == (size-1):
    #plt.subplot(121)
    #plt.title("{} on coarse level at t={}".format(typeODE, T))
    #plt.plot(xc, uc_M.real, label='Numerical solution')
    #plt.plot(xc, u_exactc, label='Exact solution')
    #plt.legend(loc='upper right')
    
    #plt.subplot(122)
    #plt.title("{} on fine level at t={}".format(typeODE, T))
    #plt.plot(xf, u0f, label='Initial condition')
    #plt.plot(xf, uf_M, label='Numerical solution')
    #plt.plot(xf, u_exactf, label='Exact solution')
    #plt.legend(loc='upper right')
    
    #plt.show()
    
    #plt.title('Residual')
    #plt.plot(np.arange(0, K), res, 'm-*')
    #plt.xlabel('PFASST iterations')
    #plt.show()
    

# Plot only for last rank (corresponds to last subinterval in time domain)
#if rank == (size-1):
#    if func == "exp":
#        if solveSDC == "Imp":
#            plt.subplot(121)
#            plt.title("Heat equation on coarse level")
#            plt.plot(xc, uc_M.real, 'o', label="Numerical solution at t={}".format(t[rank+1]))
            #plt.plot(xc, u0c.real, label="Initial condition")
#            plt.plot(xc, ifft(expm(t[rank+1]*AIc).dot(u0hatc)).real, label="Exact solution")
#            plt.legend(loc="upper right")
       
#            plt.subplot(122)
#            plt.title("Heat equation on fine level")
#            plt.plot(xf, uf_M.real, 'o', label="Numerical solution at t={}".format(t[rank+1]))
            #plt.plot(xf, u0f.real, label="Initial condition")
#            plt.plot(xf, ifft(expm(t[rank+1]*AIf).dot(u0hatf)).real, label="Exact solution")
#            plt.legend(loc="upper right")
        
#            plt.show()
#        elif solveSDC == "IMEX":
#            plt.subplot(121)
#            plt.title("Burgers' equation on coarse level")
#            plt.plot(xc, uc_M.real, 'o', label="Numerical solution at t={}".format(t[rank+1]))
#            #plt.plot(xc, u0c.real, label="Initial condition")
            #plt.plot(xc, u_exactc.real, label="Exact solution")
#            plt.legend(loc="upper right")
        
#            plt.subplot(122)
#            plt.title("Burgers' equation on fine level")
#            plt.plot(xf, uf_M.real, 'o', label="Numerical solution at t={}".format(t[rank+1]))
            #plt.plot(xf, u0f.real, label="Initial condition")
#            plt.plot(xf, u_exactf.real, label="Exact solution")
#            #plt.plot(xf, ifft(uhat_solveM), label="Solution of resolved run")
#            plt.legend(loc="upper right")
        
#            plt.show()
#        elif solveSDC == "expEuler" or solveSDC == "impEuler":
            #plt.subplot(121)
#            plt.title("Explicit Euler for Burgers' equation on coarse level")
#            plt.plot(xc, uc_M.real, 'o', label="Numerical solution at t={}".format(t[rank+1]))
            #plt.plot(xc, u0c.real, label="Initial condition")
#            plt.plot(xc, u_exactc, label="Exact solution")
#            plt.legend(loc="upper right")
        
            #plt.subplot(122)
            #plt.title("Implicit Euler for Burgers' equation on fine level")
            #plt.plot(xf, uf_M.real, 'o', label="Numerical solution at t={}".format(t[rank+1]))
            #plt.plot(xf, u0f.real, label="Initial condition")
            #plt.plot(xf, u_exactf, label="Exact solution")
            #plt.legend(loc="upper right")
        
#            plt.show()

#    elif func == "sin":
#        if solveSDC == "Imp":
#            plt.subplot(121)
#            plt.title("Heat equation on coarse level")
#            plt.plot(xc, uc_M.real, 'o', label="Numerical solution at t={}".format(T))
#            plt.plot(xc, u0c.real, label="Initial condition")
#            plt.plot(xc, np.exp(-nu*T)*np.sin(xc), label="Exact solution")
#            plt.legend(loc="upper right")
        
#            plt.subplot(122)
#            plt.title("Heat equation on fine level")
#            plt.plot(xf, uf_M.real, 'o', label="Numerical solution at t={}".format(T))
#            plt.plot(xf, u0f.real, label="Initial condition")
#            plt.plot(xf, np.exp(-nu*T)*np.sin(xf), label="Exact solution")
#            plt.legend(loc="upper right")
        
#            plt.show()
#        elif solveSDC == "IMEX":  
#            plt.subplot(121)
#            plt.title("Burgers' equation on coarse level")
#            plt.plot(xc, uc_M.real, 'o', label="Numerical solution at t={}".format(T))
            #plt.plot(xc, u0c.real, label="Initial condition")
            #plt.plot(xc, u_exactc, label="Exact solution")
#            plt.legend(loc="upper right")
        
#            plt.subplot(122)
#            plt.title("Burgers equation on fine level")
#            plt.plot(xf, uf_M.real, 'o', label="Numerical solution at t={}".format(T))
            #plt.plot(xf, u0f.real, label="Initial condition")
            #plt.plot(xf, u_exactf.real, label="Exact solution")
#            plt.plot(xf, ifft(uhat_solveM), label="Solution of resolved run")
#            plt.legend(loc="upper right")
        
#            plt.show()
#        elif solveSDC == "expEuler" or solveSDC == "impEuler":
#            plt.subplot(121)
#            plt.title("Heat equation on coarse level")
#            plt.plot(xc, uc_M.real, 'o', label="Numerical solution at t={}".format(T))
#            plt.plot(xc, u0c.real, label="Initial condition")
#            plt.plot(xc, np.exp(-nu*T)*np.sin(xc), label="Exact solution")
#            plt.legend(loc="upper right")
        
#            plt.subplot(122)
#            plt.title("Heat equation on fine level")
#            plt.plot(xf, uf_M.real, 'o', label="Numerical solution at t={}".format(T))
#            plt.plot(xf, u0f.real, label="Initial condition")
#            plt.plot(xf, np.exp(-nu*T)*np.sin(xf), label="Exact solution")
#            plt.legend(loc="upper right")
        
#            plt.show()
        
#    elif func == "poly":
#        if solveSDC == "Imp":
#            plt.subplot(121)
#            plt.title("Heat equation on coarse level")
#            plt.plot(xc, uc_M.real, 'o', label="Numerical solution at t={}".format(T))
#            plt.plot(xc, u0c.real, label="Initial condition")
#            plt.plot(xc, ifft(expm(T*AIc).dot(u0hatc)).real, label="Exact solution")
#            plt.legend(loc="upper right")
        
#            plt.subplot(122)
#            plt.title("Heat equation on fine level")
#            plt.plot(xf, uf_M.real, 'o', label="Numerical solution at t={}".format(T))
#            plt.plot(xf, u0f.real, label="Initial condition")
#            plt.plot(xf, ifft(expm(T*AIf).dot(u0hatf)).real, label="Exact solution")
#            plt.legend(loc="upper right")
        
#            plt.show()
#        elif solveSDC == "IMEX":
#            plt.subplot(121)
#            plt.title("Burgers' equation on coarse level")
#            plt.plot(xc, uc_M.real, 'o', label="Numerical solution at coarse level for t={}".format(T))
            #plt.plot(xc, u0c.real, label="Initial condition at coarse level")
            #plt.plot(xc, nu * xc**4 * (1 - xc**4) * np.cos(nu*T), label="Exact solution")
#            plt.legend(loc="upper right")
        
#            plt.subplot(122)
#            plt.title("Burgers' equation on fine level")
#            plt.plot(xf, uf_M.real, 'o', label="Numerical solution at t={}".format(T))
            #plt.plot(xf, u0f.real, label="Initial condition")
            #plt.plot(xf, nu * xf**4 * (1 - xf**4) * np.cos(nu*T), label="Exact solution")
#            plt.plot(xf, ifft(uhat_solveM), label="Solution of resolved run")
#            plt.legend(loc="upper right")
        
#            plt.show()
#        elif solveSDC == "expEuler" or solveSDC == "impEuler":
#            plt.subplot(121)
#            plt.title("Heat equation on coarse level")
#            plt.plot(xc, uc_M.real, 'o', label="Numerical solution at t={}".format(T))
#            plt.plot(xc, u0c.real, label="Initial condition")
#            plt.plot(xc, ifft(expm(T*AIc).dot(u0hatc)).real, label="Exact solution")
#            plt.legend(loc="upper right")
        
#            plt.subplot(122)
#            plt.title("Heat equation on fine level")
#            plt.plot(xf, uf_M.real, 'o', label="Numerical solution at t={}".format(T))
#            plt.plot(xf, u0f.real, label="Initial condition")
#            plt.plot(xf, ifft(expm(T*AIf).dot(u0hatf)).real, label="Exact solution")
#            plt.legend(loc="upper right")
        
#            plt.show()
        
#    else:
#        print("No other plots implemented!")
    
