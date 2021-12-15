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
typeODE = "Burgers_poly"

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

# collocation nodes on [0,1], can scale with dt
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

# number of fine SDC sweeps per rank
K = 10

uf = np.zeros(Mf * nxf)

for m in range(0, Mf):
    uf[m*nxf:m*nxf+nxf] = u0f


AEf, AIf = differentialA(L, nu, nxf, typeODE, v)

Qf, QEf, QIf, Sf = spectral_int_matrix(Mf, dt, dtf, tf[rank*Mf:rank*Mf+Mf])


# Compute serial fine SDC
if rank == 0:
    
    uf, _ = fine_sweep(AEf, AIf, dt, dtf, func, Mf, K, nu, nxf, Sf, Qf, QEf, QIf, tf[rank*Mf:rank*Mf+Mf], typeODE, uf, u0f, xf)
    
    # send value on last node to next process
    comm.send(uf[Mf*nxf-nxf:Mf*nxf], dest=rank+1, tag=1)
    
elif rank == N-1:
    # receive initial value from previous rank
    uf_M = comm.recv(source=rank-1, tag=1)
    
    uf, _ = fine_sweep(AEf, AIf, dt, dtf, func, Mf, K, nu, nxf, Sf, Qf, QEf, QIf, tf[rank*Mf:rank*Mf+Mf], typeODE, uf, uf_M, xf)
    
else:
    # receive initial value from previous rank
    uf_M = comm.recv(source=rank-1, tag=1)
    
    uf, _ = fine_sweep(AEf, AIf, dt, dtf, func, Mf, K, nu, nxf, Sf, Qf, QEf, QIf, tf[rank*Mf:rank*Mf+Mf], typeODE, uf, uf_M, xf)
    
    # send value on last node to next process
    comm.send(uf[Mf*nxf-nxf:Mf*nxf], dest=rank+1, tag=1)
    
    
if rank == 0:
    print()
    print('        -- Fine SDC Sweeps --')
    print()
    print('Working with %2i processes...' % size)
    print()
    print('   Equation:', typeODE)
    print('   End time:', T)
    print('   Number of serial fine SDC iterations: %2i' % K)  
    print()
    
    
# exact solution
if typeODE == 'heat':
    u_exactf = ifft(expm(t[rank+1]*AIf).dot(fft(u0f)))
    u_exactc = ifft(expm(t[rank+1]*AIc).dot(fft(u0c)))
    
elif typeODE == 'Burgers' or typeODE == 'Burgers_poly' or typeODE == 'advdif' or typeODE == 'heat_forced':
    u_exactf = exact_sol(func, nu, t[rank+1], xf)
    u_exactc = exact_sol(func, nu, t[rank+1], xc)
    
    
print('   Rank', rank, ": Error -- %12.8e" % (max(abs(uf[Mf*nxf-nxf:Mf*nxf] - u_exactf))))
print()


end_program = time.time() 
if rank == (size-1):
    print('   Run-time of program:', round((end_program - begin_program)/60, 6), 'Minutes')
    print()

    
    

