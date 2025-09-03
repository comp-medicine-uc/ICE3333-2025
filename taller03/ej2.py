
#Este codigo no documenta todo el contenido visto en el taller y puede contener errores

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, io, log
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import ufl
from petsc4py import PETSc


domain = mesh.create_unit_square(MPI.COMM_WORLD, 20, 20, mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))


def on_boundary(x):
    return (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |  np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))

dofs_b = fem.locate_dofs_geometrical(V, on_boundary)
u_D=fem.Function(V)
bc = fem.dirichletbc(u_D, dofs_b)


def f(u):
    return ufl.exp(u)  

u  = fem.Function(V) 
v  = ufl.TestFunction(V)
du = ufl.TrialFunction(V)


       

G = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx - f(u) * v * ufl.dx
#DG = ufl.derivative(G, u, du) 
#DG= ufl.dot(v,du)*ufl.dx #+... 


#problem = NonlinearProblem(G, u, bcs=[bc], J=DG)
problem = NonlinearProblem(G, u, bcs=[bc])
solver  = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "residual"
#solver.rtol = 1e-9
#solver.atol = 1e-10
solver.max_it = 30
solver.report = True
log.set_log_level(log.LogLevel.INFO)
nit, converged = solver.solve(u)
u.x.scatter_forward()  # importante en MPI


with io.XDMFFile(domain.comm, "NLpoisson.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(u)
