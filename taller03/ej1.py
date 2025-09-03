#Este codigo no documenta todo el contenido visto en el taller y puede contener errores

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem
import ufl
from petsc4py import PETSc

dt = 0.02
T  = 1
t = 0.0
num_steps = int(T/dt)
kappa = 1

nx, ny = 50, 50
domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([-2, -2]), np.array([2, 2])],   [nx, ny], mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))


#boundary_facets = mesh.locate_entities_boundary(domain, domain.topology.dim - 1, lambda x: np.full(x.shape[1], True))
#dofs_boundary=fem.locate_dofs_topological(V, domain.topology.dim - 1, boundary_facets)

def on_boundary(x):
    return np.isclose(x[0], -2) | np.isclose(x[0], 2.0) |  np.isclose(x[1], -2.0) | np.isclose(x[1], 2.0)
dofs_boundary = fem.locate_dofs_geometrical(V, on_boundary)


u_D = fem.Function(V)
##u_D.x.array[:] = 0.0
bc = fem.dirichletbc(u_D, dofs_boundary)


def initial_condition(x, a=5):
    return np.exp(-a * (x[0]**2 + x[1]**2))
u_n = fem.Function(V)
u_n.interpolate(initial_condition)


f = fem.Constant(domain, PETSc.ScalarType(0.0))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = u*v*ufl.dx + dt*kappa*ufl.dot(ufl.grad(u), ufl.grad(v))*ufl.dx
L = (u_n + dt*f)*v*ufl.dx

problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

with io.XDMFFile(domain.comm, "diffusion.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)


t = 0.0
times=[]
for n in range(num_steps):
    t = t+dt
    uh = problem.solve()    
    uh.x.scatter_forward()  # importante en MPI
    uh.name = "u"
    xdmf.write_function(uh, t)
    u_n.x.array[:] = uh.x.array
    times.append(t)
    print(t)









#ds = ufl.Measure("ds", domain=domain, metadata={"quadrature_degree": 4})
#n  = ufl.FacetNormal(domain)

