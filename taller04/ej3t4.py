#Este codigo no documenta todo el contenido visto en el taller y puede contener errores

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh, fem, io, default_scalar_type,log
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc

# Crear malla
N = 4
domain = mesh.create_box( MPI.COMM_WORLD, [[0, 0, 0.0], [5, 1, 1]], [5*N, N, N], mesh.CellType.tetrahedron,)
dim = domain.topology.dim

#Definir el espacio de funciones 
V = fem.functionspace(domain, ("Lagrange", 1, (dim,))) 

#Buscamos fronteras
def left(x):
    return np.isclose(x[0], 0.0)
def right(x):
    return np.isclose(x[0],5)

left_dofs = fem.locate_dofs_geometrical(V, left)
right_dofs = fem.locate_dofs_geometrical(V, right)
u_left = fem.Function(V)
u_right=fem.Function(V)
bc_left=fem.dirichletbc(u_left, left_dofs) 
bc_right=fem.dirichletbc(u_right, right_dofs) 
bcs = [bc_left,bc_right]

# Parametros material
E = 10 
nu = 0.3
mu = fem.Constant(domain, E / 2 / (1 + nu))
lmbda = fem.Constant(domain, E * nu / (1 - 2 * nu) / (1 + nu))

# Cinematica y stress
u = fem.Function(V)  
I = ufl.Identity(dim)
F = ufl.variable(I + ufl.grad(u))
I1 = ufl.tr(F.T*F)
J = ufl.det(F)
psi=(mu / 2) * (I1 - 3) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J))**2
P =ufl.diff(psi, F)



#Weak form
#du=TrialFunction(V)
v=ufl.TestFunction(V)
f = fem.Constant(domain, np.array([0.0, 0.0, -0.1], dtype=default_scalar_type))
G = ufl.inner(P, ufl.grad(v)) * ufl.dx - ufl.dot(f, v) * ufl.dx 
#DG=  # Tambien se puede implementar DG de manera explicita y entregarselo a newton 
#Nota: En problemas no lineales, es comun es resolver de manera incremental, aumentando la carga.
#for i in loads...
#         solve...
#Por simplicidad, y como no tenemos problemas de convergencia resolvemos con toda la carga aplicada.
# Problema no lineal y solver
problem = NonlinearProblem(G, u, bcs)
solver = NewtonSolver(domain.comm, problem)
#solver.convergence_criterion = "residual"
#solver.report = True
#log.set_log_level(log.LogLevel.INFO)
# Resolver
n, converged = solver.solve(u)


# Exportar
with io.XDMFFile(domain.comm, "deformation_hyperelastic.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    u.name = "u"
    xdmf.write_function(u)

