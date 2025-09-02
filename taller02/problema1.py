import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem
import ufl

#Este codigo  no documenta todo el contenido visto en el taller y puede contener errores

# Crear malla
domain = mesh.create_unit_square(MPI.COMM_WORLD, 20, 20, mesh.CellType.triangle)  # triangulos

#Espacio de funciones
V = fem.functionspace(domain, ("CG", 1)) 

#CB Dirichlet
#boundary_facets = mesh.locate_entities_boundary(domain, domain.topology.dim - 1, lambda x: np.full(x.shape[1], True))
#dofs_boundary=fem.locate_dofs_topological(V, domain.topology.dim - 1, boundary_facets)

def on_boundary(x):
    return np.isclose(x[0], -2) | np.isclose(x[0], 2.0) |  np.isclose(x[1], -2.0) | np.isclose(x[1], 2.0)
dofs_boundary_all = fem.locate_dofs_geometrical(V, on_boundary)



x = ufl.SpatialCoordinate(domain)

u_ast = x[0]*(1 - x[0]) * x[1]*(1 - x[1])
f_ufl = -ufl.div(ufl.grad(u_ast))  #2*(x[0]*(1 - x[0]) + x[1]*(1 - x[1]))


u_D = fem.Function(V)
u_D.interpolate(fem.Expression(u_ast, V.element.interpolation_points()))
bc = fem.dirichletbc(u_D, dofs_boundary_all)


# Weak form
u  = ufl.TrialFunction(V)
v  = ufl.TestFunction(V)
f = f_ufl
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx


#Solve
problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()
uh.name = "u"

#Save
with io.XDMFFile(domain.comm, "poisson.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh, 0.0)

#Post-proceso
#Norma del error L2
er = fem.assemble_scalar(fem.form(ufl.inner(uh - u_ast, uh - u_ast) * ufl.dx))
errorL2 = np.sqrt(domain.comm.allreduce(er, op=MPI.SUM))


#Proyeccion L2 (completar)
#dim = domain.topology.dim
#W = # crear espacio de funciones 
#w = ufl.TrialFunction(W)
#z = ufl.TestFunction(W)
#aproy = ufl.inner(w, z) * ufl.dx                 
#Lproy = ufl.inner(ufl.grad(uh), z) * ufl.dx      
#proj_grad=LinearProblem(...) # completar
#proj_grad=proj_grad.solve()

