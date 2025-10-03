#Este codigo no documenta todo el contenido visto en el taller y puede contener errores

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, io, mesh, default_scalar_type
from dolfinx.fem.petsc import LinearProblem

# Crear malla
N = 5
domain = mesh.create_box( MPI.COMM_WORLD, [[0, 0, 0.0], [2, 1, 1]], [2*N, N, N], mesh.CellType.tetrahedron,)
dim = domain.topology.dim


#Definir el espacio de funciones 
V = fem.functionspace(domain, ("Lagrange", 1, (dim,))) #o CG


#Funciones para buscar fronteras
def left(x):
    return np.isclose(x[0], 0.0)
def right(x):
    return np.isclose(x[0],2)  


#CB left
u_left = fem.Function(V)

left_facets = mesh.locate_entities_boundary(domain, domain.topology.dim - 1, left)
left_dofs=fem.locate_dofs_topological(V, domain.topology.dim - 1, left_facets)
bc_left=fem.dirichletbc(u_left, left_dofs) 


#CB right
Vx, _ = V.sub(0).collapse()  # Espacio escalar para componente x
u_right = fem.Function(Vx)
u_right.x.array[:] = default_scalar_type(0.5)

right_facets = mesh.locate_entities_boundary(domain, domain.topology.dim - 1, right)
right_dofs_x=fem.locate_dofs_topological( (V.sub(0),Vx), domain.topology.dim-1,right_facets)
bc_right_x=fem.dirichletbc(u_right,right_dofs_x,V.sub(0))

#bcs
bcs = [bc_left,bc_right_x]
       

# Parametros material
E = 10 
nu = 0.3
mu = fem.Constant(domain, E / 2 / (1 + nu))
lmbda = fem.Constant(domain, E * nu / (1 - 2 * nu) / (1 + nu))

# Elasticity functions
def epsilon(u):
    return ufl.sym(ufl.grad(u))
def sigma(u):
    return lmbda*ufl.nabla_div(u)*ufl.Identity(len(u))+2*mu*epsilon(u) #dim

#Weak form
u=ufl.TrialFunction(V)
v=ufl.TestFunction(V)
f = fem.Constant(domain, np.array([0.0, 0.0, 0], dtype=default_scalar_type))
a=ufl.inner(sigma(u),epsilon(v))*ufl.dx
L=ufl.dot(f,v)*ufl.dx 
problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})


uh = problem.solve()
step=0
with io.XDMFFile(domain.comm, "unideformation.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    uh.name = "u"
    xdmf.write_function(uh,step)

