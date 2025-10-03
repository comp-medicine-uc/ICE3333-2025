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

#Marcaje de caras
facets_left=mesh.locate_entities_boundary(domain,domain.topology.dim-1,left) 
#print(facets_left)
facets_right=mesh.locate_entities_boundary(domain,domain.topology.dim-1,right) 
marked_facets = np.hstack([facets_left, facets_right]) 

values_left=np.full_like(facets_left, 1) 
values_right=np.full_like(facets_right, 2) 
marked_values=np.hstack([values_left,values_right])

sorted_facets=np.argsort(marked_facets) 
facet_tag=mesh.meshtags(domain,domain.topology.dim-1,marked_facets[sorted_facets],marked_values[sorted_facets]) 

#Visualizar en ParaView
domain.topology.create_connectivity(domain.topology.dim-1, domain.topology.dim)
with io.XDMFFile(domain.comm, "facet_tags.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(facet_tag, domain.geometry)


#BC Dirichlet en left
u_left = fem.Function(V)
left_dofs = fem.locate_dofs_topological(V, facet_tag.dim, facet_tag.find(1))
bc_left=fem.dirichletbc(u_left, left_dofs) 
bcs = [bc_left]



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
T = fem.Constant(domain, np.array([0.0, 0.0, -0.1], dtype=default_scalar_type))
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)
L=ufl.dot(f,v)*ufl.dx  + ufl.dot(T,v)*ds(2) 
problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})




uh = problem.solve()
step=0
with io.XDMFFile(domain.comm, "deformation.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    uh.name = "u"
    xdmf.write_function(uh,step)

