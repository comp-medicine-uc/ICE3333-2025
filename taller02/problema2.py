import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, io, mesh, default_scalar_type
from dolfinx.fem.petsc import LinearProblem

#Este codigo no documenta todo el contenido visto en el taller y puede contener errores


# Crear malla
N = 4
domain = mesh.create_box( MPI.COMM_WORLD, [[0, 0, 0.0], [5, 1, 1]], [3*N, N, N], mesh.CellType.tetrahedron)
dim = domain.topology.dim

#Definir el espacio de funciones 
V = fem.functionspace(domain, ("CG", 1, (dim,))) 

#Buscamos fronteras, cb dirichlet
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

# Elasticity functions
def epsilon(u):
    return ufl.sym(ufl.grad(u))
def sigma(u):
    return lmbda*ufl.nabla_div(u)*ufl.Identity(len(u))+2*mu*epsilon(u) #dim


#Weak form
u=ufl.TrialFunction(V)
v=ufl.TestFunction(V)
f = fem.Constant(domain, np.array([0.0, 0.0, -0.1], dtype=default_scalar_type))
a=ufl.inner(sigma(u),epsilon(v))*ufl.dx
L=ufl.dot(f,v)*ufl.dx 
problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

#file para guardar soluci\'on
out_file = "tdef.xdmf"
with io.XDMFFile(domain.comm, out_file, "w") as xdmf:
    xdmf.write_mesh(domain)

#Solve and save
fs=np.linspace(0,0.1,10)
for i in np.arange(len(fs)):
    f.value=np.array([0,0,-fs[i]],dtype=default_scalar_type)
    uh=problem.solve()
    uh.name="u"
    #uh.x.scatter_forward()
    with io.XDMFFile(domain.comm, out_file, "a") as xdmf:
        xdmf.write_function(uh, i )


