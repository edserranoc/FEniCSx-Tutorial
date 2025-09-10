
import gmsh
import numpy as np
import time

from dolfinx.io import gmshio
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from dolfinx import fem
import ufl
from dolfinx import default_scalar_type
import dolfinx.io
from pathlib import Path

start_time = time.time()
gmsh.initialize()
membrane = gmsh.model.occ.addDisk(0, 0, 0, 1, 1) # (x, y, z, x-radius, y-radius)
gmsh.model.occ.synchronize()
gdim = 2
gmsh.model.addPhysicalGroup(gdim, [membrane], 1)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.05)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.05)
gmsh.model.mesh.generate(gdim)


gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim)

V = fem.functionspace(domain, ("Lagrange", 1))


x = ufl.SpatialCoordinate(domain)
beta = fem.Constant(domain, default_scalar_type(12))
R0 = fem.Constant(domain, default_scalar_type(0.3))
p = 4 * ufl.exp(-beta**2 * (x[0]**2 + (x[1] - R0)**2))


def on_boundary(x):
    return np.isclose(np.sqrt(x[0]**2 + x[1]**2), 1)
boundary_dofs = fem.locate_dofs_geometrical(V, on_boundary)
bc = fem.dirichletbc(default_scalar_type(0), boundary_dofs, V)


u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = p * v * ufl.dx
problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

Q = fem.functionspace(domain, ("Lagrange", 5))
expr = fem.Expression(p, Q.element.interpolation_points())
pressure = fem.Function(Q)
pressure.interpolate(expr)


final_time = time.time() - start_time
print(f"Time to generate mesh, solve, and plot: {final_time:.2f} seconds")

pressure.name = "Load"
uh.name = "Deflection"
results_folder = Path("results")
results_folder.mkdir(exist_ok=True, parents=True)
with dolfinx.io.VTXWriter(MPI.COMM_WORLD, results_folder / "membrane_pressure.bp", [pressure], engine="BP4") as vtx:
    vtx.write(0.0)
with dolfinx.io.VTXWriter(MPI.COMM_WORLD, results_folder / "membrane_deflection.bp", [uh], engine="BP4") as vtx:
    vtx.write(0.0)