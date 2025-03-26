from typing import Callable
from collections import defaultdict

import scipy
import numpy as np
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

import skfem
from skfem import MeshTet, Basis, ElementTetP1, asm
from skfem.helpers import ddot, sym_grad, eye, trace, eye
from skfem.models.elasticity import lame_parameters
from skfem.models.elasticity import linear_elasticity
from skfem.assembly import BilinearForm
from skfem.assembly import LinearForm
from skfem import asm, Basis
from skfem.models.poisson import laplace
from scipy.sparse.linalg import spsolve
from skfem import BilinearForm
from skfem.assembly import BilinearForm

from numba import njit

import numpy as np
from skfem import BilinearForm, asm, Basis
from skfem.helpers import sym_grad, ddot, trace


def assemble_stiffness_matrix(
    basis: Basis,
    rho: np.ndarray,
    E0: float, Emin: float, p: float, nu: float
):
    """
    Assemble the global stiffness matrix for 3D linear elasticity with SIMP material interpolation.
    
    Parameters:
        basis : skfem Basis for the mesh (built with ElementVector(ElementTetP1) on MeshTet).
        rho   : 1D array of length n_elements with density values for each element.
        E0    : Young's modulus of solid material (for rho = 1).
        Emin  : Minimum Young's modulus for void material (for rho = 0, ensures numerical stability).
        p     : Penalization power for SIMP (typically >= 1, e.g., 3 for standard topology optimization).
        nu    : Poisson's ratio (assumed constant for all elements).
    
    Returns:
        Sparse stiffness matrix (scipy.sparse.csr_matrix) assembled for the given density distribution.
    """
    # 1. Compute Young's modulus for each element using SIMP
    E_elem = Emin + (E0 - Emin) * (rho ** p)  # array of size [n_elements]
    
    # 2. Compute Lamé parameters for each element
    lam = (nu * E_elem) / ((1.0 + nu) * (1.0 - 2.0 * nu))   # first Lamé parameter λ_e per element
    mu  = E_elem / (2.0 * (1.0 + nu))                      # second Lamé parameter (shear modulus) μ_e per element
    
    # Reshape to allow broadcasting over integration points (each as [n_elem, 1] column vectors)
    lam = lam.reshape(-1, 1)
    mu  = mu.reshape(-1, 1)
    
    # 3. Define the bilinear form for elasticity (integrand of stiffness entries)
    @BilinearForm
    def stiffness_form(u, v, w):
        # sym_grad(u) is the strain tensor ε(u) at integration points
        # trace(sym_grad(u)) is the volumetric strain (divergence of u)
        # ddot(A, B) computes the double-dot (Frobenius) product of two matrices A and B
        strain_u = sym_grad(u)
        strain_v = sym_grad(v)
        # Apply Lamé parameters for each element (w corresponds to integration context)
        # lam and mu are arrays of shape [n_elem, 1], broadcasting to [n_elem, n_quad] with strain arrays
        term_volumetric = lam * trace(strain_u) * trace(strain_v)      # λ * tr(ε(u)) * tr(ε(v))
        term_dev = 2.0 * mu * ddot(strain_u, strain_v)                 # 2μ * (ε(u) : ε(v))
        return term_volumetric + term_dev  # integrand for stiffness
    
    # 4. Assemble the stiffness matrix using the basis
    K = asm(stiffness_form, basis)
    return K



def apply_density_filter(
    rho, mesh, radius
):
    """
    """
    from numpy.linalg import norm
    centers = mesh.p[:, mesh.t].mean(axis=1).T
    rho_new = np.zeros_like(rho)
    weight_sum = np.zeros_like(rho)

    for i in range(len(rho)):
        for j in range(len(rho)):
            dist = norm(centers[i] - centers[j])
            if dist < radius:
                w = radius - dist
                rho_new[i] += w * rho[j]
                weight_sum[i] += w

    return rho_new / (weight_sum + 1e-8)


def apply_density_filter_cKDTree(
    rho, mesh, opt_target, radius=0.1
):
    """
    """
    centers = mesh.p[:, mesh.t[:, opt_target]].mean(axis=1).T  # shape: (n_opt_elem, dim)
    tree = cKDTree(centers)
    rho_new = rho.copy()
    for i, center in enumerate(centers):
        neighbor_ids = tree.query_ball_point(center, r=radius)
        neighbor_elements = [opt_target[j] for j in neighbor_ids]

        dists = np.linalg.norm(centers[neighbor_ids] - center, axis=1)
        weights = (radius - dists)
        rho_new[opt_target[i]] = np.sum(weights * rho[neighbor_elements]) / (np.sum(weights) + 1e-8)

    return rho_new


def element_to_element_laplacian_tet(mesh, radius):
    from collections import defaultdict
    from scipy.sparse import coo_matrix

    n_elements = mesh.t.shape[1]
    volumes = np.zeros(n_elements)

    face_to_elements = defaultdict(list)
    for i in range(n_elements):
        tet = mesh.t[:, i]
        faces = [
            tuple(sorted([tet[0], tet[1], tet[2]])),
            tuple(sorted([tet[0], tet[1], tet[3]])),
            tuple(sorted([tet[0], tet[2], tet[3]])),
            tuple(sorted([tet[1], tet[2], tet[3]])),
        ]
        for face in faces:
            face_to_elements[face].append(i)

        coords = mesh.p[:, tet]
        a = coords[:, 1] - coords[:, 0]
        b = coords[:, 2] - coords[:, 0]
        c = coords[:, 3] - coords[:, 0]
        volumes[i] = abs(np.dot(a, np.cross(b, c))) / 6.0

    adjacency = defaultdict(list)
    for face, elems in face_to_elements.items():
        if len(elems) == 2:
            i, j = elems
            adjacency[i].append(j)
            adjacency[j].append(i)

    element_centers = np.mean(mesh.p[:, mesh.t], axis=1).T

    rows = []
    cols = []
    data = []
    for i in range(n_elements):
        diag = 0.0
        for j in adjacency[i]:
            dist = np.linalg.norm(element_centers[i] - element_centers[j])
            if dist < 1e-12:
                continue
            # w = 1.0 / (dist + 1e-5)
            w = np.exp(-dist**2 / (2 * radius**2)) 
            rows.append(i)
            cols.append(j)
            data.append(-w)
            diag += w
        rows.append(i)
        cols.append(i)
        data.append(diag)

    laplacian = coo_matrix((data, (rows, cols)), shape=(n_elements, n_elements)).tocsc()
    return laplacian, volumes


def helmholtz_filter_element_based_tet(rho_element: np.ndarray, basis: Basis, radius: float) -> np.ndarray:
    """
    """
    mesh = basis.mesh
    laplacian, mass_element = element_to_element_laplacian_tet(mesh, radius)

    A = laplacian + (1.0 / radius**2) * csc_matrix(np.diag(mass_element))
    rhs = (1.0 / radius**2) * mass_element * rho_element

    rho_filtered_element = spsolve(A, rhs)
    return rho_filtered_element


@njit
def compute_strain_energy_1(u, K, element_dofs):
    n_elem = element_dofs.shape[1]
    dof_per_elem = element_dofs.shape[0]
    strain_energy = np.zeros(n_elem)

    for j in range(n_elem):
        dofs = element_dofs[:, j]
        u_e = np.zeros(dof_per_elem)
        K_e = np.zeros((dof_per_elem, dof_per_elem))

        for i in range(dof_per_elem):
            u_e[i] = u[dofs[i]]
            for k in range(dof_per_elem):
                K_e[i, k] = K[dofs[i], dofs[k]]

        strain_energy[j] = u_e @ (K_e @ u_e)
    return strain_energy


@njit
def compute_strain_energy_2(u, K_data, K_rows, K_cols, element_dofs):
    n_elements = element_dofs.shape[1]
    energy = np.zeros(n_elements)
    for j in range(n_elements):
        dofs = element_dofs[:, j]
        Ke = extract_local_K(K_data, K_rows, K_cols, dofs)
        ue = u[dofs]
        energy[j] = ue @ Ke @ ue
    return energy


@njit
def extract_local_K(K_data, K_rows, K_cols, dofs):
    n = len(dofs)
    Ke = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dof_i = dofs[i]
            dof_j = dofs[j]
            for k in range(len(K_data)):
                if K_rows[k] == dof_i and K_cols[k] == dof_j:
                    Ke[i, j] = K_data[k]
                    break
    return Ke


if __name__ == '__main__':
    
    from topoptpy import problem

    prb = problem.toy2()
    rho = np.ones(prb.all_elements.shape)

    K1 = assemble_stiffness_matrix(
        prb.basis, rho, prb.E0, 0.0, 1.0, prb.nu0
    )
    
    lam, mu = lame_parameters(prb.E0, prb.nu0)
    def C(T):
        return 2. * mu * T + lam * eye(trace(T), T.shape[0])

    @skfem.BilinearForm
    def stiffness(u, v, w):
        return ddot(C(sym_grad(u)), sym_grad(v))

    _F = prb.F
    K2 = stiffness.assemble(prb.basis)
    
    K1_e, F1_e = skfem.enforce(K1, _F, D=prb.dirichlet_nodes)
    K2_e, F2_e = skfem.enforce(K2, _F, D=prb.dirichlet_nodes)

    U1_e = scipy.sparse.linalg.spsolve(K1_e, F1_e)
    U2_e = scipy.sparse.linalg.spsolve(K2_e, F2_e)

    print("U1_e:", np.average(U1_e))
    print("U1_e:", np.average(U2_e))
    
    sf = 1.0
    m1 = prb.mesh.translated(sf * U1_e[prb.basis.nodal_dofs])
    m1.save('K1.vtk')
    m2 = prb.mesh.translated(sf * U2_e[prb.basis.nodal_dofs])
    m2.save('K2.vtk')
