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


def get_stifness(
    E0: float,
    Emin: float,
    nu0: float,
    p,
    rho: np.ndarray,
    opt_target_indices: np.ndarray
) -> Callable:

    @BilinearForm
    def stiffness(u, v, w):
        """build stiffness matrix with 3D/SIMP/Voigt representation"""

        E = np.full(len(w.idx), E0, dtype=np.float64)
        nu = np.full(len(w.idx), nu0, dtype=np.float64)
        target_indices = np.searchsorted(opt_target_indices, w.idx)
        mask = np.isin(w.idx, opt_target_indices)
        rho_map = np.ones(len(w.idx))
        rho_map[mask] = rho[target_indices[mask]]
        E[mask] = Emin + (E0 - Emin) * rho_map[mask]**p
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))

        C = np.zeros((6, 6, len(w.idx)), dtype=np.float64)
        C[0, 0] = C[1, 1] = C[2, 2] = lam + 2 * mu
        C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 0] = C[2, 1] = lam
        C[3, 3] = C[4, 4] = C[5, 5] = mu

        # Voigt Transform
        def voigt_map(u):
            g = u.grad  # shape: (3, 3, npts, nshp)
            return np.stack([
                g[0, 0, :, :],          # ε_xx
                g[1, 1, :, :],          # ε_yy
                g[2, 2, :, :],          # ε_zz
                g[1, 2, :, :] + g[2, 1, :, :],  # 2*ε_yz
                g[2, 0, :, :] + g[0, 2, :, :],  # 2*ε_zx
                g[0, 1, :, :] + g[1, 0, :, :],  # 2*ε_xy
            ]) * 0.5

        eps_u = voigt_map(u)
        eps_v = voigt_map(v)
        # Stress C : ε
        sigma = np.einsum("ijm,jmn->imn", C, eps_u)  # shape: (6, npts, nshp)

        return ddot(sigma, eps_v)

    return stiffness


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


def apply_density_filter_cKDTree(rho, mesh, opt_target, radius=0.1):
    """
    """
    centers = mesh.p[:, mesh.t[:, opt_target]].mean(axis=1).T  # shape: (n_opt_elem, dim)
    tree = cKDTree(centers)
    rho_new = rho.copy()
    for i, center in enumerate(centers):
        neighbor_ids = tree.query_ball_point(center, r=radius)
        neighbor_elements = [opt_target[j] for j in neighbor_ids]

        dists = np.linalg.norm(centers[neighbor_ids] - center, axis=1)
        weights = radius - dists
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
            w = 1.0 / dist
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
