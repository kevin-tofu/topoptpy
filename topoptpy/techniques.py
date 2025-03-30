from typing import Callable
from collections import defaultdict

import scipy
import numpy as np
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

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


def simp_interpolation(rho, E0, Emin, p):
    E_elem = Emin + (E0 - Emin) * (rho ** p)
    return E_elem


def ram_interpolation(rho, E0, Emin, p):
    """
    ram: E(rho) = Emin + (E0 - Emin) * [rho / (1 + p(1 - rho))]
    Parameters:
      rho  : array of densities in [0,1]
      E0   : maximum Young's modulus
      Emin : minimum Young's modulus
      p    : ram parameter
    Returns:
      array of element-wise Young's moduli
    """
    # avoid division by zero
    E_elem = Emin + (E0 - Emin) * (rho / (1.0 + p*(1.0 - rho)))
    return E_elem


def assemble_stiffness_matrix(
    basis: Basis,
    rho: np.ndarray,
    E0: float, Emin: float, p: float, nu: float,
    elem_func: Callable=simp_interpolation
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
    E_elem = elem_func(rho, E0, Emin, p)  # array of size [n_elements]
    
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


def assemble_stiffness_matrix_simp(
    basis: Basis,
    rho: np.ndarray,
    E0: float, Emin: float, p: float, nu: float
):
    return assemble_stiffness_matrix(
        basis,
        rho,
        E0, Emin, p, nu,
        elem_func=simp_interpolation
    )


def assemble_stiffness_matrix_ramp(
    basis: Basis,
    rho: np.ndarray,
    E0: float, Emin: float, p: float, nu: float
):
    return assemble_stiffness_matrix(
        basis,
        rho,
        E0, Emin, p, nu,
        elem_func=ram_interpolation
    )


def dE_drho_simp(rho, E0, Emin, p):
    """
    Derivative of SIMP-style E(rho)
    E(rho) = Emin + (E0 - Emin) * rho^p
    """
    return p * (E0 - Emin) * np.maximum(rho, 1e-6) ** (p - 1)


def dC_drho_simp(rho, strain_energy, E0, Emin, p):
    """
    dC/drho = -p * (E0 - Emin) * rho^(p - 1) * strain_energy
    """
    # dE_drho = p * (E0 - Emin) * np.maximum(rho, 1e-6) ** (p - 1)
    dE_drho = dE_drho_simp(rho, E0, Emin, p)
    return - dE_drho * strain_energy


# def dE_drho_rationalSIMP(rho, E0, Emin, p):
def dE_drho_ramp(rho, E0, Emin, p):
    """
    Derivative of Rational SIMP-style E(rho)
    E(rho) = Emin + (E0 - Emin) * rho / (1 + p * (1 - rho))
    """
    denom = 1.0 + p * (1.0 - rho)
    return (E0 - Emin) * (denom - p * rho) / (denom ** 2)


def dC_drho_ramp(rho, strain_energy, E0, Emin, p):
    dE_drho = dE_drho_ramp(rho, E0, Emin, p)
    return - dE_drho * strain_energy


def heaviside_projection(rho, beta, eta=0.5):
    """
    Smooth Heaviside projection.
    Returns projected density in [0,1].
    """
    numerator = np.tanh(beta * eta) + np.tanh(beta * (rho - eta))
    denominator = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    return numerator / (denominator + 1e-12)


def heaviside_projection_derivative(rho, beta, eta=0.5):
    """
    Derivative of the smooth Heaviside projection w.r.t. rho.
    d/d(rho)[heaviside_projection(rho)].
    """
    # y = tanh( beta*(rho-eta) )
    # dy/d(rho) = beta*sech^2( ... )
    # factor from normalization
    numerator = beta * (1.0 - np.tanh(beta*(rho - eta))**2)
    denominator = np.tanh(beta*eta) + np.tanh(beta*(1.0 - eta))
    return numerator / (denominator + 1e-12)


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


def compute_cell_centroids(mesh):
    """
    """
    node_coords = mesh.p
    t = mesh.t
    centroids = np.mean(node_coords[:, t], axis=1).T
    return centroids


def soft_connectivity_penalty(mesh, rho, threshold=0.3):
    centroids = compute_cell_centroids(mesh)
    weights = np.clip((rho - threshold) / (1.0 - threshold), 0.0, 1.0)

    if np.sum(weights) < 1e-8:
        return 0.0

    weighted_mean = np.sum(centroids * weights[:, None], axis=0) / np.sum(weights)

    sq_dists = np.sum((centroids - weighted_mean)**2, axis=1)
    weighted_var = np.sum(sq_dists * weights) / np.sum(weights)

    return weighted_var


def adjacency_matrix(mesh: skfem.MeshTet):
    n_elements = mesh.t.shape[1]
    face_to_elements = defaultdict(list)

    # 各要素の4面を抽出（sortedで順序揃える）
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

    adjacency = [[] for _ in range(n_elements)]
    for elems in face_to_elements.values():
        if len(elems) == 2:
            i, j = elems
            adjacency[i].append(j)
            adjacency[j].append(i)

    return adjacency


def adjacency_matrix(mesh):
    n_elements = mesh.t.shape[1]
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


    adjacency = defaultdict(list)
    for face, elems in face_to_elements.items():
        if len(elems) == 2:
            i, j = elems
            adjacency[i].append(j)
            adjacency[j].append(i)

    return adjacency


def adjacency_matrix_volume(mesh):
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
    return (adjacency, volumes)


def element_to_element_laplacian_tet(mesh, radius):
    from collections import defaultdict
    from scipy.sparse import coo_matrix

    adjacency, volumes = adjacency_matrix_volume(mesh)
    n_elements = mesh.t.shape[1]
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
    laplacian, volumes = element_to_element_laplacian_tet(mesh, radius)
    volumes_normalized = volumes / np.mean(volumes)

    M = csc_matrix(np.diag(volumes_normalized))
    A = M + radius**2 * laplacian
    rhs = M @ rho_element

    rho_filtered = spsolve(A, rhs)
    return rho_filtered



def compute_filter_gradient_matrix(basis: Basis, radius: float):
    """
    Compute the Jacobian of the Helmholtz filter: d(rho_filtered)/d(rho)
    """
    mesh = basis.mesh
    laplacian, volumes = element_to_element_laplacian_tet(mesh, radius)
    volumes_normalized = volumes / np.mean(volumes)

    M = csc_matrix(np.diag(volumes_normalized))
    A = M + radius**2 * laplacian

    # Solve: d(rho_filtered)/d(rho) = A^{-1} * M
    # You can precompute LU for efficiency
    A_solver = splu(A)

    def filter_grad_vec(v: np.ndarray) -> np.ndarray:
        """Applies Jacobian to vector v"""
        return A_solver.solve(M @ v)

    def filter_jacobian_matrix() -> np.ndarray:
        """Returns the full Jacobian matrix: A^{-1} @ M"""
        n = M.shape[0]
        I = np.eye(n)
        return np.column_stack([filter_grad_vec(I[:, i]) for i in range(n)])

    return filter_grad_vec, filter_jacobian_matrix



def compute_strain_energy(
    u,
    element_dofs,
    basis,
    rho,
    E0,
    Emin, penal, nu0
):
    """Compute element-wise strain energy for a 3D tetrahedral mesh using SIMP material interpolation."""
    mesh = basis.mesh
    # Material constants for elasticity matrix
    lam_factor = lambda E: E / ((1.0 + nu0) * (1.0 - 2.0 * nu0))  # common factor for isotropic C
    mu_factor  = lambda E: E / (2.0 * (1.0 + nu0))               # shear modulus μ

    n_elems = element_dofs.shape[1]  # number of elements (columns of element_dofs)
    energies = np.zeros(n_elems)
    # Precompute base elasticity matrix for E0 (could also compute fresh each time scaled by E_e)
    C0 = lam_factor(E0) * np.array([
        [1 - nu0,    nu0,       nu0,       0,                   0,                   0                  ],
        [nu0,        1 - nu0,   nu0,       0,                   0,                   0                  ],
        [nu0,        nu0,       1 - nu0,   0,                   0,                   0                  ],
        [0,          0,         0,         (1 - 2*nu0) / 2.0,   0,                   0                  ],
        [0,          0,         0,         0,                   (1 - 2*nu0) / 2.0,   0                  ],
        [0,          0,         0,         0,                   0,                   (1 - 2*nu0) / 2.0 ]
    ])
    # Loop over each element in the design domain
    for idx in range(n_elems):
        # Global DOF indices for this element and extract their coordinates
        edofs = element_dofs[:, idx]                  # 12 DOF indices (3 per node for 4 nodes)
        # Infer the 4 node indices (each node has 3 DOFs). We assume DOFs are grouped by node.
        node_ids = [int(edofs[3*j] // 3) for j in range(4)]
        # Coordinates of the 4 nodes (3x4 matrix)
        coords = mesh.p[:, node_ids]
        # Build matrix M for shape function coefficient solve
        # Each row: [x_i, y_i, z_i, 1] for node i
        M = np.column_stack((coords.T, np.ones(4)))
        Minv = np.linalg.inv(M)
        # Gradients of shape functions (each column i gives grad(N_i) = [dN_i/dx, dN_i/dy, dN_i/dz])
        grads = Minv[:3, :]  # 3x4 matrix of gradients
        # Construct B matrix (6x12) for this element
        B = np.zeros((6, 12))
        for j in range(4):
            dNdx, dNdy, dNdz = grads[0, j], grads[1, j], grads[2, j]
            # Fill B for this node j
            B[0, 3*j    ] = dNdx
            B[1, 3*j + 1] = dNdy
            B[2, 3*j + 2] = dNdz
            B[3, 3*j    ] = dNdy
            B[3, 3*j + 1] = dNdx
            B[4, 3*j + 1] = dNdz
            B[4, 3*j + 2] = dNdy
            B[5, 3*j + 2] = dNdx
            B[5, 3*j    ] = dNdz
        # Compute volume of the tetrahedron (abs(det(M))/6)
        vol = abs(np.linalg.det(M)) / 6.0
        # Young's modulus for this element via SIMP
        E_eff = Emin + (rho[idx] ** penal) * (E0 - Emin)
        # Form elasticity matrix C_e (scale base matrix by E_eff/E0 since ν constant)
        C_e = C0 * (E_eff / E0)
        # Element nodal displacements
        u_e = u[edofs]
        # Compute strain = B * u_e
        strain = B.dot(u_e)
        # Strain energy density = 0.5 * strain^T * C_e * strain
        Ue = 0.5 * strain.dot(C_e.dot(strain)) * vol
        energies[idx] = Ue
    return energies


def compute_strain_energy_sparse(u, K: scipy.sparse.csr_matrix, element_dofs):
    n_elem = element_dofs.shape[1]
    strain_energy = np.empty(n_elem)

    for j in range(n_elem):
        dofs = element_dofs[:, j]
        u_e = u[dofs]
        K_e = K[dofs[:, None], dofs]

        strain_energy[j] = u_e @ (K_e @ u_e)
    return strain_energy


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
    print("U2_e:", np.average(U2_e))
    
    sf = 1.0
    m1 = prb.mesh.translated(sf * U1_e[prb.basis.nodal_dofs])
    m1.save('K1.vtk')
    m2 = prb.mesh.translated(sf * U2_e[prb.basis.nodal_dofs])
    m2.save('K2.vtk')


    # 
    K1_e, F1_e = skfem.enforce(K1, _F, D=prb.dirichlet_nodes)
    K1_e_np = K1_e.toarray()
    U1_e = scipy.sparse.linalg.spsolve(K1_e, F1_e)
    u = U1_e
    K = K1_e.toarray()
    U_global = 0.5 * u @ (K @ u)
    print("Global:", U_global)

    # 
    print(prb.basis.element_dofs.shape, rho.shape)
    U_elementwise1 = compute_strain_energy(
        u, prb.basis.element_dofs,
        prb.basis,
        rho,
        prb.E0,
        prb.Emin,
        1.0,
        prb.nu0,
    ).sum()
    
    element_dofs = prb.basis.element_dofs[:, prb.design_elements]
    rho_design = rho[prb.design_elements]
    print(element_dofs.shape, rho_design.shape)
    U_elementwise2 = compute_strain_energy(
        u, element_dofs,
        prb.basis,
        rho_design,
        prb.E0,
        prb.Emin,
        1.0,
        prb.nu0,
    ).sum()
    
    element_dofs = prb.basis.element_dofs[:, prb.free_nodes]
    rho_design = rho[prb.free_nodes]
    print(element_dofs.shape, rho_design.shape)
    U_elementwise3 = compute_strain_energy(
        u, element_dofs,
        prb.basis,
        rho_design,
        prb.E0,
        prb.Emin,
        1.0,
        prb.nu0,
    ).sum()
    print("Sum over elements all:", U_elementwise1)
    print("Sum over elements design:", U_elementwise2)
    print("Sum over elements design:", U_elementwise3)
