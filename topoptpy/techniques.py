from typing import Callable
import numpy as np
import scipy.sparse.linalg as spla
import scipy.sparse as sp
from scipy.spatial import cKDTree
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import skfem
from skfem import MeshTet, Basis, ElementTetP1, asm
from skfem.helpers import dot, grad
from skfem.helpers import ddot, sym_grad, eye, trace
from skfem.helpers import dot, ddot, eye, sym_grad, trace
from skfem.models.elasticity import lame_parameters
from skfem.models.elasticity import linear_elasticity
from skfem.assembly import BilinearForm
from skfem.assembly import LinearForm
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
        """3D・SIMP法・ポアソン比考慮・Voigt表現で剛性行列を構築"""

        # ヤング率・ポアソン比設定
        E = np.full(len(w.idx), E0, dtype=np.float64)
        nu = np.full(len(w.idx), nu0, dtype=np.float64)

        # 最適化対象のみに SIMP 適用
        target_indices = np.searchsorted(opt_target_indices, w.idx)
        mask = np.isin(w.idx, opt_target_indices)
        rho_map = np.ones(len(w.idx))
        rho_map[mask] = rho[target_indices[mask]]
        E[mask] = Emin + (E0 - Emin) * rho_map[mask]**p

        # ラメ定数
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))

        # 弾性テンソル C（6×6）を作る：Voigt表現
        C = np.zeros((6, 6, len(w.idx)), dtype=np.float64)
        C[0, 0] = C[1, 1] = C[2, 2] = lam + 2 * mu
        C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 0] = C[2, 1] = lam
        C[3, 3] = C[4, 4] = C[5, 5] = mu

        # Voigt変換：3D 対称ひずみテンソル → 6成分ベクトル
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

        # 応力 = C : ε
        sigma = np.einsum("ijm,jmn->imn", C, eps_u)  # shape: (6, npts, nshp)

        return ddot(sigma, eps_v)

    return stiffness


from skfem import LinearForm
from skfem.helpers import sym_grad, ddot, trace, eye
import numpy as np


# トポロジー最適化用の LinearForm で strain energy density を計算
def create_strain_energy_form(E0, Emin, nu0, p, rho, opt_target):

    @LinearForm
    def strain_energy_density(v, w):
        
        # print(w)
        # u = w['u']  # 補間された変位ベクトル場
        u = w
        # eps = sym_grad(grad(u))
        eps = sym_grad(u)

        # 要素数取得
        nelems = len(w.idx)

        # ヤング率とポアソン比初期化
        E = np.full(nelems, E0, dtype=np.float64)
        nu = np.full(nelems, nu0, dtype=np.float64)

        # 最適化対象要素にのみ SIMP 適用
        target_indices = np.searchsorted(opt_target, w.idx)
        mask = np.isin(w.idx, opt_target)
        rho_map = np.ones(nelems)
        rho_map[mask] = rho[target_indices[mask]]
        E[mask] = Emin + (E0 - Emin) * rho_map[mask]**p

        # ラメ定数計算
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))

        # 応力テンソル（Voigt でなく通常表現）
        sigma = (lam * trace(eps) * eye(3) + 2 * mu * eps)

        return 0.5 * ddot(sigma, eps)  # ひずみエネルギー密度のスカラー値

    return strain_energy_density


def apply_density_filter(
    rho, mesh, radius
):
    """
    密度フィルター：各要素の周囲の密度を加重平均してスムージング

    Parameters:
        rho (np.ndarray): 密度ベクトル（要素ごと）
        mesh: skfemのメッシュ（mesh.p: 節点, mesh.t: 要素）
        radius (float): フィルター半径

    Returns:
        np.ndarray: フィルター後の密度
    """
    from numpy.linalg import norm

    # 要素の中心座標を取得
    centers = mesh.p[:, mesh.t].mean(axis=1).T  # shape: (n_elems, 3)
    rho_new = np.zeros_like(rho)
    weight_sum = np.zeros_like(rho)

    for i in range(len(rho)):
        for j in range(len(rho)):
            dist = norm(centers[i] - centers[j])
            if dist < radius:
                w = radius - dist
                rho_new[i] += w * rho[j]
                weight_sum[i] += w

    return rho_new / (weight_sum + 1e-8)  # 安定化のため小さい数を足す



# def apply_density_filter_cKDTree(rho, mesh, opt_target, radius=0.1):
#     """
#     opt_target の要素に対応する密度だけを対象にした高速密度フィルタ
#     rho: 最適化対象要素の密度ベクトル
#     mesh: skfemのメッシュ
#     opt_target: 最適化対象の要素インデックス（rho[i] は opt_target[i] に対応）
#     radius: フィルタ半径
#     """
#     # opt_target の要素中心座標を取得（shape: (n_opt_elem, dim)）
#     centers = mesh.p[:, mesh.t[:, opt_target]].mean(axis=1).T

#     tree = cKDTree(centers)
#     rho_new = np.zeros_like(rho)
#     weight_sum = np.zeros_like(rho)

#     for i, center in enumerate(centers):
#         neighbor_ids = tree.query_ball_point(center, r=radius)

#         dists = np.linalg.norm(centers[neighbor_ids] - center, axis=1)
#         weights = radius - dists

#         rho_new[i] = np.sum(weights * rho[neighbor_ids])
#         weight_sum[i] = np.sum(weights)

#     return rho_new / (weight_sum + 1e-8)


from scipy.spatial import cKDTree
import numpy as np

def apply_density_filter_cKDTree(rho, mesh, opt_target, radius=0.1):
    """
    rho: 全要素数 (mesh.nelements) の密度ベクトル
    mesh: skfemのメッシュ
    opt_target: 最適化対象の要素インデックス（フィルタ対象）
    radius: フィルタ半径
    """
    # opt_target の要素中心座標を取得（shape: (n_opt_elem, dim)）
    centers = mesh.p[:, mesh.t[:, opt_target]].mean(axis=1).T  # shape: (n_opt_elem, dim)

    tree = cKDTree(centers)

    # フィルタ結果を opt_target にだけ適用する
    rho_new = rho.copy()
    weight_sum = np.zeros(len(opt_target))

    for i, center in enumerate(centers):
        # i は opt_target[i] に対応
        neighbor_ids = tree.query_ball_point(center, r=radius)
        neighbor_elements = [opt_target[j] for j in neighbor_ids]

        dists = np.linalg.norm(centers[neighbor_ids] - center, axis=1)
        weights = radius - dists

        # 注意：rho は mesh 全体に対応しているので global index でアクセス
        rho_new[opt_target[i]] = np.sum(weights * rho[neighbor_elements]) / (np.sum(weights) + 1e-8)

    return rho_new


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
