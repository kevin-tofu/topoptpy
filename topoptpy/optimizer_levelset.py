import os
import shutil
import json
from dataclasses import dataclass, asdict
import numpy as np
import scipy
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import LinearNDInterpolator
import scipy.sparse.linalg as spla
import skfem
import meshio
from topoptpy import utils
from topoptpy import problem
from topoptpy import techniques
from topoptpy.history import HistoryLogger


def initialize_phi_as_sphere(mesh, radius=0.3):
    """
    球の内部 -> φ > 0, 外部 -> φ < 0
    """
    center = mesh.p.mean(axis=1)
    dist = np.linalg.norm(mesh.p.T - center, axis=1)
    return radius - dist  # dist < radius のとき φ>0


def create_phi_from_mesh_3d(mesh, solid_mask, res=64):
    """
    Create 3D level set φ from a scikit-fem 3D mesh and a solid_mask
    such that solid_mask==1 -> φ>0 (inside), solid_mask==0 -> φ<0 (outside).
    """
    coords = mesh.p.T  # shape (n_nodes, 3)
    xmin, ymin, zmin = coords.min(axis=0)
    xmax, ymax, zmax = coords.max(axis=0)

    x_grid = np.linspace(xmin, xmax, res)
    y_grid = np.linspace(ymin, ymax, res)
    z_grid = np.linspace(zmin, zmax, res)
    xv, yv, zv = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
    grid_points = np.stack([xv.ravel(), yv.ravel(), zv.ravel()], axis=1)

    # solid_mask == 1 の座標点をKD-tree化
    solid_points = coords[solid_mask.astype(bool)]
    if len(solid_points) == 0:
        # 万が一、材料点が無かったら φを全部負の大きな値にする等
        # return -1e3 * np.ones_like(coords[:, 0])
        return -1.5 * np.ones(len(coords))  # ?

    tree = cKDTree(solid_points)
    dists, _ = tree.query(grid_points, k=1)

    # ボクセルの空間分解能
    voxel_size = max(xmax - xmin, ymax - ymin, zmax - zmin) / res
    # しきい値は状況に応じて変更する
    solid_region = (dists < voxel_size * 2).astype(np.uint8)
    solid_region = solid_region.reshape((res, res, res))

    inside = distance_transform_edt(solid_region)
    outside = distance_transform_edt(1 - solid_region)

    # inside -> φ>0, outside -> φ<0 を実現したいので「outside - inside」
    phi_grid = outside - inside

    # FEMノードへ補間
    interpolator = RegularGridInterpolator((x_grid, y_grid, z_grid), 
                                           phi_grid,
                                           bounds_error=False,
                                           fill_value=None)
    phi = interpolator(coords)

    # 外挿領域のNaN処理
    phi = np.nan_to_num(phi, nan=0.0)

    # スケーリング (適宜実行)
    maxabs = np.max(np.abs(phi)) + 1e-8
    phi = 1.5 * phi / maxabs

    return phi


def smoothed_heaviside(phi, epsilon=1.0):
    """
    φ > epsilon -> 1, φ < -epsilon -> 0,
    それ以外の -ε < φ < ε では滑らか遷移
    """
    H = np.zeros_like(phi)
    idx1 = (phi > epsilon)
    idx2 = (phi < -epsilon)
    idx3 = ~(idx1 | idx2)

    H[idx1] = 1.0
    H[idx2] = 0.0

    # -ε <= φ <= ε の部分のみスムーズ
    phi_smooth = np.clip(phi[idx3], -epsilon, epsilon)
    H[idx3] = (0.5
               + phi_smooth/(2*epsilon)
               + (1/(2*np.pi)) * np.sin(np.pi * phi_smooth / epsilon))
    return H


def d_smoothed_heaviside(phi, epsilon=1.0):
    """
    d/dφ ( smoothed_heaviside(phi) )
    """
    dH = np.zeros_like(phi)
    idx = np.abs(phi) <= epsilon
    dH[idx] = (1.0 / (2.0 * epsilon)) * (1.0 + np.cos(np.pi * phi[idx] / epsilon))
    return dH


def upwind_gradient_norm(phi, dx, dy, dz):
    """
    Engquist-Osher型のUpwind差分による|∇φ|の近似計算
    phi: 3次元配列
    """
    # forward diff と backward diff
    phi_xf = np.roll(phi, -1, axis=0) - phi
    phi_xb = phi - np.roll(phi, 1, axis=0)
    phi_yf = np.roll(phi, -1, axis=1) - phi
    phi_yb = phi - np.roll(phi, 1, axis=1)
    phi_zf = np.roll(phi, -1, axis=2) - phi
    phi_zb = phi - np.roll(phi, 1, axis=2)

    # EOスキーム
    phi_x_plus = np.maximum(phi_xb, 0)**2
    phi_x_minus = np.minimum(phi_xf, 0)**2
    phi_y_plus = np.maximum(phi_yb, 0)**2
    phi_y_minus = np.minimum(phi_yf, 0)**2
    phi_z_plus = np.maximum(phi_zb, 0)**2
    phi_z_minus = np.minimum(phi_zf, 0)**2

    grad_norm = np.sqrt(
        np.maximum(phi_x_plus, phi_x_minus) / dx**2 +
        np.maximum(phi_y_plus, phi_y_minus) / dy**2 +
        np.maximum(phi_z_plus, phi_z_minus) / dz**2
    )
    return grad_norm


def update_phi_with_pde(mesh, phi, Vn, grid_resolution=64, dt=1.0, reinit=False):
    """
    1. FEMノードの (φ, Vn) を等間隔グリッドへ補間
    2. HJ式でφを更新
    3. (必要なら) 再初期化
    4. グリッド上の新しいφを再びFEMノードへ補間して返す
    """
    coords = mesh.p.T
    xmin, ymin, zmin = coords.min(axis=0)
    xmax, ymax, zmax = coords.max(axis=0)

    nx = ny = nz = grid_resolution
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    z = np.linspace(zmin, zmax, nz)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]

    xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.stack([xv, yv, zv], axis=-1).reshape(-1, 3)

    # 1. FEMノード -> グリッドへ補間
    phi_interp = LinearNDInterpolator(coords, phi, fill_value=np.nan)
    Vn_interp = LinearNDInterpolator(coords, Vn, fill_value=0.0)

    phi_grid = phi_interp(grid_points).reshape((nx, ny, nz))
    Vn_grid = Vn_interp(grid_points).reshape((nx, ny, nz))

    # NaN処理
    phi_grid = np.nan_to_num(phi_grid, nan=0.0)
    Vn_grid = np.nan_to_num(Vn_grid, nan=0.0)

    # 2. PDE更新
    phi_grid_new = update_phi_hj(phi_grid, Vn_grid, dx, dy, dz, dt)

    # 3. 再初期化
    if reinit:
        # φの正負で inside/outside を判定
        if np.all(phi_grid_new >= 0.) or np.all(phi_grid_new <= 0.):
            print("⚠ すべての点が同符号なので reinit をスキップします.")
        else:
            phi_grid_new = reinitialize_phi(phi_grid_new)

    # 4. グリッド -> FEMノードへ再補間
    grid_interp_back = RegularGridInterpolator((x, y, z), phi_grid_new,
                                               bounds_error=False,
                                               fill_value=None)
    phi_new = grid_interp_back(coords)

    # NaN処理
    phi_new = np.nan_to_num(phi_new, nan=0.0)

    # スケーリング（適宜）
    maxabs = np.max(np.abs(phi_new)) + 1e-8
    phi_new = 1.5 * phi_new / maxabs

    return phi_new

# def update_phi(phi, dC_dphi, step_size=1.0):
#     return phi - step_size * dC_dphi

def update_phi_hj(phi, Vn, dx=1.0, dy=1.0, dz=1.0, dt=1.0):
    """
    HJ方程式: φ_t + Vn * |∇φ| = 0
      -> φ_{n+1} = φ_n - dt * Vn * |∇φ|
    """
    grad_phi = upwind_gradient_norm(phi, dx, dy, dz)
    return phi - dt * Vn * grad_phi



def reinitialize_phi(phi):
    """
    現在の φ を基に，φ>0 を inside として EDTで再初期化.
      inside=0距離 -> 距離_transform=0
      outside=0距離 -> 距離_transform=0
    それぞれ計算して inside->正, outside->負 となるよう差をとる
    """
    inside_mask = (phi > 0)
    outside_mask = ~inside_mask

    phi_inside = distance_transform_edt(inside_mask)
    phi_outside = distance_transform_edt(outside_mask)

    # inside -> φ>0, outside -> φ<0 に合わせるには (out - in)
    phi_new = phi_outside - phi_inside
    return phi_new


def smoothed_heaviside_derivative(phi, epsilon=1.0):
    dH = np.zeros_like(phi)
    idx = np.abs(phi) <= epsilon
    dH[idx] = (1 / (2 * epsilon)) * (1 + np.cos(np.pi * phi[idx] / epsilon))
    return dH


def save_level_set_isosurface(prb, phi, phi_prev, rho, file_path='levelset.vtk'):
    
    mesh = prb.mesh
    dirichlet_ele = utils.get_elements_with_points(mesh, [prb.dirichlet_points])
    F_ele = utils.get_elements_with_points(mesh, [prb.F_points])
    element_colors_df1 = np.zeros(mesh.nelements, dtype=int)
    element_colors_df2 = np.zeros(mesh.nelements, dtype=int)
    element_colors_df1[prb.design_elements] = 1
    element_colors_df1[prb.fixed_elements_in_rho] = 2
    element_colors_df2[dirichlet_ele] = 1
    element_colors_df2[F_ele] = 2
    
    meshio_mesh = meshio.Mesh(
        points=mesh.p.T,
        cells=[("tetra", mesh.t.T)],
        point_data={"phi": phi, "phi-diff": phi - phi_prev},
        cell_data={"density": [rho], "condition": [element_colors_df2]}
    )
    meshio.write(file_path, meshio_mesh)



@dataclass
class LevelSetConfig():
    p: float = 3
    vol_frac: float = 0.4  # the maximum valume ratio
    learning_rate: float = 0.01
    lambda_v: float = 0.0  # constraint
    mu: float = 10.0 # penalty
    alpha: float = 1e-2
    max_iter: int = 1000
    dfilter_radius: float = 0.05
    eta: float = 0.3
    rho_min: float = 1e-3
    rho_max: float = 1.0
    move_limit: float = 0.2
    
    
    def export(self, path: str):
        with open(f"{path}/cfg.json", "w") as f:
            json.dump(asdict(self), f, indent=2)


# def solve():
    

class TopOptimizer():
    def __init__(
        self,
        prb: problem.SIMPProblem,
        cfg: LevelSetConfig,
        record_times: int,
        dst_path: str
    ):
        self.prb = prb
        self.cfg = cfg
        self.record_times = record_times
        self.dst_path = dst_path
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        self.prb.export(dst_path)
        self.cfg.export(dst_path)
        self.prb.nodes_stats(dst_path)
        
        if os.path.exists(f"{self.dst_path}/mesh_rho"):
            shutil.rmtree(f"{self.dst_path}/mesh_rho")
        os.makedirs(f"{self.dst_path}/mesh_rho")
        if os.path.exists(f"{self.dst_path}/rho-histo"):
            shutil.rmtree(f"{self.dst_path}/rho-histo")
        os.makedirs(f"{self.dst_path}/rho-histo")


    def export_mesh_org(
        self,
        value: np.ndarray,
        value_name: str,
        dst_path: str
    ):
        mesh = meshio.Mesh(
            points=self.prb.mesh.p.T,
            cells=[("tetra", self.prb.mesh.t.T)],
            cell_data={value_name: [value]}
        )
        meshio.write(
            dst_path, mesh
        )
        
    def export_mesh(
        self,
        rho: np.ndarray,
        suffix: str
    ):
        self.export_mesh_org(
            rho,
            "rho",
            f"{self.dst_path}/mesh_rho/{suffix}.vtu",
        )
    
    
    def run(
        self
    ):
        prb = self.prb
        cfg = self.cfg
        # rho = np.ones(prb.all_elements.shape)
        e_rho = skfem.ElementTetP1()
        basis_rho = skfem.Basis(prb.mesh, e_rho)
        
        grid_size = 64
        solid = np.ones(prb.mesh.p.shape[1], dtype=int)

        phi = initialize_phi_as_sphere(prb.mesh, 3.0)
        # phi = create_phi_from_mesh_3d(prb.mesh, solid, res=grid_size)
        dx, dy, dz = grid_size, grid_size, grid_size
        # basis = CellBasis(mesh, element)

        compliance_history = HistoryLogger("compliance")
        phi_history = HistoryLogger("phi")
        rho_history = HistoryLogger("rho")
        for iter in range(1, cfg.max_iter+1):
            print(f" --- {iter} / {cfg.max_iter} ---")
            print("φ > 0 割合:", np.mean(phi > 0))
            phi_prev = phi.copy()
            # phi_e = phi[prb.mesh.elements].mean(axis=0)
            # rho_e = smoothed_heaviside(phi_e, epsilon=1e-2)
            # better in terms of accuracy
            phi_at_ip = basis_rho.interpolate(phi)
            print("phi_at_ip stats → min:", phi_at_ip.min(), "max:", phi_at_ip.max(), "mean:", phi_at_ip.mean())
            print("phi_at_ip.shape", phi_at_ip.shape)

            rho_at_ip = smoothed_heaviside(phi_at_ip, epsilon=1.0).mean(axis=1)

            # print("rho_at_ip.shape:", rho_at_ip.shape)
            K = techniques.assemble_stiffness_matrix(
                prb.basis, rho_at_ip, prb.E0,
                prb.Emin, 1.0, prb.nu0
            )
            K_e, F_e = skfem.enforce(K, prb.F, D=prb.dirichlet_nodes)
            U_e = scipy.sparse.linalg.spsolve(K_e, F_e)
            # K_free = K[prb.free_nodes][:, prb.free_nodes]
            f_free = prb.F[prb.free_nodes]

            # Solve Displacement
            # u = np.zeros(K.shape[0])
            # u[prb.free_nodes] = spla.spsolve(K_free, f_free)

            # Compliance
            compliance = f_free @ U_e[prb.free_nodes]
            
            # Sensitivity Analysis ueᵀ Ke ue on each element
            dC_drho = - techniques.compute_strain_energy_1(
                U_e, K.toarray(),
                prb.basis.element_dofs
                # prb.basis.element_dofs[:, prb.design_elements]
            )

            # dC/dφ = dC/dρ × dρ/dφ
            # defines dρ/dφ on each node
            drho_dphi = d_smoothed_heaviside(phi)
            # drho_dphi = d_heaviside(phi)
            

            # distribute to nodes from elements
            dC_dphi = np.zeros_like(phi)
            counts = np.zeros_like(phi)
            for i, nodes in enumerate(prb.mesh.t.T):
                dphi_vals = drho_dphi[nodes]
                for j, n in enumerate(nodes):
                    dC_dphi[n] += dC_drho[i] * dphi_vals[j] / len(nodes)
                    counts[n] += 1
            dC_dphi /= np.maximum(counts, 1)
            
            
            lambda_vol = 5.0
            # lambda_vol = 1.0
            # lambda_vol = 0.5
            # lambda_vol = 1e-3
            p = smoothed_heaviside_derivative(
                phi, epsilon=0.05
            )
            dJ_dphi = dC_dphi + lambda_vol * p
            dJ_dphi = dJ_dphi / (np.max(np.abs(dJ_dphi)) + 1e-8)
            
            Vn = - dJ_dphi
            dt = 0.3 * min(dx, dy, dz) / (np.max(np.abs(Vn)) + 1e-8)
            phi = update_phi_with_pde(
                mesh=prb.mesh,
                phi=phi,
                Vn=Vn,
                grid_resolution=64,
                dt=dt,
                reinit=((iter - 1)  % 5 == 0) 
            )

            dC_dphi_stats = (dC_dphi.min(), dC_dphi.max(), dC_dphi.mean())
            dH_stats = smoothed_heaviside_derivative(phi, epsilon=1.0)
            dH_stats = (dH_stats.min(), dH_stats.max(), dH_stats.mean())

            print("dC_drho stats:", dC_drho.min(), dC_drho.max(), dC_drho.mean())
            print("drho_dphi stats:", drho_dphi.min(), drho_dphi.max(), drho_dphi.mean())
            print("dC_dphi stats:", dC_dphi_stats)
            print("H'(φ) stats:", dH_stats)

            save_level_set_isosurface(
                prb, phi, phi_prev, rho_at_ip,
                file_path=f'{self.dst_path}/mesh_rho/levelset-{iter}.vtu'
            )

            # if iter % 5 == 0:
            #     phi = reinitialize_phi(phi)
            
            
            phi_history.add(phi)
            rho_history.add(rho_at_ip)
            compliance_history.add(compliance)
            phi_history.print()
            rho_history.print()
            compliance_history.print()
            
            if np.mean(phi > 0) < 0.3:
                print("")
                break


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description=''
    )
    # p: float = 3.
    # vol_frac: float = 0.4  # the maximum valume ratio
    # learning_rate: float = 0.01
    # lambda_v: float = 0.0  # constraint
    # mu: float = 10.0 # penalty
    # alpha: float = 1e-2
    # max_iter = 1000
    # dfilter_radius = 0.05
    parser.add_argument(
        '--p', '-P', type=float, default=3.0, help=''
    )
    parser.add_argument(
        '--vol_frac', '-V', type=float, default=0.4, help=''
    )
    parser.add_argument(
        '--learning_rate', '-LR', type=float, default=0.1, help=''
    )
    parser.add_argument(
        '--lambda_v', '-LV', type=float, default=10.0, help=''
    )
    parser.add_argument(
        '--mu', '-M', type=float, default=20.0, help=''
    )
    parser.add_argument(
        '--alpha', '-A', type=float, default=0.01, help=''
    )
    parser.add_argument(
        '--max_iter', '-NI', type=int, default=200, help=''
    )
    parser.add_argument(
        '--dfilter_radius', '-DR', type=float, default=0.05, help=''
    )
    parser.add_argument(
        '--move_limit', '-ML', type=float, default=0.2, help=''
    )
    parser.add_argument(
        '--eta', '-ET', type=float, default=0.1, help=''
    )
    
    
    parser.add_argument(
        '--record_times', '-RT', type=int, default=20, help=''
    )
    parser.add_argument(
        '--dst_path', '-DP', type=str, default="./result/test0", help=''
    )
    parser.add_argument(
        '--problem', '-PM', type=str, default="toy2", help=''
    )
    
    args = parser.parse_args()
    

    # if args.problem == "toy1":
    #     prb = problem.toy1()
    # elif args.problem == "toy2":
    #     prb = problem.toy2()
    prb = problem.toy2()
    
    
    cfg = LevelSetConfig(
        args.p, args.vol_frac, args.learning_rate,
        args.lambda_v, args.mu, args.alpha, args.max_iter,
        args.dfilter_radius
    )

    optimizer = TopOptimizer(prb, cfg, 10, args.dst_path)
    # optimizer.run_gd()
    optimizer.run()