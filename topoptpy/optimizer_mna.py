import os
import inspect
import math
import shutil
import json
from dataclasses import dataclass, asdict
import numpy as np
import scipy.sparse.linalg as spla
import scipy
import meshio
import nlopt
import skfem
import math
from skfem import MeshTet, ElementVector, ElementTetP1, InteriorBasis

from topoptpy import utils
from topoptpy import problem
from topoptpy import  techniques
from topoptpy import history


@dataclass
class MNAConfig():
    dst_path: str
    record_times: int
    iters: int = 0
    max_iters: int = 1000
    p: float = 3.0
    vol_frac: float = 0.4 
    dfilter_radius: float = 0.7
    rho_min: float = 1e-3
    rho_max: float = 1.0
    move_limit: float = 0.2
    beta: float = 1.0
    eta: float = 0.3
    lam: float = 5.0 # 1.0 - 100.0 How important is the stress constraint
    p_stress: float = 6 # 4- 10 The larger it is, the closer it is to the maximum stress, but the gradient becomes unstable.
    sigma_allow: float = 0.02 # 1.0 or 0.5, depends on spcecific problem. Maximum von Mises stress allowed by the structure (depends on the unit system)
    q: float = 4 # 2 - 4How explosively the penalty increases when the stress is exceeded

    @classmethod
    def from_defaults(cls, **args):
        sig = inspect.signature(cls)
        valid_keys = sig.parameters.keys()
        filtered_args = {k: v for k, v in args.items() if k in valid_keys}
        return cls(**filtered_args)

    
    def export(self, path: str):
        with open(f"{path}/cfg.json", "w") as f:
            json.dump(asdict(self), f, indent=2)


def save_info_on_mesh(
    prb,
    rho: np.ndarray,
    rho_prev: np.ndarray,
    beta, eta,
    file_path='levelset.vtk'
):
    
    mesh = prb.mesh
    dirichlet_ele = utils.get_elements_with_points(mesh, [prb.dirichlet_points])
    F_ele = utils.get_elements_with_points(mesh, [prb.F_points])
    element_colors_df1 = np.zeros(mesh.nelements, dtype=int)
    element_colors_df2 = np.zeros(mesh.nelements, dtype=int)
    element_colors_df1[prb.design_elements] = 1
    element_colors_df1[prb.fixed_elements_in_rho] = 2
    element_colors_df2[dirichlet_ele] = 1
    element_colors_df2[F_ele] = 2
    
    rho_projected = techniques.heaviside_projection(
        rho, beta=beta, eta=eta
    )
    cell_outputs = dict()
    cell_outputs["rho"] = [rho]
    cell_outputs["rho-diff"] = [rho - rho_prev]
    cell_outputs["rho_projected"] = [rho_projected]
    cell_outputs["desing-fixed"] = [element_colors_df1]
    cell_outputs["condition"] = [element_colors_df2]
    # if sigma_v is not None:
    #     cell_outputs["sigma_v"] = [sigma_v]
    
    meshio_mesh = meshio.Mesh(
        points=mesh.p.T,
        cells=[("tetra", mesh.t.T)],
        cell_data=cell_outputs
    )
    meshio.write(file_path, meshio_mesh)


def compute_von_mises_stress(basis, U, rho, E0, Emin, p, nu):
    sigma_vm = np.zeros(basis.nelems)
    E_eff = techniques.ram_interpolation(rho, E0, Emin, p)
    # E_eff = techniques.simp_interpolation(rho, E0, Emin, p)
    # E_eff = Emin + (E0 - Emin) * (rho ** p)
    lam = (nu * E_eff) / ((1 + nu) * (1 - 2 * nu))
    mu = E_eff / (2 * (1 + nu))

    grad = basis.interpolate(U).grad  # shape: (3, 3, n_qp_per_elem, n_elements)

    n_qp_per_elem = grad.shape[2]
    n_elements = grad.shape[3]
    for e in range(n_elements):
        sigma_sum = 0.0
        for k in range(n_qp_per_elem):
            g = grad[:, :, k, e]  # shape: (3, 3)

            eps = 0.5 * (g + g.T)
            tr_eps = np.trace(eps)
            sigma = lam[e] * tr_eps * np.eye(3) + 2 * mu[e] * eps
            dev = sigma - np.trace(sigma) / 3.0 * np.eye(3)
            sigma_vm_k = np.sqrt(1.5 * np.sum(dev ** 2))

            sigma_sum += sigma_vm_k

        sigma_vm[e] = sigma_sum / n_qp_per_elem
    return sigma_vm


def update_params(
    cfg
):
    p_init = 1.0
    beta_init = cfg.beta / 10.0
    # gamma = 2
    t = cfg.iters / (cfg.max_iters - 1)
    p_now = cfg.p - (cfg.p - p_init) * math.exp( - 20.0 * t)
    beta_now = cfg.beta - (cfg.beta - beta_init) * math.exp(- 12.0 * t)
    # beta_now = beta_init + (cfg.beta - beta_init) * t ** gamma
    
    return p_now, beta_now
    

def get_objective(
    prb, cfg, rho: np.ndarray, 
    p_now: float, beta_now: float,
    recorder
):
    e_rho = skfem.ElementTetP1()
    basis_rho = skfem.Basis(prb.mesh, e_rho)

    def objective(x, grad):
        
        # recorder.feed_data("beta", beta_now)
        # recorder.feed_data("p", p_now)
        
        rho_prev = rho.copy()
        
        #    
        rho[:] = cfg.rho_min  # 全体初期化
        rho[prb.design_elements] = x
        rho[prb.fixed_elements_in_rho] = 1.0
        
        rho_filtered = techniques.helmholtz_filter_element_based_tet(
            rho, basis_rho, cfg.dfilter_radius
        )
        rho_filtered[prb.fixed_elements_in_rho] = 1.0
        rho[:] = rho_filtered
        rho_projected = techniques.heaviside_projection(rho, beta=beta_now, eta=cfg.eta)
        # recorder.feed_data(
        #     "rho-diff", rho[prb.design_elements] - rho_prev[prb.design_elements]
        # )
        

        # rho_filltered[prb.fixed_elements_in_rho] = 1.0
        # rho[:] = rho_filltered

        # 
        K = techniques.assemble_stiffness_matrix_ramp(
            prb.basis, 
            rho_projected, prb.E0,
            prb.Emin, p_now, prb.nu0
        )
        K_e, F_e = skfem.enforce(K, prb.F, D=prb.dirichlet_nodes)
        U_e = scipy.sparse.linalg.spsolve(K_e, F_e)
        f_free = prb.F[prb.free_nodes]
        compliance = f_free @ U_e[prb.free_nodes]
        
        # sigma_vm = compute_von_mises_stress(
        #     # basis_rho,
        #     prb.basis,
        #     U_e,
        #     rho,
        #     prb.E0, prb.Emin, p_now, prb.nu0
        # )
        # # Soft Constraint (stress)
        # exceed = np.maximum(0.0, sigma_vm - cfg.sigma_allow)
        # recorder.feed_data("sigma_vm", sigma_vm)
        # penalty_sigma = cfg.lam * np.sum(exceed**cfg.q)  # q=2 or 4
        # penalty_connectivity = 5.0e-2 * techniques.soft_connectivity_penalty(prb.mesh, rho_projected)
        # penalty_displacement = 5.0e-3 * np.sum(U_e[prb.F_nodes]**2)

        # penalty = penalty_sigma + penalty_connectivity + penalty_displacement
        
        # fval = compliance + penalty
        # recorder.feed_data("penalty_sigma", penalty_sigma)
        # recorder.feed_data("penalty_connectivity", penalty_connectivity)
        # recorder.feed_data("penalty_displacement", penalty_displacement)
        # recorder.feed_data("fval", fval)
        # fval = compliance

        # Penalty p-norm aggregator (stress)
        # pnorm = (np.mean(sigma_vm ** cfg.p_stress)) ** (1./cfg.p_stress)
        # exceed = max(0.0, pnorm - cfg.sigma_allow)
        # penalty = exceed ** cfg.q
        # fval = compliance + cfg.lam * penalty
        
        fval = compliance
        

        # Optional: gradient approx (not implemented here, so NLopt uses finite diff)
        if grad.size > 0:
            
            # rho[:] = np.clip(rho, cfg.rho_min, 1.0)
            strain_energy = techniques.compute_strain_energy_1(
                U_e, K.toarray(),
                prb.basis.element_dofs[:, prb.design_elements]
            )
            # (dC / d rho_projected)
            # dC_drho = techniques.dC_drho_simp(
            #     rho_projected[prb.design_elements], strain_energy, prb.E0, prb.Emin, p_now
            # )
            dC_drho_projected = techniques.dC_drho_ramp(
                rho_projected[prb.design_elements], strain_energy, prb.E0, prb.Emin, p_now
            )
            # dC_drho = - p_now * (prb.E0 - prb.Emin) * (rho_projected[prb.design_elements] ** (p_now - 1)) * strain_energy
            
            # The derivatives of Heaviside
            dH = techniques.heaviside_projection_derivative(rho[prb.design_elements], beta=beta_now, eta=cfg.eta)

            # dC / d rho
            dC_drho = dC_drho_projected * dH
            
            # 
            # dC_drho = np.clip(dC_drho, -100.0, 100.0)
            # 
            # norm = np.max(np.abs(dC_drho)) + 1e-8
            # dC_drho /= norm
            
            # 
            mean_norm = np.mean(np.abs(dC_drho)) + 1e-8
            dC_drho /= mean_norm


            # delta = rho[prb.design_elements] - rho_prev[prb.design_elements]
            # print(f"delta.min={delta.min():.3f}, mean={delta.mean():.6f}, max={delta.max():.3f}")
            # print(f"positive count: {(delta > 0).sum()}, negative count: {(delta < 0).sum()}")

            # dc = np.clip(dc, -100, 100)

            # dc = np.clip(dc, -1e0, -1e-5)
            # dp = projection_derivatives(x, cfg.beta)

            # dc_full = np.zeros_like(rho)
            # dc_full[prb.design_elements] = dc
            # dc_full_filtered = techniques.helmholtz_filter_element_based_tet(
            #     dc_full, basis_rho, cfg.dfilter_radius
            # )
            # dc = dc_full_filtered[prb.design_elements]
            
            # penalty = 0.0
            # penalty = compute_stress_penalty(
            #     basis, u, rho, E0, Emin, p, nu, sigma_max, stress_weight
            # )
            # grad[:] = 0.
            grad[:] = dC_drho
            # grad[:] = np.clip(grad, -50.0, 50.0)
            # grad[:] = dc * dp
            
            recorder.feed_data("compliance", compliance)
            recorder.feed_data("rho", rho[prb.design_elements])
            recorder.feed_data("rho_projected", rho_projected)
            recorder.feed_data("strain_energy", strain_energy)
            recorder.feed_data("dC", dC_drho)
            # recorder.feed_data("dp", dp)
            # recorder.feed_data("penalty", penalty)
            recorder.print()


        recorder.export_progress()
        # if cfg.iters % (cfg.max_iters // cfg.record_times) == 0 or cfg.iters == 1:
        if True:
            
            save_info_on_mesh(
                prb,
                rho, rho_prev,
                beta_now, cfg.eta,
                f"{cfg.dst_path}/mesh_rho/info_mesh-{str(cfg.iters)}.vtu"
            )
            threshold = 0.5
            remove_elements = prb.design_elements[rho_projected[prb.design_elements] <= threshold]
            kept_elements = np.setdiff1d(prb.all_elements, remove_elements)
            utils.export_submesh(prb.mesh, kept_elements, f"{cfg.dst_path}/cubic_top.vtk")

        if np.isnan(fval) or np.isnan(np.sum(grad)):
            print("⚠ NaN detected in objective or gradient!")
            print("fval:", fval)
            print("grad min/max:", grad.min(), grad.max())

        return fval
    return objective


def get_volume_constraint(cfg, beta_now: float):
    def volume_constraint(x, grad):
        x_projected = techniques.heaviside_projection(
            x, beta=beta_now, eta=cfg.eta
        )
        if grad.size > 0:
            grad[:] = 1.0  # or np.ones_like(x)
        return np.sum(x_projected) - cfg.vol_frac * len(x_projected)
    return volume_constraint


class TopOptimizer():
    def __init__(
        self,
        prb: problem.SIMPProblem,
        cfg: MNAConfig
    ):
        self.prb = prb
        self.cfg = cfg
        self.record_times = cfg.record_times
        dst_path = cfg.dst_path
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        self.prb.export(dst_path)
        self.cfg.export(dst_path)
        self.prb.nodes_stats(dst_path)
        
        if os.path.exists(f"{self.cfg.dst_path}/mesh_rho"):
            shutil.rmtree(f"{self.cfg.dst_path}/mesh_rho")
        os.makedirs(f"{self.cfg.dst_path}/mesh_rho")
        if os.path.exists(f"{self.cfg.dst_path}/rho-histo"):
            shutil.rmtree(f"{self.cfg.dst_path}/rho-histo")
        os.makedirs(f"{self.cfg.dst_path}/rho-histo")

        self.recorder = history.HistoriesLogger(self.cfg.dst_path)
        self.recorder.add("rho")
        self.recorder.add("rho_projected")
        self.recorder.add("strain_energy")
        self.recorder.add("compliance")
        self.recorder.add("dC")
        
        # self.recorder.add("p")
        # self.recorder.add("beta")
    
    
    def run(
        self
    ):
        prb = self.prb
        cfg = self.cfg
        rho = np.ones(prb.all_elements.shape)
        x = np.full(len(prb.design_elements), 1.0)
        x = np.random.uniform(
            0.30, 0.60, size=len(prb.design_elements)
        )
        n_vars = len(x)
        
        for it in range(1, cfg.max_iters + 1):
            print(f" ------ Iteration: {it} / {cfg.max_iters}")
            cfg.iters = it
            
            # if it % 20 == 0:
                # cfg.vol_frac = max(0.3, cfg.vol_frac - 0.02)
            opt = nlopt.opt(nlopt.LD_MMA, n_vars)
            p, beta = update_params(cfg)
            objective = get_objective(
                prb, cfg, rho,
                p, beta,
                self.recorder
            )
            volume_constraint = get_volume_constraint(cfg, beta)
            # cons_func = create_ks_stress_constraint(
            #     basis, E0, Emin, p, nu,
            #     alpha, sigma_allow,
            #     force_vec, fixed_dofs,
            #     rho_global
            # )
        
            opt.set_min_objective(objective)
            opt.add_inequality_constraint(volume_constraint, 1e-6)
            # opt.add_inequality_constraint(cons_func, 1e-4)

            opt.set_maxeval(2)
            opt.set_ftol_rel(1e-3)
            opt.set_xtol_rel(1e-4) 

            # opt.set_lower_bounds(np.full(n_vars, cfg.rho_min))
            # opt.set_upper_bounds(np.full(n_vars, 1.0))
            opt.set_lower_bounds(np.maximum(0.0, x - cfg.move_limit))
            opt.set_upper_bounds(np.minimum(1.0, x + cfg.move_limit))
            x_new = opt.optimize(x)
            x_diff = np.max(np.abs(x_new - x))
            print(f"  x diff max: {x_diff:.6e}")

            x = x_new.copy()
            # log_objective_and_constraints(x)
            # visualize_structure(x)

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
            f"{self.cfg.dst_path}/mesh_rho/{suffix}.vtu",
        )
    
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description=''
    )
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
        '--max_iters', '-NI', type=int, default=200, help=''
    )
    parser.add_argument(
        '--dfilter_radius', '-DR', type=float, default=0.05, help=''
    )
    parser.add_argument(
        '--move_limit', '-ML', type=float, default=0.2, help=''
    )
    parser.add_argument(
        '--beta', '-ET', type=float, default=1.0, help=''
    )
    parser.add_argument(
        '--eta', '-BT', type=float, default=0.5, help=''
    )
    parser.add_argument(
        '--rho_min', '-RM', type=float, default=1e-3, help=''
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
    
    
    cfg = MNAConfig.from_detaults(
        **vars(args)
    )

    optimizer = TopOptimizer(prb, cfg)
    optimizer.run()
