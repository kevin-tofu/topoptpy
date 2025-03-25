import os
import shutil
import json
from dataclasses import dataclass, asdict
import numpy as np
import scipy.sparse.linalg as spla
import skfem
import meshio
from topoptpy import utils
from topoptpy import problem
from topoptpy import  techniques
from topoptpy import test

@dataclass
class SIMPConfig():
    p: float = 3
    vol_frac: float = 0.4  # the maximum valume ratio
    learning_rate: float = 0.01
    lambda_v: float = 0.0  # constraint
    mu: float = 10.0 # penalty
    alpha: float = 1e-2
    num_iter: int = 1000
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
        cfg: SIMPConfig,
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
    
    
    def run_oc(
        self
    ):
        prb = self.prb
        cfg = self.cfg
        rho = np.ones(prb.all_elements.shape)
        e_rho = skfem.ElementTetP1()
        basis_rho = skfem.Basis(prb.mesh, e_rho)
        
        K = skfem.asm(
            techniques.get_stifness(
                prb.E0, prb.Emin, prb.nu0, cfg.p,
                rho,
                prb.design_elements
            ),
            prb.basis
        )

        K_free = K[prb.free_nodes][:, prb.free_nodes]
        f_free = prb.F[prb.free_nodes]

        # Solve Displacement
        u = np.zeros(K.shape[0])
        u[prb.free_nodes] = spla.spsolve(K_free, f_free)

        # Compliance
        compliance = f_free @ u[prb.free_nodes]
        print(f"compliance: {compliance}")
        
        rho[prb.design_elements] = np.random.uniform(
            0.3, 0.6, size=len(prb.design_elements)
        )
        mu = cfg.mu
        compliance_history = list()
        lambda_v_history = list()
        rho_ave_history = list()
        rho_std_history = list()
        rho_nonzero_cout = list()
        rho_diff_history = list()
        dc_ave_history = list()
        lambda_v = cfg.lambda_v
        compliance_history.append(compliance)
        for iter in range(1, cfg.num_iter+1):
            print(f"iterations: {iter} / {cfg.num_iter}")
            
            # build stiffnes matrix
            print(f"prb.E0, prb.Emin, prb.nu0, {cfg.p}: {prb.E0:0.4f}, {prb.Emin:0.4f}, {prb.nu0:0.4f}, {cfg.p:0.4f}")
            K = skfem.asm(
                techniques.get_stifness(
                    prb.E0, prb.Emin, prb.nu0, cfg.p,
                    rho,
                    prb.design_elements
                ),
                prb.basis
            )

            K_free = K[prb.free_nodes][:, prb.free_nodes]
            f_free = prb.F[prb.free_nodes]

            # Solve Displacement
            u = np.zeros(K.shape[0])
            u[prb.free_nodes] = spla.spsolve(K_free, f_free)

            # Compliance
            compliance = f_free @ u[prb.free_nodes]
            compliance_history.append(compliance)

            # Compute strain energy and obtain derivatives
            strain_energy = techniques.compute_strain_energy_1(
                u, K.toarray(), prb.basis.element_dofs[:, prb.design_elements]
            )

            dc = -cfg.p * (prb.E0 - prb.Emin) * (rho[prb.design_elements] ** (cfg.p - 1)) * strain_energy
            rho_e = rho[prb.design_elements]
            vol_error = np.mean(rho_e) - cfg.vol_frac
            lambda_v += mu * vol_error
            lambda_v = np.clip(lambda_v, 1e-6, 1e3)
            if np.abs(vol_error) > 1e-3:
                mu *= 1.05
            else:
                mu *= 0.95

            print(f"mu: {mu:0.4f}")
            
            # lambda_v = np.clip(lambda_v, 1e-6, 1e3)
            lambda_v_history.append(lambda_v)
            
            # dc += lambda_v + mu * vol_error
            # dc = np.clip(dc, -1e3, -1e-6)  # enforce negative
            
            
            # dc += lambda_v + mu * vol_error
            penalty = lambda_v + mu * vol_error
            dc += penalty
            # dc = np.clip(dc, -1e3, -1e-6)
            dc = dc / (np.abs(dc).max() + 1e-8)


            
            # 
            # filter
            # 
            # dc_full = np.zeros_like(rho)
            # dc_full[prb.design_elements] = dc
            # dc_full_filtered = techniques.helmholtz_filter_element_based_tet(
            #     dc_full, basis_rho, cfg.dfilter_radius
            # )
            # dc = dc_full_filtered[prb.design_elements]
            # dc = dc / (np.max(np.abs(dc)) + 1e-8)

            if True:
                safe_dc = np.clip(dc, -1e3, -1e-6)
                log_dc = np.log(-safe_dc)
                log_dc -= np.max(log_dc)
                scaling_factor = np.exp(cfg.eta * log_dc)
                scaling_factor = np.clip(scaling_factor, 0.5, 1.5)
            else:
                scaling_factor = (-dc / (np.abs(dc).max() + 1e-8)) ** cfg.eta
            rho_candidate = np.clip(
                rho_e * scaling_factor,
                np.maximum(rho_e - cfg.move_limit, cfg.rho_min),
                np.minimum(rho_e + cfg.move_limit, cfg.rho_max)
            )
            
            
            # 
            # adjacency_matrix = test.build_element_adjacency_matrix(prb.basis.mesh)
            # rho_full = np.full(adjacency_matrix.shape[0], cfg.rho_min)
            # rho_full[prb.design_elements] = rho_candidate
            # mask_full = test.extract_main_component(rho_full, adjacency_matrix, threshold=0.3)
            # mask_candidate = mask_full[prb.design_elements]
            # rho_candidate[~mask_candidate] = cfg.rho_min
            
            # 
            # 
            rho_prev = np.copy(rho)
            rho[prb.design_elements] = rho_candidate
            rho[prb.fixed_elements_in_rho] = 1.0
            # rho_filtered = techniques.apply_density_filter_cKDTree(
            #     rho, prb.mesh, prb.design_elements, radius=cfg.dfilter_radius
            # )
            # rho_filtered = techniques.helmholtz_filter_element_based_tet(
            #     rho, basis_rho, cfg.dfilter_radius
            # )

            # rho[prb.design_elements] = rho_filtered[prb.design_elements]
            rho[prb.fixed_elements_in_rho] = 1.0
            
            vol_frac_current = np.mean(rho[prb.design_elements])
            vol_error = vol_frac_current - cfg.vol_frac
            rho_diff = np.abs(rho - rho_prev)
            # max_diff = np.max(rho_diff)
            # mean_diff = np.mean(rho_diff)
            
            dc_ave_history.append(np.average(dc))
            rho_diff_history.append(np.mean(rho_diff[prb.design_elements]))
            rho_ave_history.append(np.average(rho[prb.design_elements]))
            rho_std_history.append(np.std(rho[prb.design_elements]))
            rho_nonzero_cout.append(np.count_nonzero(rho[prb.design_elements] > cfg.rho_min * 3.0))
            rho_frac = int(len(rho[prb.design_elements]) * cfg.vol_frac)

            print(f"lambda_v: {lambda_v:.4f}, vol_error: {vol_error:.4f}, rho_ave: {rho_ave_history[-1]:.4f}")
            print("scaling_factor range:", np.min(scaling_factor), np.max(scaling_factor))
            print(f"OC update: min={np.min(rho_candidate):.4f}, max={np.max(rho_candidate):.4f}, mean={np.mean(rho_candidate):.4f}")
            print(f"Move avg: {np.mean(np.abs(rho_candidate - rho_e)):.4e}")
            print("dC stats", np.min(dc), np.max(dc), np.mean(dc))
            print(f"Current vol_frac: {vol_frac_current:.4f} (target: {cfg.vol_frac})")
            print(f"rho_diff_history: {rho_diff_history[-1]}")
            print(f"compliance_history: {compliance_history[-1]}")
            print(f"rho_ave: {rho_ave_history[-1]} - target: {cfg.vol_frac}")
            print(f"rho_std: {rho_std_history[-1]}")
            print(f"rho_candidate min/max: {rho_candidate.min():.4f} / {rho_candidate.max():.4f}")
            print(
                f"rho_nonzero_cout: {rho_nonzero_cout[-1]} / {len(rho[prb.design_elements])},\
                th: {rho_frac}"
            )
            
            
            if iter % (cfg.num_iter // self.record_times) == 0 or iter == 1:
            # if True:
                print(f"Saving at iteration {iter}")
                self.export_mesh(rho, str(iter))
                self.export_mesh_org(
                    rho - rho_prev,
                    "dp",
                    f"{self.dst_path}/rho-histo/dp-{str(iter)}.vtu"
                )
                
                utils.rho_histo_plot(
                    rho[prb.design_elements],
                    f"{self.dst_path}/rho-histo/{str(iter)}.jpg"
                )
            utils.progress_plot(
                compliance_history,
                rho_diff_history,
                lambda_v_history,
                dc_ave_history,
                rho_ave_history, cfg.vol_frac,
                rho_std_history,
                rho_nonzero_cout, rho_frac,
                f"{self.dst_path}/progress.jpg"
            )

            if len(rho_diff_history) > 10 and rho_diff_history[-1] < 1e-8:
                break

        utils.progress_plot(
            compliance_history,
            rho_diff_history,
            lambda_v_history,
            dc_ave_history,
            rho_ave_history, cfg.vol_frac,
            rho_std_history,
            rho_nonzero_cout, rho_frac,
            f"{self.dst_path}/progress.jpg"
        )
        utils.rho_histo_plot(
            rho[prb.design_elements],
            f"{self.dst_path}/rho-histo/last.jpg"
        )

        threshold = 0.05
        remove_elements = prb.design_elements[rho[prb.design_elements] <= threshold]
        kept_elements = np.setdiff1d(prb.all_elements, remove_elements)
        utils.export_submesh(prb.mesh, kept_elements, f"{self.dst_path}/cubic_top.vtk")

        self.export_mesh(rho, "last")
    
    
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
    # num_iter = 1000
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
        '--num_iter', '-NI', type=int, default=200, help=''
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
    
    
    cfg = SIMPConfig(
        args.p, args.vol_frac, args.learning_rate,
        args.lambda_v, args.mu, args.alpha, args.num_iter,
        args.dfilter_radius
    )

    optimizer = TopOptimizer(prb, cfg, 10, args.dst_path)
    # optimizer.run_gd()
    optimizer.run_oc()