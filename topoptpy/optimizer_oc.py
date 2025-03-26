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
        
        # rho[prb.design_elements] = np.random.uniform(
        #     0.3, 0.6, size=len(prb.design_elements)
        # )
        
        compliance_history = list()
        lambda_v_history = list()
        rho_ave_history = list()
        rho_std_history = list()
        rho_noncout_zero = list()
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
            # dc = dc / (np.abs(dc).max() + 1e-8)
            # dc = np.clip(dc, -1.0, -1e-3)
            dc_full = np.zeros_like(rho)
            dc_full[prb.design_elements] = dc
            dc_full_filtered = techniques.helmholtz_filter_element_based_tet(
                dc_full, basis_rho, cfg.dfilter_radius
            )
            dc = dc_full[prb.design_elements]
            dc = np.clip(dc, -1.0, -1e-2)

            
            # 
            # Correction with Lagrange multipliers Bisection Method
            # 
            rho_e = rho[prb.design_elements]
            vol_frac = cfg.vol_frac
            eta = cfg.eta
            rho_min = cfg.rho_min
            rho_max = 1.0
            move_limit = cfg.move_limit
            tolerance = 1e-4
            
            eps = 1e-6
            l1 = 1e-3
            l2 = 1e3

            while (l2 - l1) / (0.5 * (l1 + l2) + eps) > tolerance:
                lmid = 0.5 * (l1 + l2)
                scaling_factor = (-dc / (lmid + eps)) ** eta
                # scaling_factor = np.clip(scaling_factor, 0.5, 1.5)
                
                rho_candidate = np.clip(
                    rho_e * scaling_factor,
                    np.maximum(rho_e - move_limit, rho_min),
                    np.minimum(rho_e + move_limit, rho_max)
                )

                vol_error = np.mean(rho_candidate) - vol_frac
                if vol_error > 0:
                    l1 = lmid
                else:
                    l2 = lmid


            print(f"l1:{l1:0.4f}, l2:{l2:0.4f}, lmid:{lmid:0.4f}, vol_error:{vol_error:0.4f}")
            lambda_v = lmid
            lambda_v_history.append(lmid)
            
            # 
            adjacency_matrix = test.build_element_adjacency_matrix(prb.basis.mesh)
            rho_full = np.full(adjacency_matrix.shape[0], cfg.rho_min)
            rho_full[prb.design_elements] = rho_candidate
            mask_full = test.extract_main_component(rho_full, adjacency_matrix, threshold=0.3)
            mask_candidate = mask_full[prb.design_elements]
            rho_candidate[~mask_candidate] = cfg.rho_min
            
            # 
            # 
            rho_prev = np.copy(rho)
            rho[prb.design_elements] = rho_candidate
            rho[prb.fixed_elements_in_rho] = 1.0
            # rho_filtered = techniques.apply_density_filter_cKDTree(
            #     rho, prb.mesh, prb.design_elements, radius=cfg.dfilter_radius
            # )
            rho_filtered = techniques.helmholtz_filter_element_based_tet(
                rho, basis_rho, cfg.dfilter_radius
            )
            # for _ in range(2):
            #     rho_filtered = techniques.helmholtz_filter_element_based_tet(
            #         rho_filtered, basis_rho, cfg.dfilter_radius
            #     )

            
            rho[prb.design_elements] = rho_filtered[prb.design_elements]
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
            rho_noncout_zero.append(np.count_nonzero(rho[prb.design_elements] > cfg.rho_min * 10.0))
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
            print(
                f"rho_noncout_zero: {rho_noncout_zero[-1]} / {len(rho[prb.design_elements])},\
                th: {rho_frac}"
            )
            

            
            # if iter % (cfg.num_iter // self.record_times) == 0 or iter == 1:
            if True:
                print(f"Saving at iteration {iter}")
                self.export_mesh(rho, str(iter))
                self.export_mesh_org(
                    rho - rho_prev,
                    "dp",
                    f"{self.dst_path}/rho-histo/dp-{str(iter)}.vtu"
                )
                utils.progress_plot(
                    compliance_history,
                    rho_diff_history,
                    lambda_v_history,
                    dc_ave_history,
                    rho_ave_history, cfg.vol_frac,
                    rho_std_history,
                    rho_noncout_zero, rho_frac,
                    f"{self.dst_path}/progress.jpg"
                )
                utils.rho_histo_plot(
                    rho[prb.design_elements],
                    f"{self.dst_path}/rho-histo/{str(iter)}.jpg"
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
            rho_noncout_zero, rho_frac,
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
    
    
    def run_gd(
        self
    ):
        prb = self.prb
        cfg = self.cfg
        rho = np.ones(prb.all_elements.shape)
        rho[prb.design_elements] = np.random.uniform(
            0.3, 0.6, size=len(prb.design_elements)
        )
        
        compliance_history = list()
        # rho_history = list()
        lambda_v_history = list()
        rho_ave_history = list()
        rho_std_history = list()
        rho_noncout_zero = list()
        dC_drho_ave_history = list()
        lambda_v = cfg.lambda_v
        for iter in range(1, cfg.num_iter+1):
            print(f"iterations: {iter} / {cfg.num_iter}")
            
            # build stiffnes matrix
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
            dC_drho = -cfg.p * (prb.E0 - prb.Emin) * (rho[prb.design_elements] ** (cfg.p - 1)) * strain_energy
            vol_error = np.mean(rho[prb.design_elements]) - cfg.vol_frac
            dC_drho += (lambda_v + cfg.mu * vol_error) / len(prb.design_elements)
            lambda_v += cfg.mu * vol_error
            dC_drho += cfg.alpha
            
            # Update Density
            rho[prb.design_elements] -= cfg.learning_rate * dC_drho
            rho[prb.fixed_elements_in_rho] = 1
            rho = techniques.apply_density_filter_cKDTree(
                rho, prb.mesh, prb.design_elements, radius=cfg.dfilter_radius
            )
            rho[prb.design_elements] = np.clip(rho[prb.design_elements], 0.01, 1.0)
            
            lambda_v_history.append(lambda_v)
            dC_drho_ave_history.append(np.average(dC_drho))
            rho_ave_history.append(np.average(rho[prb.design_elements]))
            rho_std_history.append(np.std(rho[prb.design_elements]))
            rho_noncout_zero.append(np.count_nonzero(rho[prb.design_elements] <= 0.02)) 

            
            print(f"rho: {rho}")
            print(f"compliance_history: {compliance_history[-1]}")
            print(f"rho_ave: {rho_ave_history[-1]} - target: {cfg.vol_frac}")
            print(f"rho_std: {rho_std_history[-1]}")
            print(
                f"rho_noncout_zero: {rho_noncout_zero[-1]} / {len(rho[prb.design_elements])},\
                th: {len(rho[prb.design_elements]) * cfg.vol_frac}"
            )
            
            if iter % (cfg.num_iter // self.record_times) == 0:
            # if True:
                print(f"Saving at iteration {iter}")
                self.export_mesh(rho, str(iter))
                
                utils.progress_plot(
                    compliance_history,
                    dC_drho_ave_history,
                    lambda_v_history,
                    rho_ave_history,
                    rho_std_history,
                    rho_noncout_zero,
                    f"{self.dst_path}/progress.jpg"
                )
                

            if rho_noncout_zero[-1] > len(rho[prb.design_elements]) * cfg.vol_frac * 0.95:
                break
            # break


        utils.progress_plot(
            compliance_history,
            dC_drho_ave_history,
            lambda_v_history,
            rho_ave_history,
            rho_std_history,
            rho_noncout_zero,
            f"{self.dst_path}/progress.jpg"
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