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


@dataclass
class SIMPConfig():
    p: float = 3.
    vol_frac: float = 0.4  # the maximum valume ratio
    learning_rate: float = 0.01
    lambda_v: float = 0.0  # constraint
    mu: float = 10.0 # penalty
    alpha: float = 1e-2
    num_iter: int = 1000
    dfilter_radius: float = 0.05
    eta: float = 0.02
    rho_min: float = 0.001
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

    def export_mesh(
        self,
        rho: np.ndarray,
        suffix: str
    ):
        mesh = meshio.Mesh(
            points=self.prb.mesh.p.T,
            cells=[("tetra", self.prb.mesh.t.T)],
            cell_data={"rho": [rho]}
        )
        meshio.write(
            f"{self.dst_path}/mesh_rho/{suffix}.vtu", mesh
        )
    
    
    def run_oc(
        self
    ):
        prb = self.prb
        cfg = self.cfg
        rho = np.ones(prb.all_elements.shape)
        rho[prb.design_elements] = np.random.uniform(
            0.3, 0.6, size=len(prb.design_elements)
        )
        e_rho = skfem.ElementTetP1()
        basis_rho = skfem.Basis(prb.mesh, e_rho)
        
        compliance_history = list()
        lambda_history = list()
        rho_ave_history = list()
        rho_std_history = list()
        rho_cout_nonzero = list()
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

            dc = -cfg.p * (prb.E0 - prb.Emin) * (rho[prb.design_elements] ** (cfg.p - 1)) * strain_energy

            # 
            # Correction with Lagrange multipliers Bisection Method
            # 
            rho_e = rho[prb.design_elements]
            vol_frac = cfg.vol_frac
            eta = cfg.eta
            rho_min = cfg.rho_min
            rho_max = 1.0
            move_limit = cfg.move_limit
            l1, l2 = 1e-9, 1e9
            tolerance = 1e-4
            while (l2 - l1) / (l1 + l2) > tolerance:
                lmid = 0.5 * (l1 + l2)
                eps = 1e-8
                rho_candidate = np.clip(
                    rho_e * ((-dc / (lmid + eps)) ** eta),
                    rho_e - move_limit,
                    rho_e + move_limit
                )

                rho_candidate = np.clip(rho_candidate, rho_min, rho_max)

                if rho_candidate.mean() - vol_frac > 0:
                    l1 = lmid
                else:
                    l2 = lmid

            # 
            # Filtering and Result
            # 
            rho[prb.design_elements] = rho_candidate
            # rho_filtered = techniques.apply_density_filter_cKDTree(
            #     rho, prb.mesh, prb.design_elements, radius=cfg.dfilter_radius
            # )
            rho_filtered = techniques.helmholtz_filter_element_based_tet(
                rho, basis_rho, cfg.dfilter_radius
            )
            rho[prb.design_elements] = rho_filtered[prb.design_elements]
            
            lambda_history.append(lambda_v)
            dC_drho_ave_history.append(np.average(dc))
            rho_ave_history.append(np.average(rho[prb.design_elements]))
            rho_std_history.append(np.std(rho[prb.design_elements]))
            rho_cout_nonzero.append(np.count_nonzero(rho[prb.design_elements] <= 0.02)) 

            
            print(f"rho[prb.design_elements]: {rho[prb.design_elements]}")
            print(f"compliance_history: {compliance_history[-1]}")
            print(f"rho_ave: {rho_ave_history[-1]} - target: {cfg.vol_frac}")
            print(f"rho_std: {rho_std_history[-1]}")
            print(
                f"rho_cout_nonzero: {rho_cout_nonzero[-1]} / {len(rho[prb.design_elements])},\
                th: {len(rho[prb.design_elements]) * cfg.vol_frac}"
            )
            
            if iter % (cfg.num_iter // self.record_times) == 0:
            # if True:
                print(f"Saving at iteration {iter}")
                self.export_mesh(rho, str(iter))
                
                utils.progress_plot(
                    compliance_history,
                    dC_drho_ave_history,
                    lambda_history,
                    rho_ave_history,
                    rho_std_history,
                    rho_cout_nonzero,
                    f"{self.dst_path}/progress.jpg"
                )

            if rho_cout_nonzero[-1] > len(rho[prb.design_elements]) * cfg.vol_frac * 0.95:
                break
            # break


        utils.progress_plot(
            compliance_history,
            dC_drho_ave_history,
            lambda_history,
            rho_ave_history,
            rho_std_history,
            rho_cout_nonzero,
            f"{self.dst_path}/progress.jpg"
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
        lambda_history = list()
        rho_ave_history = list()
        rho_std_history = list()
        rho_cout_nonzero = list()
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
            
            lambda_history.append(lambda_v)
            dC_drho_ave_history.append(np.average(dC_drho))
            rho_ave_history.append(np.average(rho[prb.design_elements]))
            rho_std_history.append(np.std(rho[prb.design_elements]))
            rho_cout_nonzero.append(np.count_nonzero(rho[prb.design_elements] <= 0.02)) 

            
            print(f"rho: {rho}")
            print(f"compliance_history: {compliance_history[-1]}")
            print(f"rho_ave: {rho_ave_history[-1]} - target: {cfg.vol_frac}")
            print(f"rho_std: {rho_std_history[-1]}")
            print(
                f"rho_cout_nonzero: {rho_cout_nonzero[-1]} / {len(rho[prb.design_elements])},\
                th: {len(rho[prb.design_elements]) * cfg.vol_frac}"
            )
            
            if iter % (cfg.num_iter // self.record_times) == 0:
            # if True:
                print(f"Saving at iteration {iter}")
                self.export_mesh(rho, str(iter))
                
                utils.progress_plot(
                    compliance_history,
                    dC_drho_ave_history,
                    lambda_history,
                    rho_ave_history,
                    rho_std_history,
                    rho_cout_nonzero,
                    f"{self.dst_path}/progress.jpg"
                )
                

            if rho_cout_nonzero[-1] > len(rho[prb.design_elements]) * cfg.vol_frac * 0.95:
                break
            # break


        utils.progress_plot(
            compliance_history,
            dC_drho_ave_history,
            lambda_history,
            rho_ave_history,
            rho_std_history,
            rho_cout_nonzero,
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
        '--p', '-P', type=float, default=0.3, help=''
    )
    parser.add_argument(
        '--vol_frac', '-V', type=float, default=0.4, help=''
    )
    parser.add_argument(
        '--learning_rate', '-LR', type=float, default=0.1, help=''
    )
    parser.add_argument(
        '--lambda_v', '-LV', type=float, default=0.0, help=''
    )
    parser.add_argument(
        '--mu', '-M', type=float, default=200.0, help=''
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
        '--dst_path', '-DP', type=str, default="./result/test0", help=''
    )
    args = parser.parse_args()
    

    
    prb = problem.toy1()
    cfg = SIMPConfig(
        args.p, args.vol_frac, args.learning_rate,
        args.lambda_v, args.mu, args.alpha, args.num_iter,
        args.dfilter_radius
    )

    optimizer = TopOptimizer(prb, cfg, 10, args.dst_path)
    # optimizer.run()
    optimizer.run_oc()