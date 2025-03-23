from dataclasses import dataclass
import numpy as np
import scipy.sparse.linalg as spla
import skfem
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
    
    
    def run(
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
        rho_ave_history = list()
        rho_std_history = list()
        rho_cout_nonzero = list()
        dC_drho_ave_history = list()

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
            dC_drho += (cfg.lambda_v + cfg.mu * vol_error) / len(prb.design_elements)
            cfg.lambda_v += cfg.mu * vol_error
            dC_drho += cfg.alpha
            dC_drho_ave_history.append(np.average(dC_drho))
            
            # Update Density
            rho[prb.design_elements] -= cfg.learning_rate * dC_drho
            rho[prb.fixed_elements_in_rho] = 1
            rho = techniques.apply_density_filter_cKDTree(
                rho, prb.mesh, prb.design_elements, radius=cfg.dfilter_radius
            )

            rho[prb.design_elements] = np.clip(rho[prb.design_elements], 0.01, 1.0)
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
            
            
            if rho_cout_nonzero[-1] > len(rho[prb.design_elements]) * cfg.vol_frac * 0.95:
                break
            # break


        utils.progress_plot(
            compliance_history,
            rho_ave_history,
            rho_std_history,
            rho_cout_nonzero,
            f"{self.dst_path}/progress.jpg"
        )

        threshold = 0.05
        remove_elements = prb.design_elements[rho[prb.design_elements] <= threshold]
        kept_elements = np.setdiff1d(prb.all_elements, remove_elements)
        utils.export_submesh(prb.mesh, kept_elements, f"{self.dst_path}/cubic_top.vtk")


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
        '--num_iter', '-NI', type=float, default=200, help=''
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
    optimizer.run()