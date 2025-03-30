import os
import math
import inspect
import shutil
import json
from dataclasses import dataclass, asdict
import numpy as np
import scipy
import scipy.sparse.linalg as spla
import skfem
import meshio
from topoptpy import utils
from topoptpy import problem
from topoptpy import  techniques
from topoptpy import history


@dataclass
class MOCConfig():
    dst_path: str = "./result"
    record_times: int=20
    p: float = 3
    p_tau: float = 20.0
    vol_frac: float = 0.4  # the maximum valume ratio
    vol_frac_tau: float = 20.0
    learning_rate: float = 0.01
    lambda_v: float = 0.0  # constraint
    mu: float = 10.0 # penalty
    alpha: float = 1e-2
    max_iters: int = 1000
    dfilter_radius: float = 0.05
    eta: float = 0.3
    rho_min: float = 1e-3
    rho_max: float = 1.0
    move_limit: float = 0.2
    

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
    
    # rho_projected = techniques.heaviside_projection(
    #     rho, beta=beta, eta=eta
    # )
    cell_outputs = dict()
    cell_outputs["rho"] = [rho]
    cell_outputs["rho-diff"] = [rho - rho_prev]
    # cell_outputs["rho_projected"] = [rho_projected]
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
    

def oc_update_with_projection(
    rho: np.ndarray, dc: np.ndarray,
    move: float=0.2, eta: float=1.0, rho_min: float=1e-3, rho_max: float=1.0
):
    dc = np.minimum(dc, -1e-8)
    dc_norm = dc / (np.abs(dc).max() + 1e-8)
    scale = (-dc_norm) ** eta
    scale = np.clip(scale, 0.5, 1.5)

    rho_candidate = rho * scale
    rho_new = np.clip(
        rho_candidate,
        np.maximum(rho - move, rho_min),
        np.minimum(rho + move, rho_max)
    )
    return rho_new


def update_params(
    cfg, iter: int
):
    p_init = 1.0
    t = iter / (cfg.max_iters - 1)
    p_now = cfg.p - (cfg.p - p_init) * math.exp( - cfg.p_tau * t)
    
    # beta_init = cfg.beta / 10.0
    # beta_now = cfg.beta - (cfg.beta - beta_init) * math.exp(- 12.0 * t)
    
    vol_frac_init = 0.8
    vol_frac_now = cfg.vol_frac + (vol_frac_init - cfg.vol_frac) * math.exp(- cfg.vol_frac_tau * t)
    
    return p_now, vol_frac_now



class TopOptimizer():
    def __init__(
        self,
        prb: problem.SIMPProblem,
        cfg: MOCConfig
    ):
        self.prb = prb
        self.cfg = cfg
        if not os.path.exists(self.cfg.dst_path):
            os.makedirs(self.cfg.dst_path)
        self.prb.export(self.cfg.dst_path)
        self.cfg.export(self.cfg.dst_path)
        self.prb.nodes_stats(self.cfg.dst_path)
        
        if os.path.exists(f"{self.cfg.dst_path}/mesh_rho"):
            shutil.rmtree(f"{self.cfg.dst_path}/mesh_rho")
        os.makedirs(f"{self.cfg.dst_path}/mesh_rho")
        if os.path.exists(f"{self.cfg.dst_path}/rho-histo"):
            shutil.rmtree(f"{self.cfg.dst_path}/rho-histo")
        os.makedirs(f"{self.cfg.dst_path}/rho-histo")
        
        self.recorder = history.HistoriesLogger(self.cfg.dst_path)
        self.recorder.add("rho")
        self.recorder.add("rho_diff")
        self.recorder.add("lambda_v")
        self.recorder.add("vol_error")
        self.recorder.add("compliance")
        self.recorder.add("dC")
        self.recorder.add("penalty")
        self.recorder.add("rho_frac")
        self.recorder_params = history.HistoriesLogger(self.cfg.dst_path)
        self.recorder_params.add("p")
        self.recorder_params.add("vol_frac")
    
    
    def run(
        self
    ):
        prb = self.prb
        cfg = self.cfg
        
        e_rho = skfem.ElementTetP1()
        basis_rho = skfem.Basis(prb.mesh, e_rho)
        p, vol_frac = update_params(cfg, 0)
        rho = np.ones(prb.all_elements.shape)
        rho[prb.design_elements] = 0.95
        # rho[prb.design_elements] = cfg.vol_frac
        # rho[prb.design_elements] = np.random.uniform(
        #     0.8, 1.0, size=len(prb.design_elements)
        # )

        K = techniques.assemble_stiffness_matrix(
            prb.basis, rho, prb.E0,
            prb.Emin, p, prb.nu0
        )

        K_e, F_e = skfem.enforce(K, prb.F, D=prb.dirichlet_nodes)
        u = scipy.sparse.linalg.spsolve(K_e, F_e)
        f_free = prb.F[prb.free_nodes]

        # Compliance
        compliance = f_free @ u[prb.free_nodes]
        self.recorder.feed_data("compliance", compliance)
        
        mu = 0.1
        lambda_v = cfg.lambda_v
        for iter in range(1, cfg.max_iters+1):
            print(f"iterations: {iter} / {cfg.max_iters}")
            p, vol_frac = update_params(cfg, iter)
            rho_prev = rho.copy()
            
            mu = max(mu*1.1, cfg.mu)
        
            #    
            rho_filtered = techniques.helmholtz_filter_element_based_tet(
                rho, basis_rho, cfg.dfilter_radius
            )
            rho_filtered[prb.fixed_elements_in_rho] = 1.0
            rho[:] = rho_filtered
            # build stiffnes matrix
            K = techniques.assemble_stiffness_matrix(
                prb.basis, rho, prb.E0,
                prb.Emin, p, prb.nu0
            )
            K_e, F_e = skfem.enforce(K, prb.F, D=prb.dirichlet_nodes)
            u = scipy.sparse.linalg.spsolve(K_e, F_e)
            f_free = prb.F[prb.free_nodes]

            # Compliance
            compliance = f_free @ u[prb.free_nodes]
            # Compute strain energy and obtain derivatives
            strain_energy = techniques.compute_strain_energy(
                u, K.toarray(), prb.basis.element_dofs[:, prb.design_elements]
            )
            dC_drho = techniques.dC_drho_simp(
                rho[prb.design_elements], strain_energy, prb.E0, prb.Emin, p
            )
            rho_e = rho[prb.design_elements]
            vol_error = np.mean(rho_e) - vol_frac
            # vol_error = rho_e - vol_frac
            # lambda_v += max(lambda_v + mu * vol_error, 0.0)
            # lambda_v += max(min(lambda_v + mu * vol_error, 1e3), 0.0)
            lambda_v += mu * vol_error
            dv = np.ones_like(dC_drho) / len(dC_drho)
            penalty = (lambda_v + mu * vol_error) * dv
            dC_drho += penalty
            rho[prb.design_elements] = oc_update_with_projection(
                rho[prb.design_elements], dC_drho,
                move=cfg.move_limit, eta=cfg.eta,
                rho_min=cfg.rho_min, rho_max=cfg.rho_max
            )
            rho[prb.fixed_elements_in_rho] = 1.0

            # 
            # 
            rho_diff = rho - rho_prev
            rho_frac = int(len(rho[prb.design_elements]) * vol_frac)
            
            self.recorder.feed_data("rho", rho[prb.design_elements])
            # self.recorder.feed_data("rho", rho)
            self.recorder.feed_data("rho_diff", rho_diff[prb.design_elements])
            self.recorder.feed_data("compliance", compliance)
            self.recorder.feed_data("dC", dC_drho)
            self.recorder.feed_data("lambda_v", lambda_v)
            self.recorder.feed_data("vol_error", vol_error)
            self.recorder.feed_data("rho_frac", rho_frac)
            self.recorder.feed_data("penalty", penalty)
            self.recorder_params.feed_data("p", p)
            self.recorder_params.feed_data("vol_frac", vol_frac)
            
            if iter % (cfg.max_iters // self.cfg.record_times) == 0 or iter == 1:
            # if True:
                print(f"Saving at iteration {iter}")
                self.recorder.print()
                self.recorder_params.print()

                self.recorder.export_progress()
                self.recorder_params.export_progress("params-sequence.jpg")
                if True:
                    
                    save_info_on_mesh(
                        prb,
                        rho, rho_prev,
                        f"{cfg.dst_path}/mesh_rho/info_mesh-{iter}.vtu"
                    )
                    threshold = 0.5
                    remove_elements = prb.design_elements[rho[prb.design_elements] <= threshold]
                    kept_elements = np.setdiff1d(prb.all_elements, remove_elements)
                    utils.export_submesh(prb.mesh, kept_elements, f"{cfg.dst_path}/cubic_top.vtk")

            # https://qiita.com/fujitagodai4/items/7cad31cc488bbb51f895

        utils.rho_histo_plot(
            rho[prb.design_elements],
            f"{self.cfg.dst_path}/rho-histo/last.jpg"
        )

        threshold = 0.05
        remove_elements = prb.design_elements[rho[prb.design_elements] <= threshold]
        kept_elements = np.setdiff1d(prb.all_elements, remove_elements)
        utils.export_submesh(prb.mesh, kept_elements, f"{self.cfg.dst_path}/cubic_top.vtk")

        self.export_mesh(rho, "last")
    

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
    # p: float = 3.
    # vol_frac: float = 0.4  # the maximum valume ratio
    # learning_rate: float = 0.01
    # lambda_v: float = 0.0  # constraint
    # mu: float = 10.0 # penalty
    # alpha: float = 1e-2
    # max_iters = 1000
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
        '--max_iters', '-NI', type=int, default=200, help=''
    )
    parser.add_argument(
        '--dfilter_radius', '-DR', type=float, default=0.05, help=''
    )
    parser.add_argument(
        '--move_limit', '-ML', type=float, default=0.2, help=''
    )
    parser.add_argument(
        '--eta', '-ET', type=float, default=1.0, help=''
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
    parser.add_argument(
        '--vol_frac_tau', '-VFT', type=float, default=20.0, help=''
    )
    parser.add_argument(
        '--p_tau', '-PT', type=float, default=20.0, help=''
    )
    args = parser.parse_args()
    

    # if args.problem == "toy1":
    #     prb = problem.toy1()
    # elif args.problem == "toy2":
    #     prb = problem.toy2()
    prb = problem.toy2()
    
    
    cfg = MOCConfig.from_defaults(
        **vars(args)
    )

    optimizer = TopOptimizer(prb, cfg)
    # optimizer.run_gd()
    optimizer.run()