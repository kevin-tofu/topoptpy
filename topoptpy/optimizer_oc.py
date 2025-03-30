import os
import inspect
import math
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
class OCConfig():
    dst_path: str = "./result"
    record_times: int=20
    max_iters: int=200
    p: float = 3
    p_rate: float = 20.0
    vol_frac: float = 0.4  # the maximum valume ratio
    vol_frac_rate: float = 20.0
    learning_rate: float = 0.01
    beta: float = 8
    beta_rate: float = 20.
    beta_eta: float = 0.3
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


# def schedule_exp(
#     iteration, total_iter,
#     start=1.0, target=0.4, 
#     rate=3.0
# ):
#     t = iteration / total_iter
#     return target + (start - target) * np.exp(-rate * t)


def schedule_exp_slowdown(
    it, total, start=1.0, target=0.4, rate=10.0
):
    t = it / total
    if start > target:
        return target + (start - target) * np.exp(-rate * t)
    else:
        return target - (target - start) * np.exp(-rate * t)


def schedule_exp_accelerate(
    it, total, start=1.0, target=0.4, rate=10.0
):
    t = it / total
    if start > target:
        return target + (start - target) * (1 - np.exp(rate * (t - 1)))
    else:
        return target - (target - start) * (1 - np.exp(rate * (t - 1)))


def get_update_params(p_init, vol_frac_init, beta_init):
    def update_params(
        cfg, iter: int
    ):
        t = iter / (cfg.max_iters - 1)
        # p_now = cfg.p - (cfg.p - p_init) * math.exp( - cfg.p_rate * t)
        p_now = schedule_exp_slowdown(iter, cfg.max_iters, p_init, cfg.p, cfg.p_rate)
        beta_now = schedule_exp_slowdown(iter, cfg.max_iters, beta_init, cfg.beta, cfg.beta_rate)
        # vol_frac_init = 0.95
        vol_frac_now = schedule_exp_slowdown(
            iter, cfg.max_iters, vol_frac_init, cfg.vol_frac, cfg.vol_frac_rate
        )
        return p_now, vol_frac_now, beta_now

    return update_params
    

def plot_schedule(
    cfg,
    p_init: float, vol_frac_init: float,
    dst_path: str
):
    import matplotlib.pyplot as plt
    update_params = get_update_params(p_init, vol_frac_init, cfg.beta / 100.0)
    p_list = list()
    vol_frac_list = list()
    beta_list = list()
    for iter in range(1, cfg.max_iters+1):
        p, vol_frac, beta = update_params(cfg, iter)
        p_list.append(p)
        vol_frac_list.append(vol_frac)
        beta_list.append(beta)

    fig, ax = plt.subplots(1, 3, figsize=(12, 8))
    ax[0].plot(p_list, marker='o', linestyle='-')
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("P")
    ax[0].set_title("p schedule")
    ax[0].grid(True)
    ax[1].plot(vol_frac_list, marker='o', linestyle='-')
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("vol_frac")
    ax[1].set_title("vol_frac schedule")
    ax[1].grid(True)
    ax[2].plot(beta_list, marker='o', linestyle='-')
    ax[2].set_xlabel("Iteration")
    ax[2].set_ylabel("beta")
    ax[2].set_title("beta schedule")
    ax[2].grid(True)

    fig.tight_layout()
    fig.savefig(dst_path)
    plt.close("all")


class TopOptimizer():
    def __init__(
        self,
        prb: problem.SIMPProblem,
        cfg: OCConfig
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
        # self.recorder.add("penalty")
        self.recorder.add("scaling_rate")
        self.recorder.add("strain_energy")
        self.recorder_params = history.HistoriesLogger(self.cfg.dst_path)
        self.recorder_params.add("p")
        self.recorder_params.add("vol_frac")
        self.recorder_params.add("beta")
        


    def run(
        self
    ):
        
        e_rho = skfem.ElementTetP1()
        basis_rho = skfem.Basis(prb.mesh, e_rho)
        rho = np.ones(prb.all_elements.shape)
        # rho[prb.design_elements] = 0.95
        # rho[prb.design_elements] = cfg.vol_frac
        rho[prb.design_elements] = np.random.uniform(
            0.5, 0.8, size=len(prb.design_elements)
        )
        plot_schedule(
            cfg, 1.0, np.mean(rho[prb.design_elements]),
            f"{self.cfg.dst_path}/schedule.jpg"
        )
        update_params = get_update_params(1.0, np.mean(rho[prb.design_elements]))
        p, vol_frac = update_params(cfg, 0)
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
        eta = cfg.eta
        rho_min = cfg.rho_min
        rho_max = 1.0
        move_limit = cfg.move_limit
        tolerance = 1e-4
        eps = 1e-6
        # l1 = 1e-5
        # l2 = 1e5
        # l1, l2 = 1e-9, 1e9
        l1, l2 = 1e-9, 1e4
        # l1, l2 = 1e-4, 1e4
        # l1, l2 = 1e-5, 1e5
        rho_prev = np.zeros_like(rho)
        for iter in range(1, cfg.max_iters+1):
            print(f"iterations: {iter} / {cfg.max_iters}")
            p, vol_frac = update_params(cfg, iter)
            rho_prev[:] = rho[:]
            # build stiffnes matrix
            
            K = techniques.assemble_stiffness_matrix_ramp(
            # K = techniques.assemble_stiffness_matrix(
                prb.basis, rho, prb.E0,
                prb.Emin, p, prb.nu0
            )
            K_e, F_e = skfem.enforce(K, prb.F, D=prb.dirichlet_nodes)
            u = scipy.sparse.linalg.spsolve(K_e, F_e)
            f_free = prb.F[prb.free_nodes]
            # Compliance
            compliance = f_free @ u[prb.free_nodes]
            
            rho_filtered = techniques.helmholtz_filter_element_based_tet(
                rho, basis_rho, cfg.dfilter_radius
            )
            rho_filtered[prb.fixed_elements_in_rho] = 1.0
            rho[:] = rho_filtered
            # Compute strain energy and obtain derivatives
            strain_energy = techniques.compute_strain_energy(
                u, prb.basis.element_dofs[:, prb.design_elements],
                prb.basis, rho[prb.design_elements],
                prb.E0, prb.Emin, p, prb.nu0
            )
            dC_drho = techniques.dC_drho_simp(
                rho[prb.design_elements], strain_energy, prb.E0, prb.Emin, p
            )
            # 
            # Correction with Lagrange multipliers Bisection Method
            # 
            safe_dC = dC_drho - np.mean(dC_drho)
            safe_dC = safe_dC / (np.max(np.abs(safe_dC)) + 1e-8)
            
            rho_e = rho[prb.design_elements].copy()
            lmid = 0.5 * (l1 + l2)
            vol_error = np.mean(rho_e) - vol_frac
            l1, l2 = 1e-9, 1e9
            while abs(l2 - l1) > tolerance * (l1 + l2) / 2.0:
            # while (l2 - l1) / (0.5 * (l1 + l2) + eps) > tolerance:
            # while abs(vol_error) > 1e-2:
                lmid = 0.5 * (l1 + l2)
                scaling_rate = (- safe_dC / (lmid + eps)) ** eta
                scaling_rate = np.clip(scaling_rate, 0.5, 1.5)

                rho_candidate = np.clip(
                    rho_e * scaling_rate,
                    np.maximum(rho_e - move_limit, rho_min),
                    np.minimum(rho_e + move_limit, rho_max)
                )
                vol_error = np.mean(rho_candidate) - vol_frac
                if vol_error > 0:
                    l1 = lmid
                else:
                    l2 = lmid

            rho_diff = rho - rho_prev

            self.recorder.feed_data("rho_diff", rho_diff[prb.design_elements])
            self.recorder.feed_data("scaling_rate", scaling_rate)
            self.recorder.feed_data("rho", rho[prb.design_elements])
            self.recorder.feed_data("compliance", compliance)
            self.recorder.feed_data("dC", dC_drho)
            self.recorder.feed_data("lambda_v", lmid)
            self.recorder.feed_data("vol_error", vol_error)
            self.recorder.feed_data("strain_energy", strain_energy)
            self.recorder_params.feed_data("p", p)
            self.recorder_params.feed_data("vol_frac", vol_frac)
            
            
            
            
            #     noise_strength = 0.03
            #     # rho[prb.design_elements] += np.random.uniform(
            #     #     -noise_strength, noise_strength, size=prb.design_elements.shape
            #     # )
            #     rho[prb.design_elements] += -safe_dC / (np.abs(safe_dC).max() + 1e-8) * 0.05 \
            #         + np.random.normal(0, noise_strength, size=prb.design_elements.shape)
            #     rho[prb.design_elements] = np.clip(rho[prb.design_elements], cfg.rho_min, cfg.rho_max)

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



    def run_ramp(
        self
    ):
        
        e_rho = skfem.ElementTetP1()
        basis_rho = skfem.Basis(prb.mesh, e_rho)
        rho = np.ones(prb.all_elements.shape)
        # rho[prb.design_elements] = 0.95
        # rho[prb.design_elements] = cfg.vol_frac
        rho[prb.design_elements] = np.random.uniform(
            0.5, 0.8, size=len(prb.design_elements)
        )
        rho_projected = techniques.heaviside_projection(
            rho, beta=cfg.beta / 10.0, eta=cfg.beta_eta
        )
        plot_schedule(
            cfg,
            1.0, np.mean(rho[prb.design_elements]), 
            f"{self.cfg.dst_path}/schedule.jpg"
        )
        update_params = get_update_params(
            1.0, np.mean(rho[prb.design_elements]), cfg.beta / 10.0
        )
        p, vol_frac, beta = update_params(cfg, 0)
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
        eta = cfg.eta
        rho_min = cfg.rho_min
        rho_max = 1.0
        move_limit = cfg.move_limit
        tolerance = 1e-4
        eps = 1e-6
        # l1 = 1e-5
        # l2 = 1e5
        # l1, l2 = 1e-9, 1e9
        l1, l2 = 1e-9, 1e4
        # l1, l2 = 1e-4, 1e4
        # l1, l2 = 1e-5, 1e5
        rho_prev = np.zeros_like(rho)
        for iter in range(1, cfg.max_iters+1):
            print(f"iterations: {iter} / {cfg.max_iters}")
            p, vol_frac, beta = update_params(cfg, iter)
            rho_prev[:] = rho[:]
            # build stiffnes matrix
            
            rho_filtered = techniques.helmholtz_filter_element_based_tet(
                rho, basis_rho, cfg.dfilter_radius
            )
            rho_filtered[prb.fixed_elements_in_rho] = 1.0
            rho[:] = rho_filtered
            rho_projected = techniques.heaviside_projection(
                rho, beta=beta, eta=cfg.beta_eta
            )
            K = techniques.assemble_stiffness_matrix_ramp(
            # K = techniques.assemble_stiffness_matrix(
                prb.basis, rho_projected, prb.E0,
                prb.Emin, p, prb.nu0
            )
            K_e, F_e = skfem.enforce(K, prb.F, D=prb.dirichlet_nodes)
            u = scipy.sparse.linalg.spsolve(K_e, F_e)
            f_free = prb.F[prb.free_nodes]
            # Compliance
            compliance = f_free @ u[prb.free_nodes]
            
            
            # Compute strain energy and obtain derivatives
            strain_energy = techniques.compute_strain_energy(
                u, prb.basis.element_dofs[:, prb.design_elements],
                prb.basis, rho_projected[prb.design_elements],
                prb.E0, prb.Emin, p, prb.nu0
            )
            dC_drho_projected = techniques.dC_drho_ramp(
                rho_projected[prb.design_elements], strain_energy, prb.E0, prb.Emin, p
            )
            dH = techniques.heaviside_projection_derivative(
                rho[prb.design_elements], beta=beta, eta=cfg.beta_eta
            )
            dC_drho = dC_drho_projected * dH
            # 
            # Correction with Lagrange multipliers Bisection Method
            # 
            safe_dC = dC_drho - np.mean(dC_drho)
            safe_dC = safe_dC / (np.max(np.abs(safe_dC)) + 1e-8)
            
            safe_dC = np.clip(safe_dC, -1.0, 1.0)
            
            rho_e = rho[prb.design_elements].copy()
            l1, l2 = 1e-9, 1e4
            lmid = 0.5 * (l1 + l2)
            while abs(l2 - l1) > tolerance * (l1 + l2) / 2.0:
            # while (l2 - l1) / (0.5 * (l1 + l2) + eps) > tolerance:
            # while abs(vol_error) > 1e-2:
                lmid = 0.5 * (l1 + l2)
                scaling_rate = (- safe_dC / (lmid + eps)) ** eta
                scaling_rate = np.clip(scaling_rate, 0.5, 1.5)

                rho_candidate = np.clip(
                    rho_e * scaling_rate,
                    np.maximum(rho_e - move_limit, rho_min),
                    np.minimum(rho_e + move_limit, rho_max)
                )
                rho_candidate_projected = techniques.heaviside_projection(
                    rho_candidate, beta=beta, eta=cfg.beta_eta
                )
                vol_error = np.mean(rho_candidate_projected) - vol_frac
                if vol_error > 0:
                    l1 = lmid
                else:
                    l2 = lmid

            rho[prb.design_elements] = rho_candidate
            rho_diff = rho - rho_prev

            self.recorder.feed_data("rho_diff", rho_diff[prb.design_elements])
            self.recorder.feed_data("scaling_rate", scaling_rate)
            self.recorder.feed_data("rho", rho_projected[prb.design_elements])
            self.recorder.feed_data("compliance", compliance)
            self.recorder.feed_data("dC", dC_drho)
            self.recorder.feed_data("lambda_v", lmid)
            self.recorder.feed_data("vol_error", vol_error)
            self.recorder.feed_data("strain_energy", strain_energy)
            self.recorder_params.feed_data("p", p)
            self.recorder_params.feed_data("vol_frac", vol_frac)
            self.recorder_params.feed_data("beta", beta)
            
            
            
            # if np.sum(np.abs(rho_diff)) < 1e-3:
            #     noise_strength = 0.03
            #     # rho[prb.design_elements] += np.random.uniform(
            #     #     -noise_strength, noise_strength, size=prb.design_elements.shape
            #     # )
            #     rho[prb.design_elements] += -safe_dC / (np.abs(safe_dC).max() + 1e-8) * 0.05 \
            #         + np.random.normal(0, noise_strength, size=prb.design_elements.shape)
            #     rho[prb.design_elements] = np.clip(rho[prb.design_elements], cfg.rho_min, cfg.rho_max)

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
                        rho_projected, rho_prev,
                        f"{cfg.dst_path}/mesh_rho/info_mesh-{iter}.vtu"
                    )
                    threshold = 0.5
                    remove_elements = prb.design_elements[rho_projected[prb.design_elements] <= threshold]
                    kept_elements = np.setdiff1d(prb.all_elements, remove_elements)
                    utils.export_submesh(prb.mesh, kept_elements, f"{cfg.dst_path}/cubic_top.vtk")

            # https://qiita.com/fujitagodai4/items/7cad31cc488bbb51f895

        utils.rho_histo_plot(
            rho_projected[prb.design_elements],
            f"{self.cfg.dst_path}/rho-histo/last.jpg"
        )

        threshold = 0.05
        remove_elements = prb.design_elements[rho_projected[prb.design_elements] <= threshold]
        kept_elements = np.setdiff1d(prb.all_elements, remove_elements)
        utils.export_submesh(prb.mesh, kept_elements, f"{self.cfg.dst_path}/cubic_top.vtk")

        self.export_mesh(rho_projected, "last")


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
        '--vol_frac_rate', '-VFT', type=float, default=20.0, help=''
    )
    parser.add_argument(
        '--p_rate', '-PT', type=float, default=20.0, help=''
    )
    parser.add_argument(
        '--beta', '-B', type=float, default=100.0, help=''
    )
    parser.add_argument(
        '--beta_rate', '-BR', type=float, default=20.0, help=''
    )
    args = parser.parse_args()
    

    # if args.problem == "toy1":
    #     prb = problem.toy1()
    # elif args.problem == "toy2":
    #     prb = problem.toy2()
    prb = problem.toy2()
    
    
    cfg = OCConfig.from_defaults(
        **vars(args)
    )

    optimizer = TopOptimizer(prb, cfg)
    # optimizer.run_gd()
    # optimizer.run()
    optimizer.run_ramp()