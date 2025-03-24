from dataclasses import dataclass
import numpy as np
import skfem
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import meshio
from topoptpy import utils



@dataclass
class SIMPProblem():
    E0: float
    nu0: float
    Emin: float
    mesh: skfem.Mesh
    basis: skfem.Basis
    dirichlet_points: np.ndarray
    dirichlet_nodes: np.ndarray
    F_points: np.ndarray
    F_nodes: np.ndarray
    F: np.ndarray
    design_elements: np.ndarray
    free_nodes: np.ndarray
    all_elements: np.ndarray
    fixed_elements_in_rho: np.ndarray
    
    def export(
        self,
        dst_path: str
    ):
        m = self.mesh
        dirichlet_ele = utils.get_elements_with_points(m, [self.dirichlet_points])
        F_ele = utils.get_elements_with_points(m, [self.F_points])
        n_elements = m.t.shape[1]
        element_colors = np.zeros(n_elements, dtype=int)
        element_colors[self.design_elements] = 1
        element_colors[self.fixed_elements_in_rho] = 4
        element_colors[dirichlet_ele] = 2
        element_colors[F_ele] = 3

        mesh_export = meshio.Mesh(
            points=m.p.T,
            cells=[("tetra", m.t.T)],
            cell_data={"highlight": [element_colors]}  # ← この順番が mesh.t.T と合ってることが重要
        )
        meshio.write(f"{dst_path}/prb_org.vtu", mesh_export)
    
    @classmethod
    def from_defaults(
        cls,
        E0: float,
        nu0: float,
        Emin: float,
        mesh: skfem.Mesh,
        basis: skfem.Basis,
        dirichlet_points: np.ndarray,
        dirichlet_nodes: np.ndarray,
        F_points: np.ndarray,
        F_nodes: np.ndarray,
        F_value: float,
        design_elements: np.ndarray,
    ) -> 'SIMPProblem':
        elements_without_bc = utils.get_elements_with_points(
            mesh, [dirichlet_points, F_points]
        )
        design_elements = np.setdiff1d(design_elements, elements_without_bc)
        if len(design_elements) == 0:
            error_msg = "⚠️Warning: `design_elements` is empty"
            raise ValueError(error_msg)
        
        all_elements = np.arange(mesh.nelements)
        fixed_elements_in_rho = np.setdiff1d(all_elements, design_elements)
        print(
            f"all_elements: {all_elements.shape}",
            f"design_elements: {design_elements.shape}",
            f"fixed_elements_in_rho: {fixed_elements_in_rho.shape}"
        )
        free_nodes = np.setdiff1d(np.arange(basis.N), dirichlet_nodes)
        F = np.zeros(basis.N)
        F[F_nodes] = F_value / len(F_nodes)

        return cls(
            E0,
            nu0,
            Emin,
            mesh,
            basis,
            dirichlet_points,
            dirichlet_nodes,
            F_points,
            F_nodes,
            F,
            design_elements,
            free_nodes,
            all_elements,
            fixed_elements_in_rho
        )
        
    def nodes_stats(self, dst_path: str):
        points = self.mesh.p.T  # shape = (n_points, 3)
        tree = cKDTree(points)
        dists, _ = tree.query(points, k=2)
        nearest_dists = dists[:, 1]  # shape = (n_points,)

        print(f"The minimum distance: {np.min(nearest_dists):.4f}")
        print(f"The maximum distance: {np.max(nearest_dists):.4f}")
        print(f"The average distance: {np.mean(nearest_dists):.4f}")
        print(f"The median distance: {np.median(nearest_dists):.4f}")
        print(f"The std distance: {np.std(nearest_dists):.4f}")

        plt.clf()
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        ax.hist(nearest_dists, bins=30, edgecolor='black')
        ax.set_xlabel("Distance from nearest node")
        ax.set_ylabel("Number of Nodes")
        ax.set_title("The histogram of nearest neighbors")
        ax.grid(True)

        fig.tight_layout() 
        fig.savefig(f"{dst_path}/nodes_stats.jpg")


def toy1():
    
    mesh = skfem.MeshTet().refined(4).with_defaults()
    e = skfem.ElementVector(skfem.ElementTetP1())
    basis = skfem.Basis(mesh, e, intorder=3)
    
    
    dirichlet_points = utils.get_point_indices_in_range(
        basis, (0.0, 0.03), (0.0, 1.0), (0.0, 1.0)
    )
    dirichlet_nodes = utils.get_dofs_in_range(
        basis, (0.0, 0.03), (0.0, 1.0), (0.0, 1.0)
    ).all()
    F_points = utils.get_point_indices_in_range(
        basis, (1.0, 1.0), (0.0, 1.0), (0.0, 1.0)
    )
    F_nodes = utils.get_dofs_in_range(
        basis, (1.0, 1.0), (0.0, 1.0), (0.0, 1.0)
    ).nodal['u^3']
    design_elements = utils.get_elements_in_box(
        mesh,
        # (0.3, 0.7), (0.0, 1.0), (0.0, 1.0)
        (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)
    )
    print("design_elements:", design_elements)

    
    return SIMPProblem.from_defaults(
        2.0e9,
        0.38,
        1e-6,
        mesh,
        basis,
        dirichlet_points,
        dirichlet_nodes,
        F_points,
        F_nodes,
        -1000.0,
        design_elements
    )