import pathlib
from dataclasses import dataclass
import numpy as np
import skfem
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import meshio
from topoptpy import tools
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
        element_colors_df1 = np.zeros(n_elements, dtype=int)
        element_colors_df2 = np.zeros(n_elements, dtype=int)
        element_colors_df1[self.design_elements] = 1
        element_colors_df1[self.fixed_elements_in_rho] = 2
        element_colors_df2[dirichlet_ele] = 1
        element_colors_df2[F_ele] = 2

        mesh_export = meshio.Mesh(
            points=m.p.T,
            cells=[("tetra", m.t.T)],
            cell_data={"highlight": [element_colors_df1]}
        )
        meshio.write(f"{dst_path}/prb_org_design_fix.vtu", mesh_export)
        mesh_export = meshio.Mesh(
            points=m.p.T,
            cells=[("tetra", m.t.T)],
            cell_data={"highlight": [element_colors_df2]}
        )
        meshio.write(f"{dst_path}/prb_org_dirichlet_f.vtu", mesh_export)
    
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
        
        bc_elements = utils.get_elements_with_points(
            mesh, [dirichlet_points]
        )
        bc_elements_adj = tools.get_adjacent_elements(mesh, bc_elements)
        f_elements = utils.get_elements_with_points(
            mesh, [F_points]
        )
        elements_related_with_bc = np.concatenate([bc_elements, bc_elements_adj, f_elements])
        
        design_elements = np.setdiff1d(design_elements, elements_related_with_bc)
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

def toy2():
    import gmsh

    gmsh.initialize()
    x_len = 8.0
    y_len = 12.0
    z_len = 3
    mesh_size = 0.8
    mesh_size = 0.5

    gmsh.model.add('plate')
    gmsh.model.occ.addBox(0, 0, 0, x_len, y_len, z_len)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)

    gmsh.model.mesh.setOrder(1)
    gmsh.model.mesh.generate(3)
    gmsh.write("plate.msh")
    gmsh.finalize()

    # mesh = skfem.MeshTet().refined(4).with_defaults()
    mesh = skfem.MeshTet.load(pathlib.Path('plate.msh'))
    e = skfem.ElementVector(skfem.ElementTetP1())
    basis = skfem.Basis(mesh, e, intorder=3)

    dirichlet_points = utils.get_point_indices_in_range(
        basis, (0.0, 0.03), (0.0, y_len), (0.0, z_len)
    )
    dirichlet_nodes = utils.get_dofs_in_range(
        basis, (0.0, 0.03), (0.0, y_len), (0.0, z_len)
    ).all()
    F_points = utils.get_point_indices_in_range(
        basis, (x_len, x_len), (y_len*2/5, y_len*3/5), (0.0, z_len/5)
    )
    F_nodes = utils.get_dofs_in_range(
        basis, (x_len, x_len), (y_len*2/5, y_len*3/5), (0.0, z_len/5)
    ).nodal['u^3']
    design_elements = utils.get_elements_in_box(
        mesh,
        # (0.3, 0.7), (0.0, 1.0), (0.0, 1.0)
        (0.0, x_len), (0.0, y_len), (0.0, z_len)
    )

    print("design_elements:", design_elements)
    print("F_nodes", F_nodes.shape)

    # E0 = 2.0e9
    E0 = 1.0
    # F = -1000.0
    F = -0.3
    return SIMPProblem.from_defaults(
        E0,
        0.30,
        1e-3 * E0,
        mesh,
        basis,
        dirichlet_points,
        dirichlet_nodes,
        F_points,
        F_nodes,
        F,
        design_elements
    )