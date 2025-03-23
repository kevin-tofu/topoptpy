from dataclasses import dataclass
import numpy as np
import skfem
from topoptpy import utils


def get_elements_without_nodes(mesh, excluded_nodes_list):
    """
    指定した複数のノード配列に含まれるノードを含まない要素のインデックスを返す
    
    Parameters:
        t (ndarray): shape = (n_nodes_per_elem, n_elements)
        excluded_nodes_list (List[np.ndarray]): 除外ノードの配列リスト
    
    Returns:
        ndarray: 除外ノードを含まない要素のインデックス（1D）
    """
    # すべての excluded_nodes を1つにまとめる
    all_excluded_nodes = np.unique(np.concatenate(excluded_nodes_list))
    
    # 含んでない要素をフィルタ
    mask = ~np.any(np.isin(mesh.t, all_excluded_nodes), axis=0)
    return np.where(mask)[0]


@dataclass
class SIMPProblem():
    E0: float
    nu0: float
    Emin: float
    mesh: skfem.Mesh
    basis: skfem.Basis
    dirichlet_nodes: np.ndarray
    F_nodes: np.ndarray
    F: np.ndarray
    design_elements: np.ndarray
    free_nodes: np.ndarray
    all_elements: np.ndarray
    fixed_elements_in_rho: np.ndarray

    
    @classmethod
    def from_defaults(
        cls,
        E0: float,
        nu0: float,
        Emin: float,
        mesh: skfem.Mesh,
        basis: skfem.Basis,
        dirichlet_nodes: np.ndarray,
        F_nodes: np.ndarray,
        F_value: float,
        design_elements: np.ndarray,
    ) -> 'SIMPProblem':
        elements_without_bc = get_elements_without_nodes(
            mesh, [dirichlet_nodes, F_nodes]
        )
        design_elements = np.setdiff1d(design_elements, elements_without_bc)
        if len(design_elements) == 0:
            error_msg = "⚠️Warning: `opt_target` is empty"
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
            dirichlet_nodes,
            F_nodes,
            F,
            design_elements,
            free_nodes,
            all_elements,
            fixed_elements_in_rho
        )


def toy1():
    
    mesh = skfem.MeshTet().refined(4).with_defaults()
    e = skfem.ElementVector(skfem.ElementTetP1())
    basis = skfem.Basis(mesh, e, intorder=3)
    mesh.save("cubic_org.vtu")
    
    dirichlet_nodes = utils.get_dofs_in_range(
        basis, (0.0, 0.03), (0.0, 1.0), (0.0, 1.0)
    ).all()
    F_nodes = utils.get_dofs_in_range(
        basis, (1.0, 1.0), (0.0, 1.0), (0.0, 1.0)
    ).nodal['u^3']
    design_elements = utils.get_elements_in_box(
        mesh, (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)
    )
    
    return SIMPProblem.from_defaults(
        2.0e9,
        0.38,
        1e-6,
        mesh,
        basis,
        dirichlet_nodes,
        F_nodes,
        -1000.0,
        design_elements
    )