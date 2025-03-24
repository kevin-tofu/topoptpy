import numpy as np
import matplotlib.pyplot as plt
import skfem


def get_elements_with_points(mesh: skfem.mesh, target_nodes_list: list[np.ndarray]) -> np.ndarray:
    """
    """
    all_target_nodes = np.unique(np.concatenate(target_nodes_list))
    mask = np.any(np.isin(mesh.t, all_target_nodes), axis=0)

    return np.where(mask)[0]


def get_elements_without_points(mesh: skfem.mesh, excluded_nodes_list: list[np.ndarray]):
    """
    """
    all_excluded_nodes = np.unique(np.concatenate(excluded_nodes_list))
    mask = ~np.any(np.isin(mesh.t, all_excluded_nodes), axis=0)
    return np.where(mask)[0]


def get_point_indices_in_range(
    basis: skfem.Basis, x_range: tuple, y_range: tuple, z_range: tuple
):
    x = basis.mesh.p  # (3, n_points)
    mask = (
        (x_range[0] <= x[0]) & (x[0] <= x_range[1]) &
        (y_range[0] <= x[1]) & (x[1] <= y_range[1]) &
        (z_range[0] <= x[2]) & (x[2] <= z_range[1])
    )

    return np.where(mask)[0]


def get_dofs_in_range(
    basis: skfem.Basis, x_range: tuple, y_range: tuple, z_range: tuple
):
    return basis.get_dofs(
        lambda x: (x_range[0] <= x[0]) & (x[0] <= x_range[1]) &
                  (y_range[0] <= x[1]) & (x[1] <= y_range[1]) &
                  (z_range[0] <= x[2]) & (x[2] <= z_range[1])
    )

def get_elements_in_box(
    mesh: skfem.Mesh,
    x_range: tuple,
    y_range: tuple,
    z_range: tuple
) -> np.ndarray:
    """
    Returns the indices of elements whose centers lie within the specified rectangular box.

    Parameters:
        mesh (skfem.Mesh): The mesh object.
        x_range (tuple): Range in the x-direction (xmin, xmax).
        y_range (tuple): Range in the y-direction (ymin, ymax).
        z_range (tuple): Range in the z-direction (zmin, zmax).

    Returns:
        np.ndarray: Array of indices of elements that satisfy the given conditions.

    """
    # element_centers = mesh.p[:, mesh.t].mean(axis=0)
    element_centers = np.array([np.mean(mesh.p[:, elem], axis=1) for elem in mesh.t.T]).T

    mask = (
        (x_range[0] <= element_centers[0]) & (element_centers[0] <= x_range[1]) &
        (y_range[0] <= element_centers[1]) & (element_centers[1] <= y_range[1]) &
        (z_range[0] <= element_centers[2]) & (element_centers[2] <= z_range[1])
    )

    return np.where(mask)[0]


def progress_plot(
    compliance_history,
    dC_drho_ave_history,
    lambda_history,
    rho_ave_history,
    rho_std_history,
    rho_cout_nonzero,
    dst_path: str
):
    plt.clf()
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))

    ax[0, 0].plot(compliance_history, marker='o', linestyle='-')
    ax[0, 0].set_xlabel("Iteration")
    ax[0, 0].set_ylabel("Compliance (Objective)")
    ax[0, 0].set_yscale('log')
    ax[0, 0].set_title("Compliance Minimization Progress")
    ax[0, 0].grid(True)
    
    ax[0, 1].plot(dC_drho_ave_history, marker='o', linestyle='-')
    ax[0, 1].set_xlabel("Iteration")
    ax[0, 1].set_ylabel("Compliance (Objective)")
    ax[0, 1].set_title("Compliance Minimization Progress")
    ax[0, 1].grid(True)
    
    ax[0, 2].plot(lambda_history, marker='o', linestyle='-')
    ax[0, 2].set_xlabel("Iteration")
    ax[0, 2].set_ylabel("Lagrange Multiplier")
    ax[0, 2].set_title("Lagrange Multiplier Progress")
    ax[0, 2].grid(True)

    ax[1, 0].plot(rho_ave_history, marker='o', linestyle='-')
    ax[1, 0].set_xlabel("Iteration")
    ax[1, 0].set_ylabel("Rho Ave")
    ax[1, 0].set_title("Rho Ave Progress")
    ax[1, 0].grid(True)
    
    ax[1, 1].plot(rho_std_history, marker='o', linestyle='-')
    ax[1, 1].set_xlabel("Iteration")
    ax[1, 1].set_ylabel("rho_std_history")
    ax[1, 1].set_title("rho_std_history Progress")
    ax[1, 1].grid(True)
    
    ax[1, 2].plot(rho_cout_nonzero, marker='o', linestyle='-')
    ax[1, 2].set_xlabel("Iteration")
    ax[1, 2].set_ylabel("rho_cout_nonzero")
    ax[1, 2].set_title("rho_cout_nonzero Progress")
    ax[1, 2].grid(True)
    fig.tight_layout()

    fig.savefig(dst_path)


def export_submesh(
    mesh, kept_elements, dst_path: str
):

    kept_t = mesh.t[:, kept_elements]
    unique_vertex_indices = np.unique(kept_t)
    new_points = mesh.p[:, unique_vertex_indices]
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertex_indices)}
    new_elements = np.vectorize(index_map.get)(kept_t)
    meshtype = type(mesh)
    submesh = meshtype(new_points, new_elements)
    submesh.save(dst_path)
