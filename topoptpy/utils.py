import numpy as np
import matplotlib.pyplot as plt
import skfem


def get_dofs_in_range(basis, x_range, y_range, z_range):
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
    element_centers = mesh.p[:, mesh.t].mean(axis=1)

    mask = (
        (x_range[0] <= element_centers[0]) & (element_centers[0] <= x_range[1]) &
        (y_range[0] <= element_centers[1]) & (element_centers[1] <= y_range[1]) &
        (z_range[0] <= element_centers[2]) & (element_centers[2] <= z_range[1])
    )

    return np.where(mask)[0]


def progress_plot(
    compliance_history,
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
    ax[0, 0].set_title("Compliance Minimization Progress")

    ax[1, 0].plot(rho_ave_history, marker='o', linestyle='-')
    ax[1, 0].set_xlabel("Iteration")
    ax[1, 0].set_ylabel("Rho Ave")
    ax[1, 0].set_title("Rho Ave Progress")
    
    ax[1, 1].plot(rho_std_history, marker='o', linestyle='-')
    ax[1, 1].set_xlabel("Iteration")
    ax[1, 1].set_ylabel("rho_std_history")
    ax[1, 1].set_title("rho_std_history Progress")
    
    ax[1, 2].plot(rho_cout_nonzero, marker='o', linestyle='-')
    ax[1, 2].set_xlabel("Iteration")
    ax[1, 2].set_ylabel("rho_cout_nonzero")
    ax[1, 2].set_title("rho_cout_nonzero Progress")
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
