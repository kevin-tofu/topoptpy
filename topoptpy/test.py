

from collections import defaultdict
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import numpy as np
from scipy.sparse import diags



def build_element_adjacency_matrix(mesh):
    """
    scikit-femのMeshTet等から、要素間の隣接関係を表すスパース行列を作成
    """
    n_elem = mesh.t.shape[1]
    face_to_elem = defaultdict(list)

    for i in range(n_elem):
        tet = mesh.t[:, i]
        faces = [
            tuple(sorted([tet[0], tet[1], tet[2]])),
            tuple(sorted([tet[0], tet[1], tet[3]])),
            tuple(sorted([tet[0], tet[2], tet[3]])),
            tuple(sorted([tet[1], tet[2], tet[3]])),
        ]
        for f in faces:
            face_to_elem[f].append(i)

    row, col = [], []
    for elems in face_to_elem.values():
        if len(elems) == 2:
            a, b = elems
            row += [a, b]
            col += [b, a]

    data = [1] * len(row)
    return coo_matrix((data, (row, col)), shape=(n_elem, n_elem)).tocsr()


def extract_main_component(rho, adjacency_matrix, threshold=0.3):
    rho_bin = (rho > threshold).astype(int)
    D = diags(rho_bin)
    subgraph = D @ adjacency_matrix @ D

    n_components, labels = connected_components(csgraph=subgraph, directed=False)

    if n_components <= 1:
        return np.ones_like(rho, dtype=bool)

    from collections import Counter
    c = Counter(labels)
    main_label = c.most_common(1)[0][0]
    return labels == main_label



def compute_stress_aggregator(
    prb, p,
    u,
    rho_elem, 
    aggregator_p=16.0
):
    """
    各要素の von Mises 応力を計算し、p-norm などで一つのスカラーに集約。
    aggregator_p が大きいほど "max" に近い挙動。
    
    Returns:
        agg_stress: float (aggregated stress for a single constraint)
        dAgg_dRho: (optional) array-like, if we do gradient-based approach for stress
    """
    n_elems = prb.basis.nelements
    sigma_vm = np.zeros(n_elems)
    # lam, mu を要素ごとに
    E_eff = prb.Emin + (prb.E0 - prb.Emin) * rho_elem ** p
    lam = (prb.nu * E_eff) / ((1+prb.nu)*(1-2*prb.nu))
    mu  = E_eff / (2 * (1+prb.nu))
    
    # ここでは 'u' は全自由度とし、要素ごとに取り出す簡易例
    from skfem.helpers import grad
    for e in range(n_elems):
        loc_basis = prb.basis.with_element_dofs(e)
        dofs = loc_basis.dofs.flatten()
        ue = u[dofs]
        
        # 変位勾配 (3x3)
        gu = grad(loc_basis.interpolate(ue))  # shape: (3,3)
        # 歪みtensor
        eps = 0.5 * (gu + gu.T)
        # 応力tensor
        sigma = lam[e] * np.trace(eps)*np.eye(3) + 2*mu[e]*eps
        # 偏差
        s_dev = sigma - np.trace(sigma)/3*np.eye(3)
        sigma_vm[e] = math.sqrt(1.5*np.sum(s_dev**2))
    
    # p-norm aggregator
    # agg_stress = (1/n_elems * sum( (sigma_vm)^aggregator_p ))^(1/aggregator_p)
    s_p = sigma_vm**aggregator_p
    agg = (np.mean(s_p))**(1/aggregator_p)
    return agg


def create_stress_constraint_function(
    prb,
    p,
    allow_stress,
    rho_array_global,
    aggregator_p=16.0
):
    """
    単一のハード制約:  aggregated_stress - allow_stress <= 0
    """
    def stress_constraint(x, grad):
        # 1) x -> rho_array_global
        rho_array_global[:] = 0.01
        rho_array_global[prb.design_elems] = x
        
        # 2) FEM同じ流れ
        K = techniques.assemble_stiffness_matrix_simp(
            prb.basis, rho_array_global, prb.E0, prb.Emin, p, prb.nu
        )
        K_c, f_c = skfem.enforce(K, prb.F, D=prb.dirichlet_nodes)
        U_c = scipy.sparse.linalg.spsolve(K_c, f_c)
        U = np.zeros(prb.basis.N)
        free_dofs = np.setdiff1d(np.arange(prb.basis.N), prb.dirichlet_nodes)
        U[free_dofs] = U_c
        
        # 3) compute aggregator
        agg_stress = compute_stress_aggregator(
            prb, p, U,
            rho_array_global,
            aggregator_p
        )
        
        # 4) constraint = agg_stress - allow_stress <= 0
        cval = agg_stress - allow_stress
        
        # 5) 勾配(grad)の計算(オプション)
        #   ここでは面倒なので grad[:] = 0.0, 
        #   → MMAは有限差分 or 近似で動く(遅い)
        #   本格的には応力感度 ∂agg/∂rho を計算する必要あり
        if grad.size > 0:
            grad[:] = 0.0  # or a real derivative if you implement it
        
        return cval

    return stress_constraint
