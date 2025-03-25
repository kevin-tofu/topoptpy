

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
