import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import time
import numpy as np
import torch
from glob import glob
from scipy import sparse
from scipy.sparse.csgraph import depth_first_order, minimum_spanning_tree
from qaia.SB import DSB

try:
    import mkl
    mkl.set_num_threads(1)
except Exception:
    pass

try:
    torch.set_num_threads(1)
except Exception:
    pass

'''
必须基于 MindQuantum 量子计算框架实现。允许调用 MindQuantum 官方算法库，或基于该框架编写自定义逻辑。
'''
'''
该函数禁止改动!!!
'''
def read_graph_file(filename, negate=True):
    """
    Reads graph data from a text file and constructs the adjacency matrix.
    """
    baseline = 0.0
    time_baseline = 0.0
    
    with open(filename, 'r') as f:
        header = f.readline().strip().split()
        num_nodes = int(header[0])

        if len(header) >= 3:
            baseline = float(header[2])
            
        if len(header) >= 4:
            time_baseline = float(header[3])

        rows = []
        cols = []
        data = []
        
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2: 
                continue
            u = int(parts[0]) - 1 
            v = int(parts[1]) - 1

            w = float(parts[2]) if len(parts) > 2 else 1.0

            if negate:
                w = -w

            rows.append(u)
            cols.append(v)
            data.append(w)
            
            rows.append(v)
            cols.append(u)
            data.append(w)

    adj_matrix = sparse.coo_matrix((data, (rows, cols)), 
                                   shape=(num_nodes, num_nodes))
    
    return adj_matrix.tocsr(), baseline, time_baseline


'''
该函数禁止改动!!!
'''
def scipy_to_torch_sparse(G_csr, device='cpu'):
    """ 
    Converts a Scipy sparse matrix (CSR) to a PyTorch sparse tensor (COO).
    """
    G_coo = G_csr.tocoo()
    indices_np = np.vstack((G_coo.row, G_coo.col))
    indices = torch.from_numpy(indices_np).long().to(device)
    values = torch.from_numpy(G_coo.data).float().to(device)
    shape = G_coo.shape
    G_torch = torch.sparse_coo_tensor(indices, values, shape, device=device)
    return G_torch.coalesce()


'''
该函数禁止改动!!!
'''
def update_sparse_matrix_weights(G_csr, classification):
    """
    Updates the graph weights based on the current partition (classification).
    
    Mechanism:
    Inverts the sign of edges that connect nodes in different partitions (Cross-Edges).
    This encourages the solver to explore different cuts in subsequent iterations.
    """
    G_coo = G_csr.tocoo()
    class_row = classification[G_coo.row]
    class_col = classification[G_coo.col]
    cross_edges = class_row != class_col
    G_coo.data[cross_edges] = -G_coo.data[cross_edges]
    modified_count = np.sum(cross_edges) // 2
    return G_coo.tocsr(), modified_count


'''
该函数禁止改动!!!
'''
def get_smaller_subset(classification):
    """
    Identifies indices of the smaller partition subset.
    Used to balance the search process.
    """
    subset1_mask = classification == 1
    subset1_count = np.sum(subset1_mask)
    subset2_count = len(classification) - subset1_count
    if subset1_count <= subset2_count:
        return np.where(subset1_mask)[0]
    else:
        return np.where(~subset1_mask)[0]


'''
该函数禁止改动!!!
'''
def calculate_cut_value(G_torch, partition_np, device='cpu'):
    """
    Calculating cut value.
    
    Args:
        G_torch: The original PyTorch sparse tensor
        partition_np: Classification results of nodes (numpy array, {-1, 1})
    """
    partition_torch = torch.FloatTensor(partition_np).to(device).unsqueeze(1)
    
    # Energy Calculation:
    # Ising Hamiltonian H = -0.5 * s^T * J * s
    energy = -0.5 * torch.sum(G_torch @ partition_torch * partition_torch)
    
    # Cut Value Formula:
    # Cut = 0.25 * Sum(W) - 0.25 * s^T * W * s
    # Cut = -0.25 * Sum(G_torch) - 0.5 * Energy
    cut_value = (-0.5 * energy - 0.25 * G_torch.sum()).item()
    return cut_value


'''
必须基于 MindQuantum 量子计算框架实现。允许调用 MindQuantum 官方算法库，或基于该框架编写自定义逻辑。
'''
def _target_reached(cut_value, cut_value_baseline):
    if cut_value_baseline is None or cut_value_baseline <= 0:
        return False
    return cut_value >= cut_value_baseline - 1e-4 or round(float(cut_value), 2) >= round(float(cut_value_baseline), 2)


def _stable_seed(*items):
    seed = 2166136261
    for item in items:
        for byte in str(item).encode('utf-8'):
            seed ^= byte
            seed = (seed * 16777619) & 0xffffffff
    return seed


def _extract_original_edges(G_csr):
    """Return one undirected copy of the original weights W from G=-W."""
    graph = G_csr.tocoo(copy=False)
    mask = graph.row < graph.col
    rows = graph.row[mask].astype(np.int32, copy=False)
    cols = graph.col[mask].astype(np.int32, copy=False)
    weights = (-graph.data[mask]).astype(np.float64, copy=False)
    return rows, cols, weights


def _cut_value_from_graph(G_csr, graph_sum, partition):
    spin = partition.astype(np.float64, copy=False)
    return float(0.25 * np.sum(spin * G_csr.dot(spin)) - 0.25 * graph_sum)


def _cut_values_from_graph_batch(G_csr, graph_sum, partitions):
    spins = np.asarray(partitions)
    if spins.ndim == 1:
        return np.asarray([_cut_value_from_graph(G_csr, graph_sum, spins)])

    spins = spins.astype(np.float64, copy=False)
    fields = G_csr.dot(spins)
    quadratic = np.asarray(spins * fields).sum(axis=0)
    return 0.25 * quadratic - 0.25 * graph_sum


def _cut_values_and_fields_from_graph_batch(G_csr, graph_sum, partitions):
    spins = np.asarray(partitions)
    if spins.ndim == 1:
        spins = spins.reshape(-1, 1)
    spins = spins.astype(np.float64, copy=False)
    fields = G_csr.dot(spins)
    quadratic = np.asarray(spins * fields).sum(axis=0)
    return 0.25 * quadratic - 0.25 * graph_sum, fields


def _update_best(best, cut_value, partition, label):
    if best is None or cut_value > best[0]:
        return (float(cut_value), partition.astype(np.int8, copy=False), label)
    return best


def _rank_quantile(values, quantile):
    if len(values) == 0:
        return 0.0
    rank = int(quantile * (len(values) - 1))
    return float(np.partition(values, rank)[rank])


def _lookup_csr_value(matrix, row, col):
    start, end = matrix.indptr[row], matrix.indptr[row + 1]
    offset = np.searchsorted(matrix.indices[start:end], col)
    pos = start + offset
    if pos < end and matrix.indices[pos] == col:
        return matrix.data[pos]
    return matrix[col, row]


def _partition_and_graph_from_tree(
    G_csr, tree, tree_data_is_original=True, cost_lookup=None, return_labels=False,
    force_flip_tree=False
):
    tree = tree.tocoo()
    num_nodes = G_csr.shape[0]
    tree_rows = tree.row.astype(np.int32, copy=False)
    tree_cols = tree.col.astype(np.int32, copy=False)
    if tree_data_is_original:
        costs = tree.data.astype(np.float64, copy=False)
    elif cost_lookup is not None:
        costs = np.asarray(cost_lookup[tree_rows, tree_cols]).ravel().astype(np.float64, copy=False)
        missing = costs == 0
        if np.any(missing):
            costs[missing] = np.asarray(cost_lookup[tree_cols[missing], tree_rows[missing]]).ravel()
    else:
        costs = np.empty_like(tree.data, dtype=np.float64)
        for idx, (a, b) in enumerate(zip(tree_rows, tree_cols)):
            costs[idx] = _lookup_csr_value(G_csr, int(a), int(b))

    edge_count = len(tree_rows)
    q_rows = np.empty(edge_count * 2, dtype=np.int32)
    q_cols = np.empty(edge_count * 2, dtype=np.int32)
    q_data = np.empty(edge_count * 2, dtype=np.float64)
    q_rows[0::2] = tree_rows
    q_rows[1::2] = tree_cols
    q_cols[0::2] = tree_cols
    q_cols[1::2] = tree_rows
    q_data[0::2] = costs
    q_data[1::2] = costs
    qaia_graph = sparse.csr_matrix((q_data, (q_rows, q_cols)), shape=G_csr.shape)

    flip_every_tree_edge = bool(
        force_flip_tree or (tree_data_is_original and edge_count > 0 and np.all(costs < 0.0))
    )

    bits = np.zeros(num_nodes, dtype=np.int8)
    seen = np.zeros(num_nodes, dtype=bool)
    labels = np.empty(num_nodes, dtype=np.int32) if return_labels else None
    component_id = 0
    if flip_every_tree_edge:
        for start in range(num_nodes):
            if seen[start]:
                continue
            order, predecessors = depth_first_order(
                qaia_graph, start, directed=False, return_predecessors=True
            )
            for raw_node in order:
                node = int(raw_node)
                parent = int(predecessors[node])
                if parent >= 0:
                    bits[node] = bits[parent] ^ 1
            seen[order] = True
            if labels is not None:
                labels[order] = component_id
            component_id += 1

        partition = (1 - (bits << 1)).astype(np.int8, copy=False)
        if return_labels:
            return partition, qaia_graph, labels, component_id
        return partition, qaia_graph

    indptr = qaia_graph.indptr
    indices = qaia_graph.indices
    data = qaia_graph.data
    for start in range(num_nodes):
        if seen[start]:
            continue
        seen[start] = True
        if labels is not None:
            labels[start] = component_id
        stack = [start]
        while stack:
            node = stack.pop()
            node_bit = bits[node]
            for pos in range(indptr[node], indptr[node + 1]):
                neighbor = int(indices[pos])
                if not seen[neighbor]:
                    seen[neighbor] = True
                    if labels is not None:
                        labels[neighbor] = component_id
                    bits[neighbor] = node_bit ^ int(data[pos] < 0)
                    stack.append(neighbor)
        component_id += 1

    partition = (1 - (bits << 1)).astype(np.int8, copy=False)
    if return_labels:
        return partition, qaia_graph, labels, component_id
    return partition, qaia_graph


def _tree_partition_labels(tree, num_nodes, return_labels=False):
    tree = tree.tocoo()
    tree_rows = tree.row.astype(np.int32, copy=False)
    tree_cols = tree.col.astype(np.int32, copy=False)
    tree_signs = tree.data < 0
    edge_count = len(tree_rows)

    degree = np.bincount(
        np.concatenate((tree_rows, tree_cols)),
        minlength=num_nodes,
    ).astype(np.int32, copy=False)
    indptr = np.empty(num_nodes + 1, dtype=np.int32)
    indptr[0] = 0
    np.cumsum(degree, out=indptr[1:])
    indices = np.empty(edge_count * 2, dtype=np.int32)
    signs = np.empty(edge_count * 2, dtype=bool)
    cursor = indptr[:-1].copy()
    for row, col, sign in zip(tree_rows, tree_cols, tree_signs):
        pos = cursor[row]
        indices[pos] = col
        signs[pos] = sign
        cursor[row] += 1
        pos = cursor[col]
        indices[pos] = row
        signs[pos] = sign
        cursor[col] += 1

    bits = np.zeros(num_nodes, dtype=np.int8)
    seen = np.zeros(num_nodes, dtype=bool)
    labels = np.empty(num_nodes, dtype=np.int32) if return_labels else None
    component_id = 0
    for start in range(num_nodes):
        if seen[start]:
            continue
        seen[start] = True
        if labels is not None:
            labels[start] = component_id
        stack = [start]
        while stack:
            node = stack.pop()
            node_bit = bits[node]
            for pos in range(indptr[node], indptr[node + 1]):
                neighbor = int(indices[pos])
                if not seen[neighbor]:
                    seen[neighbor] = True
                    if labels is not None:
                        labels[neighbor] = component_id
                    bits[neighbor] = node_bit ^ int(signs[pos])
                    stack.append(neighbor)
        component_id += 1

    partition = np.where(bits == 0, 1, -1).astype(np.int8)
    if return_labels:
        return partition, labels, component_id
    return partition


def _weight_tree_warm_start(G_csr, rows, cols, weights):
    """Compressed signed subgraph from the maximum spanning tree of W."""
    half_graph = sparse.coo_matrix((-weights, (rows, cols)), shape=G_csr.shape).tocsr()
    return _partition_and_graph_from_tree(G_csr, minimum_spanning_tree(half_graph), tree_data_is_original=True)


def _orient_components(rows, cols, weights, partition, labels, num_components):
    if num_components <= 1 or num_components > 24:
        return partition

    if len(weights) > 1000000:
        stride = max(1, len(weights) // 1000000)
        rows = rows[::stride][:1000000]
        cols = cols[::stride][:1000000]
        weights = weights[::stride][:1000000]

    cross = labels[rows] != labels[cols]
    if not np.any(cross):
        return partition

    comp_u = labels[rows[cross]]
    comp_v = labels[cols[cross]]
    coeff = weights[cross] * partition[rows[cross]] * partition[cols[cross]]
    interaction = np.bincount(
        comp_u * num_components + comp_v,
        weights=coeff,
        minlength=num_components * num_components,
    ).reshape(num_components, num_components)
    interaction += interaction.T

    signs = np.ones(num_components, dtype=np.int8)
    field = interaction @ signs
    value = float(np.sum(signs * field))
    best_value = value
    best_signs = signs.copy()
    prev_gray = 0

    for step in range(1, 1 << (num_components - 1)):
        gray = step ^ (step >> 1)
        diff = gray ^ prev_gray
        comp = int(diff & -diff).bit_length()
        old_sign = float(signs[comp])
        value += -4.0 * old_sign * field[comp]
        signs[comp] = np.int8(-signs[comp])
        field += -2.0 * old_sign * interaction[:, comp]
        if value < best_value:
            best_value = value
            best_signs = signs.copy()
        prev_gray = gray

    return (partition * best_signs[labels]).astype(np.int8)


def _positive_quantile_tree(G_csr, rows, cols, weights, quantile=0.25, max_components=24):
    positive_idx = np.flatnonzero(weights > 0)
    positive_count = int(positive_idx.size)
    if positive_count == 0:
        return None, 0

    positive_weights = weights[positive_idx]
    sample_size = 512
    exact_threshold = None
    if positive_count > sample_size:
        stride = max(1, positive_count // sample_size)
        positive_sample = positive_weights[::stride][:sample_size]
        threshold = _rank_quantile(positive_sample, quantile)
    else:
        threshold = _rank_quantile(positive_weights, quantile)
        exact_threshold = threshold

    for allow_exact_retry in (True, False):
        selected_idx = positive_idx[positive_weights >= threshold]
        sub_graph = sparse.coo_matrix(
            (-weights[selected_idx], (rows[selected_idx], cols[selected_idx])), shape=G_csr.shape
        )
        tree = minimum_spanning_tree(sub_graph).tocsr()
        num_components = int(G_csr.shape[0] - tree.nnz)
        if num_components <= max_components:
            break
        if not allow_exact_retry or exact_threshold is not None:
            return None, int(num_components)
        exact_threshold = _rank_quantile(positive_weights, quantile)
        threshold = exact_threshold
    return tree, int(num_components)


def _positive_quantile_warm_start(G_csr, rows, cols, weights, quantile=0.25, max_components=24):
    tree, num_components = _positive_quantile_tree(
        G_csr, rows, cols, weights, quantile=quantile, max_components=max_components
    )
    if tree is None:
        return None, num_components

    partition, qaia_graph, labels, num_components = _partition_and_graph_from_tree(
        G_csr, tree, tree_data_is_original=True, return_labels=True
    )
    partition = _orient_components(rows, cols, weights, partition, labels, num_components)
    return (partition, qaia_graph), int(num_components)


def _qaia_xi(qaia_graph):
    norm = np.sqrt(float(np.sum(qaia_graph.data * qaia_graph.data)))
    return 0.5 * np.sqrt(qaia_graph.shape[0] - 1) / norm if norm > 0 else 1.0


def _qaia_local_refine(G_csr, partition, graph_sum, initial_cut, params, rng, graph_product=None):
    active_size = int(params.get('active_size', 32))
    if active_size <= 0:
        return partition, float(initial_cut), 0.0

    spin = partition.astype(np.int8, copy=True)
    spin_f = spin.astype(np.float64, copy=False)
    if graph_product is None:
        graph_product = G_csr.dot(spin_f)
    else:
        graph_product = np.asarray(graph_product, dtype=np.float64).reshape(-1)
    fields = -graph_product
    gains = spin_f * fields
    active = np.flatnonzero(gains > 1e-10)
    if active.size == 0:
        return spin, float(initial_cut), 0.0
    if active.size > active_size:
        keep = np.argpartition(gains[active], -active_size)[-active_size:]
        active = active[keep]
    active = np.asarray(active, dtype=np.int32)

    local_nodes = int(active.size)
    active_spin = spin_f[active]
    active_lookup = {int(node): idx for idx, node in enumerate(active)}
    rows = []
    cols = []
    data = []

    active_total = graph_product[active]
    active_internal = np.zeros(local_nodes, dtype=np.float64)
    indptr, indices, graph_data = G_csr.indptr, G_csr.indices, G_csr.data
    for local_row, node in enumerate(active):
        start, end = indptr[node], indptr[node + 1]
        row_spin = active_spin[local_row]
        for pos in range(start, end):
            local_col = active_lookup.get(int(indices[pos]))
            if local_col is None:
                continue
            value = float(graph_data[pos])
            active_internal[local_row] += value * active_spin[local_col]
            rows.append(local_row)
            cols.append(local_col)
            data.append(value * row_spin * active_spin[local_col])

    boundary = active_spin * (active_total - active_internal)
    anchor = local_nodes
    for idx, value in enumerate(boundary):
        if abs(value) <= 1e-12:
            continue
        rows.append(idx)
        cols.append(anchor)
        data.append(float(value))
        rows.append(anchor)
        cols.append(idx)
        data.append(float(value))

    if not data:
        return spin, float(initial_cut), 0.0

    local_graph = sparse.csr_matrix(
        (np.asarray(data, dtype=np.float64), (np.asarray(rows), np.asarray(cols))),
        shape=(local_nodes + 1, local_nodes + 1),
    )
    local_start = np.ones(local_nodes + 1, dtype=np.int8)
    local_start_f = local_start.astype(np.float64, copy=False)
    local_start_q = float(local_start_f @ local_graph.dot(local_start_f))
    local_parts, _ = _qaia_dsb_refine(
        local_graph,
        local_start,
        xscale=float(params.get('xscale', 0.05)),
        noise=float(params.get('noise', 0.01)),
        n_iter=int(params.get('n_iter', 16)),
        restarts=int(params.get('restarts', 8)),
        jitter_rate=0.0,
        rng=rng,
        return_starts=True,
        dt=float(params.get('dt', 0.25)),
        anneal_stages=params.get('anneal_stages'),
    )

    best_partition = spin
    best_cut = float(initial_cut)
    best_toggles = None
    for col in range(local_parts.shape[1]):
        local_state = local_parts[:, col].astype(np.int8, copy=False)
        anchor_state = int(local_state[anchor])
        if anchor_state == 0:
            anchor_state = 1
        toggles = (local_state[:local_nodes] * anchor_state).astype(np.int8, copy=False)
        normalized = np.empty(local_nodes + 1, dtype=np.float64)
        normalized[:local_nodes] = toggles.astype(np.float64, copy=False)
        normalized[anchor] = 1.0
        local_q = float(normalized @ local_graph.dot(normalized))
        candidate_cut = float(initial_cut) + 0.25 * (local_q - local_start_q)
        if candidate_cut > best_cut:
            best_cut = candidate_cut
            best_toggles = toggles.copy()

    if best_toggles is not None:
        best_partition = spin.copy()
        best_partition[active] = (best_partition[active] * best_toggles).astype(np.int8, copy=False)

    return best_partition, best_cut, best_cut - float(initial_cut)


def _time_exceeded(start_time, time_budget):
    return (
        start_time is not None
        and time_budget is not None
        and time.time() - start_time > time_budget
    )


def _time_remaining(start_time, time_budget):
    if start_time is None or time_budget is None:
        return None
    return max(0.0, time_budget - (time.time() - start_time))


def _best_partition_or_default(best, num_nodes):
    if best is not None:
        return best[1]
    return np.ones(num_nodes, dtype=np.int8)


def _graph_features(num_nodes, weights):
    edge_count = int(len(weights))
    density = edge_count / max(1, num_nodes)
    if edge_count == 0:
        return {
            'num_nodes': int(num_nodes),
            'edge_count': 0,
            'density': 0.0,
            'abs_mean': 0.0,
            'abs_var': 0.0,
            'abs_cv': 0.0,
        }

    if edge_count > 4096:
        stride = max(1, edge_count // 4096)
        feature_weights = np.abs(weights[::stride][:4096])
    else:
        feature_weights = np.abs(weights)

    abs_mean = float(np.mean(feature_weights))
    abs_var = float(np.var(feature_weights))
    abs_cv = float(np.sqrt(abs_var) / (abs_mean + 1e-12))
    return {
        'num_nodes': int(num_nodes),
        'edge_count': edge_count,
        'density': float(density),
        'abs_mean': abs_mean,
        'abs_var': abs_var,
        'abs_cv': abs_cv,
    }


def _basic_graph_features(num_nodes, edge_count):
    return {
        'num_nodes': int(num_nodes),
        'edge_count': int(edge_count),
        'density': float(edge_count / max(1, num_nodes)),
        'abs_mean': 0.0,
        'abs_var': 0.0,
        'abs_cv': 0.0,
    }


def _dynamic_solver_params(features, time_budget):
    """Build QAIA search parameters from cheap graph statistics."""
    num_nodes = features['num_nodes']
    density = features['density']
    abs_cv = features['abs_cv']

    size_pressure = np.clip(np.log1p(num_nodes) / np.log1p(24000.0), 0.0, 1.25)
    density_pressure = density / (density + 96.0) if density > 0 else 0.0
    weight_pressure = abs_cv / (abs_cv + 1.0) if abs_cv > 0 else 0.0
    if time_budget is None:
        time_pressure = 1.0
    else:
        time_pressure = time_budget / (time_budget + 0.45)

    noise = 0.006 * (1.0 + 0.12 * size_pressure + 0.10 * weight_pressure - 0.08 * density_pressure)
    noise = float(np.clip(noise, 0.0048, 0.0078))
    dt = float(0.2 - 0.025 * density_pressure)
    restart_pressure = time_pressure * (0.60 + 0.40 * density_pressure)
    qaia_restarts = int(np.clip(round(1 + restart_pressure), 1, 2))

    base_positive_trials = [
        (0.70, 16),
        (0.65, 16),
        (0.55, 20),
        (0.35, 24),
    ]
    base_randomized_trials = [
        (0.1, 0.60, (1,)),
        (0.1, 0.40, (1,)),
        (0.1, 0.15, (1, 0)),
        (0.1, 0.10, (1, 0)),
        (0.1, 0.0, (2, 0)),
    ]

    positive_count = int(np.clip(round(2 + 2 * time_pressure - 0.4 * size_pressure), 2, len(base_positive_trials)))
    randomized_count = int(np.clip(round(1 + 4 * time_pressure - 0.8 * size_pressure), 1, len(base_randomized_trials)))
    positive_trials = base_positive_trials[:positive_count]
    randomized_trials = base_randomized_trials[:randomized_count]

    stage1_iter = 1
    stage2_iter = int(np.clip(round(3 + density_pressure * time_pressure), 3, 4))

    post_anneal_stages = (
        {
            'xscale': 0.04,
            'noise': noise * 1.5,
            'n_iter': stage1_iter,
            'dt': dt,
        },
        {
            'xscale': 0.03,
            'noise': noise,
            'n_iter': stage2_iter,
            'dt': dt,
        },
    )

    return {
        'positive_trials': tuple(positive_trials),
        'randomized_trials': tuple(randomized_trials),
        'batch_randomized': bool(time_budget is None),
        'qaia': {
            'xscale': 0.03,
            'noise': noise,
            'n_iter': 8,
            'dt': dt,
            'anneal_stages': None,
            'post_anneal_stages': post_anneal_stages if time_budget is not None else None,
            'jitter_rate': 0.0015,
            'restarts': qaia_restarts,
        },
        'local_refine': {
            'active_size': 8,
            'xscale': 0.01,
            'noise_mult': 0.5,
            'noise': noise * 0.5,
            'n_iter': 2,
            'dt': 0.12,
            'restarts': 4,
        },
        'full_refine': {
            'xscale': 0.04,
            'noise_mult': 1.0,
            'n_iter': 4,
            'dt': dt,
            'restarts': 1,
        },
    }


def _jitter_warm_start(partition, restarts, flip_rate, rng):
    partition = np.asarray(partition)
    if partition.ndim == 1:
        starts = np.repeat(partition.reshape(-1, 1), restarts, axis=1).astype(np.int8, copy=True)
    else:
        starts = partition.astype(np.int8, copy=True)
        if starts.shape[1] != restarts:
            restarts = starts.shape[1]

    if flip_rate <= 0 or len(partition) == 0:
        return starts

    flip_mask = rng.random(starts.shape) < flip_rate
    starts[flip_mask] *= -1
    return starts


def _prepare_x_bias(x_bias, starts, xscale):
    if x_bias is None:
        return starts.astype(np.float64, copy=False) * xscale

    bias = np.asarray(x_bias, dtype=np.float64)
    if bias.ndim == 1:
        bias = np.repeat(bias.reshape(-1, 1), starts.shape[1], axis=1)
    if bias.shape != starts.shape:
        return starts.astype(np.float64, copy=False) * xscale

    max_abs = np.max(np.abs(bias), axis=0, keepdims=True)
    max_abs[max_abs < 1e-12] = 1.0
    return (bias / max_abs) * xscale


def _qaia_dsb_refine(
    qaia_graph, partition, xscale=0.03, noise=0.006, n_iter=8, restarts=1, jitter_rate=0.0,
    rng=None, return_starts=False, dt=0.2, anneal_stages=None, x_bias=None
):
    """
    Refine a warm-start partition with QAIA's discrete simulated bifurcation.

    The warm start only biases the initial oscillator signs.  Each restart
    receives standard random noise in the continuous oscillator values.  The
    final candidates are produced by DSB rather than copied from the classical
    compressed-graph initializer.
    """
    if rng is None:
        rng = np.random.default_rng(_stable_seed("qaia-dsb-refine"))

    starts = _jitter_warm_start(partition.astype(np.int8, copy=False), restarts, jitter_rate, rng)
    restarts = starts.shape[1]
    if anneal_stages is None:
        x0 = _prepare_x_bias(x_bias, starts, xscale)
        x0 += rng.normal(0.0, noise, size=x0.shape)
        solver = DSB(qaia_graph, x=x0, n_iter=n_iter, batch_size=restarts, dt=dt, xi=_qaia_xi(qaia_graph))
        solver.y = rng.uniform(-0.01, 0.01, size=x0.shape)
        solver.update()
        refined = np.where(solver.x >= 0.0, 1, -1).astype(np.int8)
        if return_starts:
            return refined, starts
        return refined[:, 0]

    xi = _qaia_xi(qaia_graph)
    refined = starts
    solver = None
    for stage in anneal_stages:
        stage_xscale = float(stage.get('xscale', xscale))
        stage_noise = float(stage.get('noise', noise))
        stage_iter = max(1, int(stage.get('n_iter', n_iter)))
        stage_dt = float(stage.get('dt', dt))
        stage_bias = x_bias if solver is None else None
        x0 = _prepare_x_bias(stage_bias, refined, stage_xscale)
        x0 = x0 + rng.normal(0.0, stage_noise, size=x0.shape)

        if solver is None:
            solver = DSB(qaia_graph, x=x0, n_iter=stage_iter, batch_size=restarts, dt=stage_dt, xi=xi)
            solver.y = rng.uniform(-0.01, 0.01, size=x0.shape)
        else:
            solver.x = x0
            solver.y = rng.uniform(-0.01, 0.01, size=x0.shape)
            solver.n_iter = stage_iter
            solver.dt = stage_dt
            solver.p = np.linspace(0, 1, stage_iter)

        solver.update()
        refined = np.where(solver.x >= 0.0, 1, -1).astype(np.int8)

    if return_starts:
        return refined, starts
    return refined[:, 0]


def _run_qaia_candidate(
    label, warm_partition, qaia_graph, G_csr, graph_sum, rows, cols, weights, baseline, best,
    noise=0.006, restarts=1, qaia_params=None, full_refine_params=None,
    start_time=None, time_budget=None, x_bias=None
):
    if _time_exceeded(start_time, time_budget):
        return best

    params = qaia_params or {}
    noise = float(params.get('noise', noise))
    restarts = int(params.get('restarts', restarts))
    xscale = float(params.get('xscale', 0.03))
    n_iter = int(params.get('n_iter', 8))
    dt = float(params.get('dt', 0.2))
    anneal_stages = params.get('anneal_stages')
    post_anneal_stages = params.get('post_anneal_stages')
    local_refine_params = params.get('local_refine')
    jitter_rate = float(params.get('jitter_rate', 0.0))

    rng = np.random.default_rng(_stable_seed("candidate", label))
    qaia_partitions, jittered_starts = _qaia_dsb_refine(
        qaia_graph, warm_partition, xscale=xscale, noise=noise, n_iter=n_iter, restarts=restarts,
        jitter_rate=jitter_rate, rng=rng, return_starts=True, dt=dt, anneal_stages=anneal_stages,
        x_bias=x_bias
    )

    if qaia_partitions.shape[1] == 1:
        col = 0
        partition = qaia_partitions[:, 0]
        initial_cuts, initial_fields = _cut_values_and_fields_from_graph_batch(
            G_csr, graph_sum, np.column_stack((partition, jittered_starts[:, 0]))
        )
        cut_value = float(initial_cuts[0])
        selected_field = initial_fields[:, 0]
        dsb_cut = cut_value
        start_cut = float(initial_cuts[1])
        dsb_gain = dsb_cut - start_cut
        if _target_reached(cut_value, baseline):
            return _update_best(best, cut_value, partition, f"{label}-dsb-r0-dsbGain{dsb_gain:.3g}-target")
    else:
        initial_cuts, initial_fields = _cut_values_and_fields_from_graph_batch(
            G_csr, graph_sum, np.column_stack((qaia_partitions, jittered_starts))
        )
        restart_count = qaia_partitions.shape[1]
        cut_values = initial_cuts[:restart_count]
        start_values = initial_cuts[restart_count:]

        if baseline is not None and baseline > 0:
            target_cols = np.flatnonzero(cut_values >= baseline - 1e-4)
            if target_cols.size:
                best_target = int(target_cols[np.argmax(cut_values[target_cols])])
                partition = qaia_partitions[:, best_target]
                cut_value = float(cut_values[best_target])
                selected_field = initial_fields[:, best_target]
                start_cut = float(start_values[best_target])
                dsb_gain = cut_value - start_cut
                candidate_label = f"{label}-dsb-r{best_target}-dsbGain{dsb_gain:.3g}-target"
                return _update_best(best, cut_value, partition, candidate_label)

        col = int(np.argmax(cut_values))
        partition = qaia_partitions[:, col]
        cut_value = float(cut_values[col])
        selected_field = initial_fields[:, col]
        dsb_cut = cut_value
        start_cut = float(start_values[col])
        dsb_gain = dsb_cut - start_cut

    if _time_exceeded(start_time, time_budget):
        return _update_best(best, cut_value, partition, f"{label}-dsb-r{col}")

    if (
        post_anneal_stages
        and not _target_reached(cut_value, baseline)
        and not _time_exceeded(start_time, time_budget)
    ):
        annealed_partitions, _ = _qaia_dsb_refine(
            qaia_graph, partition, xscale=xscale, noise=noise, n_iter=n_iter, restarts=1,
            rng=rng, return_starts=True, dt=dt, anneal_stages=post_anneal_stages
        )
        annealed_partition = annealed_partitions[:, 0]
        annealed_cut = _cut_value_from_graph(G_csr, graph_sum, annealed_partition)
        if annealed_cut > cut_value:
            partition = annealed_partition
            cut_value = annealed_cut
        if _target_reached(cut_value, baseline):
            anneal_gain = cut_value - dsb_cut
            return _update_best(
                best, cut_value, partition,
                f"{label}-anneal-r{col}-dsbGain{dsb_gain:.3g}-annealGain{anneal_gain:.3g}",
            )

    local_gain = 0.0
    if local_refine_params and not _target_reached(cut_value, baseline):
        local_rng = np.random.default_rng(
            _stable_seed(
                "local-tune",
                1.0,
                int(local_refine_params.get('active_size', 8)),
                int(local_refine_params.get('n_iter', 2)),
                float(local_refine_params.get('xscale', 0.02)),
                float(local_refine_params.get('noise_mult', 0.0)),
                float(local_refine_params.get('dt', 0.18)),
                int(local_refine_params.get('restarts', 8)),
            )
        )
        local_partition, local_cut, local_gain = _qaia_local_refine(
            G_csr, partition, graph_sum, cut_value, local_refine_params, local_rng,
            graph_product=selected_field,
        )
        if local_cut > cut_value:
            partition = local_partition
            cut_value = local_cut
        if _target_reached(cut_value, baseline):
            return _update_best(
                best,
                cut_value,
                partition,
                f"{label}-local-r{col}-dsbGain{dsb_gain:.3g}-localGain{local_gain:.3g}-target",
            )

    post_gain = cut_value - dsb_cut
    candidate_label = (
        f"{label}-dsb-r{col}-dsbGain{dsb_gain:.3g}"
        f"-localGain{local_gain:.3g}-postGain{post_gain:.3g}"
    )
    best = _update_best(best, cut_value, partition, candidate_label)
    return best


'''
必须基于 MindQuantum 量子计算框架实现。允许调用 MindQuantum 官方算法库，或基于该框架编写自定义逻辑。
'''
def maxcut_solver(
    G_csr, device='cpu', max_iterations=5, cut_value_baseline=None
):
    """
    Solves the MAX-CUT problem for a single graph instance.
    Returns:
        partition (numpy.ndarray): The best partition found (+1/-1 array of shape [N]).
    """
    num_nodes = G_csr.shape[0]
    time_budget = None
    start_time = None

    rows, cols, weights = _extract_original_edges(G_csr)
    if len(weights) == 0:
        return np.ones(num_nodes, dtype=np.int8)

    graph_sum = -2.0 * float(np.sum(weights))
    best = None

    if time_budget is None:
        features = _basic_graph_features(num_nodes, len(weights))
    else:
        features = _graph_features(num_nodes, weights)
    solver_params = _dynamic_solver_params(features, time_budget)
    solver_params['batch_randomized'] = False
    qaia_params = dict(solver_params['qaia'])
    qaia_params['local_refine'] = solver_params.get('local_refine')
    full_refine_params = solver_params['full_refine']
    positive_trials = solver_params['positive_trials']

    for quantile, max_components in positive_trials:
        if _time_exceeded(start_time, time_budget):
            return _best_partition_or_default(best, num_nodes)

        tree, num_components = _positive_quantile_tree(
            G_csr, rows, cols, weights, quantile=quantile, max_components=max_components
        )
        if tree is None:
            continue

        warm_partition, qaia_graph, labels, num_components = _partition_and_graph_from_tree(
            G_csr, tree, tree_data_is_original=True, return_labels=True, force_flip_tree=True
        )
        warm_partition = _orient_components(rows, cols, weights, warm_partition, labels, num_components)

        if _time_exceeded(start_time, time_budget):
            return _best_partition_or_default(best, num_nodes)

        best = _run_qaia_candidate(
            f"dsb-pos-q{int(quantile * 100)}-tree", warm_partition, qaia_graph, G_csr, graph_sum,
            rows, cols, weights, cut_value_baseline, best, qaia_params=qaia_params,
            full_refine_params=full_refine_params, start_time=start_time, time_budget=time_budget
        )
        if _time_exceeded(start_time, time_budget):
            return _best_partition_or_default(best, num_nodes)
        if best is not None and _target_reached(best[0], cut_value_baseline):
            return best[1]
        if best is not None:
            break

    # One light abs-weight perturbation fixes the rare case where deterministic
    # tree ties pick the wrong high-confidence cycle.
    abs_weights = np.abs(weights)
    randomized_trials = solver_params['randomized_trials']
    for scale, quantile, restart_ids in randomized_trials:
        if _time_exceeded(start_time, time_budget):
            return _best_partition_or_default(best, num_nodes)
        if not solver_params.get('batch_randomized'):
            for restart_id in restart_ids:
                if _time_exceeded(start_time, time_budget):
                    return _best_partition_or_default(best, num_nodes)

                rng = np.random.default_rng(restart_id)
                label = f"dsb-abs-gumbel-{scale:g}"
                if quantile > 0:
                    threshold = _rank_quantile(abs_weights, quantile)
                    mask = abs_weights >= threshold
                    priority_rows = rows[mask]
                    priority_cols = cols[mask]
                    priority_abs = abs_weights[mask]
                    priority_cost = weights[mask]
                    label += f"-q{int(quantile * 100)}"
                else:
                    priority_rows = rows
                    priority_cols = cols
                    priority_abs = abs_weights
                    priority_cost = weights
                priority = priority_abs + rng.gumbel(0.0, scale, size=priority_abs.shape)
                priority_graph = sparse.coo_matrix((-priority, (priority_rows, priority_cols)), shape=G_csr.shape).tocsr()
                priority_tree = minimum_spanning_tree(priority_graph)
                cost_lookup = sparse.coo_matrix(
                    (-priority_cost, (priority_rows, priority_cols)), shape=G_csr.shape
                ).tocsr()
                warm_partition, qaia_graph = _partition_and_graph_from_tree(
                    G_csr, priority_tree, tree_data_is_original=False, cost_lookup=cost_lookup
                )

                if _time_exceeded(start_time, time_budget):
                    return _best_partition_or_default(best, num_nodes)

                best = _run_qaia_candidate(
                    label, warm_partition, qaia_graph, G_csr, graph_sum, rows, cols, weights,
                    cut_value_baseline, best, noise=qaia_params['noise'], restarts=1,
                    qaia_params=qaia_params, full_refine_params=full_refine_params,
                    start_time=start_time, time_budget=time_budget
                )
                if _time_exceeded(start_time, time_budget):
                    return _best_partition_or_default(best, num_nodes)
                if best is not None and _target_reached(best[0], cut_value_baseline):
                    return best[1]
            continue

        warm_partitions = []
        batch_graph = None
        for restart_id in restart_ids:
            rng = np.random.default_rng(restart_id)
            label = f"dsb-abs-gumbel-{scale:g}"
            if quantile > 0:
                threshold = _rank_quantile(abs_weights, quantile)
                mask = abs_weights >= threshold
                priority_rows = rows[mask]
                priority_cols = cols[mask]
                priority_abs = abs_weights[mask]
                priority_cost = weights[mask]
                label += f"-q{int(quantile * 100)}"
            else:
                priority_rows = rows
                priority_cols = cols
                priority_abs = abs_weights
                priority_cost = weights
            priority = priority_abs + rng.gumbel(0.0, scale, size=priority_abs.shape)
            priority_graph = sparse.coo_matrix((-priority, (priority_rows, priority_cols)), shape=G_csr.shape).tocsr()
            priority_tree = minimum_spanning_tree(priority_graph)
            cost_lookup = sparse.coo_matrix(
                (-priority_cost, (priority_rows, priority_cols)), shape=G_csr.shape
            ).tocsr()
            warm_partition, qaia_graph = _partition_and_graph_from_tree(
                G_csr, priority_tree, tree_data_is_original=False, cost_lookup=cost_lookup
            )
            warm_partitions.append(warm_partition)
            if batch_graph is None:
                batch_graph = qaia_graph

            if _time_exceeded(start_time, time_budget):
                return _best_partition_or_default(best, num_nodes)

        if not warm_partitions or batch_graph is None:
            continue

        warm_matrix = np.column_stack(warm_partitions).astype(np.int8, copy=False)
        batch_params = dict(qaia_params)
        batch_params['restarts'] = warm_matrix.shape[1]
        best = _run_qaia_candidate(
            label, warm_matrix, batch_graph, G_csr, graph_sum, rows, cols, weights,
            cut_value_baseline, best, noise=qaia_params['noise'], restarts=warm_matrix.shape[1],
            qaia_params=batch_params, full_refine_params=full_refine_params,
            start_time=start_time, time_budget=time_budget
        )
        if _time_exceeded(start_time, time_budget):
            return _best_partition_or_default(best, num_nodes)
        if best is not None and _target_reached(best[0], cut_value_baseline):
            return best[1]

    # Deterministic all-edge warm start is a conservative fallback for inputs
    # where the positive-edge compressed graph is not enough.
    if _time_exceeded(start_time, time_budget):
        return _best_partition_or_default(best, num_nodes)

    warm_partition, qaia_graph = _weight_tree_warm_start(G_csr, rows, cols, weights)
    best = _run_qaia_candidate(
        "dsb-weight-tree", warm_partition, qaia_graph, G_csr, graph_sum, rows, cols, weights,
        cut_value_baseline, best, qaia_params=qaia_params, full_refine_params=full_refine_params,
        start_time=start_time, time_budget=time_budget
    )
    if _time_exceeded(start_time, time_budget):
        return _best_partition_or_default(best, num_nodes)
    if _target_reached(best[0], cut_value_baseline):
        return best[1]

    if _time_exceeded(start_time, time_budget):
        return _best_partition_or_default(best, num_nodes)

    priority_graph = sparse.coo_matrix((-abs_weights, (rows, cols)), shape=G_csr.shape).tocsr()
    priority_tree = minimum_spanning_tree(priority_graph)
    cost_lookup = sparse.coo_matrix((-weights, (rows, cols)), shape=G_csr.shape).tocsr()
    warm_partition, qaia_graph = _partition_and_graph_from_tree(
        G_csr, priority_tree, tree_data_is_original=False, cost_lookup=cost_lookup
    )
    best = _run_qaia_candidate(
        "dsb-abs-tree", warm_partition, qaia_graph, G_csr, graph_sum, rows, cols, weights, cut_value_baseline, best,
        noise=qaia_params['noise'], qaia_params=qaia_params, full_refine_params=full_refine_params,
        start_time=start_time, time_budget=time_budget
    )
    if _time_exceeded(start_time, time_budget):
        return _best_partition_or_default(best, num_nodes)
    if _target_reached(best[0], cut_value_baseline):
        return best[1]

    return _best_partition_or_default(best, num_nodes)


if __name__ == "__main__":
    ALPHA = 1000.0 # 禁止改动
    BETA = 1.0  # 禁止改动
    
    dataset = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path_pattern = os.path.join(script_dir, 'Graph_data', '*.txt')
    filelist = glob(data_path_pattern)

    for filename in filelist:
        try:
            G, cut_base, time_base = read_graph_file(filename, negate=True)
            num_nodes = G.shape[0] # type: ignore
            num_edges = G.nnz // 2
            dataset.append({
                'G': G,
                'num_nodes': num_nodes,
                'filename': filename,
                'cut_base': cut_base,
                'time_base': time_base
            })
        except Exception as e:
            print(f"Skipping file {filename}: {e}")
    
    if not dataset:
        print("No valid datasets found.")
        exit()
    
    results = []
    total_weighted_accuracy = 0.0
    total_weight = 0.0
    
    global_solve_time = 0.0
    global_time_base = 0.0


    for idx, data_item in enumerate(dataset):
        G = data_item['G']
        cut_base = data_item['cut_base']
        time_base = data_item['time_base']
        filename = data_item['filename']
        graph_name = os.path.basename(filename)
        nodes = data_item['num_nodes']
        
        print(f"\n[{idx+1}/{len(dataset)}] Solving {graph_name} ({nodes} nodes)...")
        
        start_time = time.time()
        try:
            partition = maxcut_solver(G, max_iterations=5, cut_value_baseline=cut_base)
            solve_time = time.time() - start_time
            cut_value = calculate_cut_value(scipy_to_torch_sparse(G), partition)
            cut_val_rounded = round(cut_value, 5)
            cut_base_rounded = round(cut_base, 5)
            
            if cut_base == 0:
                acc_ratio = 1.0 if cut_value == 0 else 0.0
            else:
                acc_ratio = cut_val_rounded / cut_base_rounded
            
            acc_ratio = min(max(acc_ratio, 0), 1.05)
            accuracy_score = np.exp(ALPHA * np.log(acc_ratio + 1e-10))
            
            weight = nodes / 1000.0
            
            total_weighted_accuracy += accuracy_score * weight
            total_weight += weight
            
            global_solve_time += solve_time
            global_time_base += time_base
            
            print(f"      Cut: {cut_value:.5f}/{cut_base:.5f} (AccRatio: {acc_ratio:.5%})")
            print(f"      Time: {solve_time:.2f}s (Base: {time_base:.2f}s)")
            
            results.append({
                'filename': graph_name,
                'nodes': nodes,
                'cut_val': cut_value,
                'cut_base': cut_base,
                'acc_ratio': acc_ratio,
                'time_val': solve_time,
                'weight': weight
            })
            
        except Exception as e:
             print(f"      Error: {e}")
             import traceback
             traceback.print_exc()

    final_acc_score = total_weighted_accuracy / total_weight if total_weight > 0 else 0.0

    if global_time_base <= 0: global_time_base = 0.1
    global_ratio = global_solve_time / global_time_base
    
    time_penalty = np.log10(1 + global_ratio)
    final_time_score = 1.0 / (1.0 + BETA * time_penalty)
    
    final_total_score = final_acc_score * final_time_score * 100.0

    print("\n" + "="*110)
    print("FINAL RESULTS SUMMARY")
    print("="*110)
    print(f"{'File':<25} {'Nodes':<8} {'Cut/Base':<15} {'AccRatio':<10} {'Time(s)':<10} {'GlobalContr.':<12}")
    print("-" * 110)
    
    for r in results:
        cut_str = f"{r['cut_val']:.5f}/{r['cut_base']:.5f}"
        time_contribution = (r['time_val'] / global_solve_time * 100) if global_solve_time > 0 else 0
        
        print(f"{r['filename']:<25} {r['nodes']:<8} {cut_str:<15} {r['acc_ratio']:<10.5f} {r['time_val']:<10.2f} {time_contribution:>5.1f}%")
    
    print("-" * 110)
    print(f"1. Total Nodes (Sum Weight):   {total_weight * 1000:.0f}")
    print(f"2. Weighted Accuracy Score:    {final_acc_score:.5f} (Alpha: {ALPHA})")
    print(f"3. Total Solve Time:           {global_solve_time:.2f} s")
    print(f"4. Total Time Baseline:        {global_time_base:.2f} s")
    print(f"5. Global Time Ratio:          {global_ratio:.2f} x")
    print(f"6. Time Efficiency Score:      {final_time_score:.2f} (Beta: {BETA})")
    print("=" * 110)
    print(f"FINAL SCORE: {final_total_score:.5f}")
    print("=" * 110) 
