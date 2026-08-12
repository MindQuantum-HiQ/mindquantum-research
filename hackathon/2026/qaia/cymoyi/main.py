import atexit
import math
import os
import time
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from scipy import sparse
from scipy.sparse import csr_matrix, coo_matrix
from qaia import BSB, DSB, NMFA


np.random.seed(666)

QAIA_TRACE = False
QAIA_PROFILE = False
WINDOW_QAIA_WORKERS = 4
WINDOW_DENSE_BATCH = 35
LOCAL_REPAIR_FLIP_CAP = 16
PRIMARY_FAST_KERNEL = os.environ.get('QAIA_PRIMARY_FAST_KERNEL', 'nmfa')
PRIMARY_FAST_EXIT = os.environ.get('QAIA_PRIMARY_FAST_EXIT', '0') == '1'
PRIMARY_FAST_EXIT_RATIO = float(os.environ.get('QAIA_PRIMARY_FAST_EXIT_RATIO', '1.0'))
PRIMARY_PROBE_TARGET_MARGIN = float(os.environ.get('QAIA_PRIMARY_PROBE_TARGET_MARGIN', '0.0'))
PRIMARY_PROBE_ADAPTIVE_STITCH = os.environ.get('QAIA_PRIMARY_PROBE_ADAPTIVE_STITCH', '0') == '1'
PRIMARY_PROBE_REPAIR = os.environ.get('QAIA_PRIMARY_PROBE_REPAIR', '0') == '1'
PRIMARY_REPAIR_NODE_LIMIT = int(os.environ.get('QAIA_REPAIR_NODE_LIMIT', '512'))
PRIMARY_REPAIR_BATCH_SIZE = int(os.environ.get('QAIA_REPAIR_BATCH_SIZE', '8'))
PRIMARY_REPAIR_N_ITER = int(os.environ.get('QAIA_REPAIR_N_ITER', '36'))
PRIMARY_REPAIR_ROUNDS = int(os.environ.get('QAIA_REPAIR_ROUNDS', '1'))
PRIMARY_REPAIR_ALGO = os.environ.get('QAIA_REPAIR_ALGO', 'nmfa')
PRIMARY_REPAIR_CANDIDATES = int(os.environ.get('QAIA_REPAIR_CANDIDATES', '2'))
PRIMARY_REPAIR_EXPAND = float(os.environ.get('QAIA_REPAIR_EXPAND', '0.35'))
PRIMARY_REPAIR_LOCAL_FLIPS = int(os.environ.get('QAIA_REPAIR_LOCAL_FLIPS', '256'))
PRIMARY_BOUNDARY_REPAIR = os.environ.get('QAIA_BOUNDARY_REPAIR', '0') == '1'
PRIMARY_BOUNDARY_REPAIR_TOPK = int(os.environ.get('QAIA_BOUNDARY_REPAIR_TOPK', '4'))
PRIMARY_BOUNDARY_REPAIR_RADIUS = int(os.environ.get('QAIA_BOUNDARY_REPAIR_RADIUS', '96'))
PRIMARY_BOUNDARY_REPAIR_N_ITER = int(os.environ.get('QAIA_BOUNDARY_REPAIR_N_ITER', '80'))
PRIMARY_BOUNDARY_REPAIR_BATCH = int(os.environ.get('QAIA_BOUNDARY_REPAIR_BATCH', '8'))

PRIMARY_QMF_N_ITER = int(os.environ.get('QAIA_QMF_N_ITER', '48'))
PRIMARY_QMF_ALPHA = float(os.environ.get('QAIA_QMF_ALPHA', '0.35'))
PRIMARY_QMF_SIGMA = float(os.environ.get('QAIA_QMF_SIGMA', '0.05'))
PRIMARY_QMF_BETA_FINAL = float(os.environ.get('QAIA_QMF_BETA_FINAL', '1.35'))

PRIMARY_AMF_N_ITER = int(os.environ.get('QAIA_AMF_N_ITER', '24'))
PRIMARY_AMF_ALPHA = float(os.environ.get('QAIA_AMF_ALPHA', '0.30'))
PRIMARY_AMF_SIGMA = float(os.environ.get('QAIA_AMF_SIGMA', '0.12'))
PRIMARY_AMF_BETA_FINAL = float(os.environ.get('QAIA_AMF_BETA_FINAL', '1.60'))
PRIMARY_AMF_DIVERSE = os.environ.get('QAIA_AMF_DIVERSE', '0') == '1'
PRIMARY_AMF_CHECKPOINTS = os.environ.get('QAIA_AMF_CHECKPOINTS', '1') == '1'
PRIMARY_AMF_TOPK = int(os.environ.get('QAIA_AMF_TOPK', '1'))
PRIMARY_AMF_HARDEN_FRAC = float(os.environ.get('QAIA_AMF_HARDEN_FRAC', '0.0'))
PRIMARY_AMF_TAIL_N_ITER = int(os.environ.get('QAIA_AMF_TAIL_N_ITER', '0'))
PRIMARY_AMF_TAIL_SIGMA = float(os.environ.get('QAIA_AMF_TAIL_SIGMA', '0.0'))
PRIMARY_AMF_ENSEMBLE = int(os.environ.get('QAIA_AMF_ENSEMBLE', '1'))
PRIMARY_AMF_BETA_POWER = float(os.environ.get('QAIA_AMF_BETA_POWER', '1.0'))
PRIMARY_AMF_FUSION = os.environ.get('QAIA_AMF_FUSION', '0') == '1'
PRIMARY_AMF_FUSION_MODE = os.environ.get('QAIA_AMF_FUSION_MODE', 'amf')
PRIMARY_AMF_FUSION_TOPK = int(os.environ.get('QAIA_AMF_FUSION_TOPK', '4'))
PRIMARY_AMF_FUSION_BATCH = int(os.environ.get('QAIA_AMF_FUSION_BATCH', '4'))
PRIMARY_AMF_FUSION_N_ITER = int(os.environ.get('QAIA_AMF_FUSION_N_ITER', '18'))
PRIMARY_AMF_FUSION_ALPHA = float(os.environ.get('QAIA_AMF_FUSION_ALPHA', '0.45'))
PRIMARY_AMF_FUSION_SIGMA = float(os.environ.get('QAIA_AMF_FUSION_SIGMA', '0.10'))
PRIMARY_AMF_FUSION_BETA_FINAL = float(os.environ.get('QAIA_AMF_FUSION_BETA_FINAL', '2.0'))
PRIMARY_AMF_FUSION_SELECT_FINAL = float(os.environ.get('QAIA_AMF_FUSION_SELECT_FINAL', '4.0'))
PRIMARY_AMF_FUSION_SEED = int(os.environ.get('QAIA_AMF_FUSION_SEED', '17011'))
PRIMARY_AMF_BLOCK_FUSION_PASSES = int(os.environ.get('QAIA_AMF_BLOCK_FUSION_PASSES', '2'))
PRIMARY_AMF_SELECTIVE_NMFA = os.environ.get('QAIA_AMF_SELECTIVE_NMFA', '0') == '1'
PRIMARY_AMF_SELECTIVE_NMFA_TOPK = int(os.environ.get('QAIA_AMF_SELECTIVE_NMFA_TOPK', '12'))
PRIMARY_AMF_SELECTIVE_NMFA_BATCH = int(os.environ.get('QAIA_AMF_SELECTIVE_NMFA_BATCH', '8'))
PRIMARY_AMF_SELECTIVE_NMFA_N_ITER = int(os.environ.get('QAIA_AMF_SELECTIVE_NMFA_N_ITER', '98'))
PRIMARY_AMF_SELECTIVE_NMFA_SCORE = os.environ.get('QAIA_AMF_SELECTIVE_NMFA_SCORE', 'unstable')
PRIMARY_AMF_SELECTIVE_NMFA_STAGES = os.environ.get('QAIA_AMF_SELECTIVE_NMFA_STAGES', '')
PRIMARY_AMF_SELECTIVE_NMFA_MIN_RATIO = float(os.environ.get('QAIA_AMF_SELECTIVE_NMFA_MIN_RATIO', '0.9996'))
PRIMARY_AMF_SELECTIVE_NMFA_WARM_START = os.environ.get('QAIA_AMF_SELECTIVE_NMFA_WARM_START', '0') == '1'
PRIMARY_AMF_SELECTIVE_NMFA_WARM_SIGMA = float(os.environ.get('QAIA_AMF_SELECTIVE_NMFA_WARM_SIGMA', '0.08'))
PRIMARY_AMF_SELECTIVE_NMFA_WARM_SCALE = float(os.environ.get('QAIA_AMF_SELECTIVE_NMFA_WARM_SCALE', '0.03'))
PRIMARY_AMF_SINGLE_WINDOW_REPAIR = os.environ.get('QAIA_AMF_SINGLE_WINDOW_REPAIR', '0') == '1'
PRIMARY_AMF_SINGLE_WINDOW_LIMIT = int(os.environ.get('QAIA_AMF_SINGLE_WINDOW_LIMIT', '36'))
PRIMARY_HYBRID_NMFA_N_ITER = int(os.environ.get('QAIA_HYBRID_NMFA_N_ITER', '24'))
PRIMARY_HYBRID_NMFA_ALPHA = float(os.environ.get('QAIA_HYBRID_NMFA_ALPHA', '0.15'))
PRIMARY_HYBRID_NMFA_SIGMA = float(os.environ.get('QAIA_HYBRID_NMFA_SIGMA', '0.06'))
PRIMARY_HYBRID_NMFA_BETA_FINAL = float(os.environ.get('QAIA_HYBRID_NMFA_BETA_FINAL', '1.00'))
PRIMARY_HYBRID_TOPK = int(os.environ.get('QAIA_HYBRID_TOPK', '1'))

PRIMARY_DSB_N_ITER = int(os.environ.get('QAIA_DSB_N_ITER', '40'))
PRIMARY_DSB_DT = float(os.environ.get('QAIA_DSB_DT', '0.90'))
PRIMARY_DSB_XI_SCALE = float(os.environ.get('QAIA_DSB_XI_SCALE', '0.95'))
PRIMARY_DSB_TOPK = int(os.environ.get('QAIA_DSB_TOPK', '1'))

PRIMARY_SIMCIM_N_ITER = int(os.environ.get('QAIA_SIMCIM_N_ITER', '48'))
PRIMARY_SIMCIM_DT = float(os.environ.get('QAIA_SIMCIM_DT', '0.015'))
PRIMARY_SIMCIM_MOMENTUM = float(os.environ.get('QAIA_SIMCIM_MOMENTUM', '0.90'))
PRIMARY_SIMCIM_SIGMA = float(os.environ.get('QAIA_SIMCIM_SIGMA', '0.03'))
PRIMARY_SIMCIM_PT = float(os.environ.get('QAIA_SIMCIM_PT', '6.5'))
PRIMARY_SIMCIM_TOPK = int(os.environ.get('QAIA_SIMCIM_TOPK', '1'))
AQMF_ENABLED = os.environ.get('QAIA_AQMF', '1') == '1'
AQMF_SWEEP_ROUNDS = int(os.environ.get('QAIA_AQMF_SWEEP_ROUNDS', '1'))
AQMF_LOCAL_REPAIR_FLIPS = int(os.environ.get('QAIA_AQMF_LOCAL_REPAIR_FLIPS', '16'))
AQMF_NMFA_COMPETE = os.environ.get('QAIA_AQMF_NMFA_COMPETE', '0') == '1'
AQMF_NMFA_BATCH = int(os.environ.get('QAIA_AQMF_NMFA_BATCH', '3'))
AQMF_NMFA_N_ITER = int(os.environ.get('QAIA_AQMF_NMFA_N_ITER', '24'))
AQMF_NMFA_ALPHA = float(os.environ.get('QAIA_AQMF_NMFA_ALPHA', '0.15'))
AQMF_NMFA_SIGMA = float(os.environ.get('QAIA_AQMF_NMFA_SIGMA', '0.08'))
AQMF_NMFA_X_SCALE = float(os.environ.get('QAIA_AQMF_NMFA_X_SCALE', '0.04'))
AQMF_NMFA_SEED = int(os.environ.get('QAIA_AQMF_NMFA_SEED', '24017'))
AQMF_QAIA_REPAIR_ENABLED = os.environ.get('QAIA_AQMF_QAIA_REPAIR', '1') == '1'
AQMF_QAIA_REPAIR_NODE_LIMIT = int(os.environ.get('QAIA_AQMF_QAIA_REPAIR_NODE_LIMIT', '256'))
AQMF_QAIA_REPAIR_BATCH_SIZE = int(os.environ.get('QAIA_AQMF_QAIA_REPAIR_BATCH_SIZE', '4'))
AQMF_QAIA_REPAIR_N_ITER = int(os.environ.get('QAIA_AQMF_QAIA_REPAIR_N_ITER', '20'))
AQMF_QAIA_REPAIR_LOCAL_FLIPS = int(os.environ.get('QAIA_AQMF_QAIA_REPAIR_LOCAL_FLIPS', '0'))
AQMF_QAIA_REPAIR_MIN_GAP = float(os.environ.get('QAIA_AQMF_QAIA_REPAIR_MIN_GAP', '80.0'))
SELECTIVE_REFINE_ENABLED = os.environ.get('QAIA_SELECTIVE_REFINE', '1') == '1'
SELECTIVE_REFINE_GAP_ABS = float(os.environ.get('QAIA_SELECTIVE_REFINE_GAP_ABS', '8.0'))
SELECTIVE_REFINE_GAP_RATIO = float(os.environ.get('QAIA_SELECTIVE_REFINE_GAP_RATIO', '8e-5'))
SELECTIVE_REFINE_MAX = int(os.environ.get('QAIA_SELECTIVE_REFINE_MAX', '2'))

_WINDOW_EXECUTOR = None
_WINDOW_EXECUTOR_WORKERS = 0


def _profile_start():
    return time.perf_counter() if QAIA_PROFILE else 0.0


def _profile_ms(started):
    return (time.perf_counter() - started) * 1000.0 if QAIA_PROFILE else 0.0


def _profile_value(value):
    if value is None:
        return 'none'
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if np.isneginf(value):
            return '-inf'
        if np.isposinf(value):
            return 'inf'
        return f'{value:.6f}'
    text = str(value)
    return text.replace(' ', '')


def _profile_emit(stage, **fields):
    if not QAIA_PROFILE:
        return
    parts = [f'stage={stage}']
    parts.extend(f'{key}={_profile_value(value)}' for key, value in fields.items())
    print('QAIA_PROFILE ' + ' '.join(parts), flush=True)


def _get_window_executor():
    """Reuse one thread pool across graphs to avoid per-instance setup overhead."""
    global _WINDOW_EXECUTOR, _WINDOW_EXECUTOR_WORKERS
    if WINDOW_QAIA_WORKERS <= 1:
        return None
    if _WINDOW_EXECUTOR is None or _WINDOW_EXECUTOR_WORKERS != WINDOW_QAIA_WORKERS:
        if _WINDOW_EXECUTOR is not None:
            _WINDOW_EXECUTOR.shutdown(wait=True)
        _WINDOW_EXECUTOR = ThreadPoolExecutor(max_workers=WINDOW_QAIA_WORKERS)
        _WINDOW_EXECUTOR_WORKERS = WINDOW_QAIA_WORKERS
    return _WINDOW_EXECUTOR


def _shutdown_window_executor():
    global _WINDOW_EXECUTOR, _WINDOW_EXECUTOR_WORKERS
    if _WINDOW_EXECUTOR is not None:
        _WINDOW_EXECUTOR.shutdown(wait=True)
        _WINDOW_EXECUTOR = None
        _WINDOW_EXECUTOR_WORKERS = 0


atexit.register(_shutdown_window_executor)


'''
必须基于 MindQuantum 量子计算框架实现。允许调用 MindQuantum 官方算法库，或基于该框架编写自定义逻辑。
'''
class SB:
    """
    Simulated Bifurcation (SB) Algorithm Class.
    """
    
    def __init__(self, A, h=0., tabu=None, dt=1., n_iter=1000, xi=None, 
                 batch_size=10, num_tabu=2, device='cpu'):
        self.N = A.shape[0]
        self.A = A
        self.h = h
        self.tabu = tabu
        self.batch_size = batch_size
        self.dt = dt
        self.n_iter = n_iter
        self.device = device
        self.p = np.linspace(0, 1, self.n_iter)
        self.num_tabu = num_tabu
        
        if xi is None:
            self.xi = 1 / torch.abs(torch.sparse.sum(A, dim=1).to_dense()).max()
        else:
            self.xi = xi
        
        self.initialize()
    
    def initialize(self):
        self.x = 0.01 * (torch.rand(self.N, self.batch_size, device=self.device) - 0.5)
        self.y = 0.01 * (torch.rand(self.N, self.batch_size, device=self.device) - 0.5)
    
    def update_b(self, beta=1):
        for i in range(self.n_iter):
            if self.tabu is None:
                force = self.xi * (torch.sparse.mm(self.A, self.x) + self.h)
            else:
                num_tabu_sample = self.tabu.shape[1]
                if num_tabu_sample > 1:
                    num_tabu = self.num_tabu
                    tabu_index = np.random.randint(0, num_tabu_sample, num_tabu)
                    t = beta * (self.tabu[:, tabu_index].sum(dim=1, keepdim=True) / num_tabu)
                    force = self.xi * (torch.sparse.mm(self.A, self.x) + self.h - t)
                else:
                    force = self.xi * (torch.sparse.mm(self.A, self.x) + self.h - self.tabu)

            self.y += (-(1 - self.p[i]) * self.x + force) * self.dt
            self.x += self.dt * self.y
            
            cond = torch.abs(self.x) > 1
            self.x = torch.where(cond, torch.sign(self.x), self.x)
            self.y = torch.where(cond, torch.zeros_like(self.y), self.y)


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


def _cut_value_numpy(G_csr, partition, graph_sum=None):
    """Fast cut evaluation equivalent to the official formula."""
    if graph_sum is None:
        graph_sum = float(G_csr.sum())
    spins = partition.astype(np.float32, copy=False)
    return float(0.25 * np.dot(spins, G_csr.dot(spins)) - 0.25 * graph_sum)


def _normalize_partition(spins):
    """Convert arbitrary real-valued spins to a 1-D {-1, 1} partition."""
    spins = np.sign(np.asarray(spins)).astype(np.int8, copy=False).reshape(-1)
    spins[spins == 0] = 1
    return spins


def _normalize_candidate_matrix(spins):
    """Convert solver output to an (N, batch) {-1, 1} matrix."""
    spins = np.asarray(spins)
    if spins.ndim == 1:
        spins = spins[:, None]
    signs = np.sign(spins).astype(np.int8, copy=False)
    signs[signs == 0] = 1
    return signs


def _cut_value_from_state(spins_f, field, graph_sum):
    """Evaluate the cut from an already computed local-search field state."""
    return float(0.25 * np.dot(spins_f, field) - 0.25 * graph_sum)


def _build_local_search_state(G_csr, partition):
    """Prepare spins, local field and 1-flip gains for incremental search."""
    spins = _normalize_partition(partition).astype(np.int8, copy=True)
    spins_f = spins.astype(np.float32)
    field = G_csr.dot(spins_f)
    gain = -spins_f * field
    return spins, spins_f, field, gain


def _apply_local_flip(G_csr, spins, spins_f, field, gain, node):
    """Flip one node and update the incremental local-search state."""
    old_spin = int(spins[node])
    step_gain = float(gain[node])
    spins[node] = np.int8(-old_spin)
    spins_f[node] = -spins_f[node]
    delta = np.float32(-2 * old_spin)

    start, end = G_csr.indptr[node], G_csr.indptr[node + 1]
    neigh = G_csr.indices[start:end]
    field[neigh] += G_csr.data[start:end] * delta
    gain[neigh] = -spins_f[neigh] * field[neigh]
    gain[node] = -spins_f[node] * field[node]
    return step_gain


def _steepest_ascent_local_search(
    G_csr,
    initial_partition,
    max_flips=None,
    graph_sum=None,
    target_cut=None,
):
    """
    Greedy 1-flip ascent with incremental gain updates.

    When ``max_flips`` is ``None`` the search runs to a full 1-flip local
    optimum; otherwise it behaves as a bounded repair pass.
    """
    if graph_sum is None:
        graph_sum = float(G_csr.sum())

    spins = _normalize_partition(initial_partition).astype(np.int8, copy=True)
    spins_f = spins.astype(np.float32)
    field = G_csr.dot(spins_f)
    current_cut = _cut_value_from_state(spins_f, field, graph_sum)
    flips = 0

    if max_flips is not None and max_flips <= 0:
        return spins.astype(np.int8, copy=False), current_cut, field, None, flips

    gain = -spins_f * field
    while max_flips is None or flips < max_flips:
        node = int(np.argmax(gain))
        best_gain = float(gain[node])
        if best_gain <= 1e-7:
            break

        current_cut += _apply_local_flip(G_csr, spins, spins_f, field, gain, node)
        flips += 1
        if target_cut is not None and current_cut >= target_cut:
            break

    if flips:
        current_cut = _cut_value_from_state(spins_f, field, graph_sum)
    return spins.astype(np.int8, copy=False), current_cut, field, gain, flips


def _one_flip_local_search(G_csr, initial_partition, max_flips):
    """Compatibility wrapper for the submission's bounded 1-flip repair."""
    spins, _cut, _field, _gain, _flips = _steepest_ascent_local_search(
        G_csr,
        initial_partition,
        max_flips=max_flips,
    )
    return spins


def _collect_repaired_candidates(
    G_original,
    candidates,
    max_flips,
    graph_sum=None,
    target_cut=None,
    seen_final=None,
):
    """Score candidate batches on the original graph and keep repaired outputs."""
    if graph_sum is None:
        graph_sum = float(G_original.sum())
    if seen_final is None:
        seen_final = set()

    scored = []
    seen_raw = set()

    for item in candidates:
        if isinstance(item, dict):
            label = item['label']
            spin_values = item['spin_values']
        else:
            label, spin_values = item[:2]
        candidate_matrix = _normalize_candidate_matrix(spin_values)
        is_batched = candidate_matrix.shape[1] > 1

        for col in range(candidate_matrix.shape[1]):
            raw = candidate_matrix[:, col].astype(np.int8, copy=True)
            raw_key = raw.tobytes()
            if raw_key in seen_raw:
                continue
            seen_raw.add(raw_key)

            repaired, repaired_cut, _field, _gain, flips = _steepest_ascent_local_search(
                G_original,
                raw,
                max_flips=max_flips,
                graph_sum=graph_sum,
                target_cut=target_cut,
            )
            repaired_key = repaired.tobytes()
            if repaired_key in seen_final:
                continue
            seen_final.add(repaired_key)

            scored_item = {
                'label': f'{label}[{col}]' if is_batched else label,
                'partition': repaired.astype(np.int8, copy=False),
                'cut': float(repaired_cut),
                'repair_flips': int(flips),
            }

            if QAIA_TRACE:
                raw_cut = _cut_value_numpy(G_original, raw, graph_sum=graph_sum)
                scored_item['raw_cut'] = float(raw_cut)
                hamming = int(np.count_nonzero(raw != repaired))
                print(
                    f"{label}[{col}] raw={raw_cut:.5f} repaired={repaired_cut:.5f} "
                    f"gain={repaired_cut - raw_cut:.5f} flips={hamming}"
                )

            scored.append(scored_item)

    scored.sort(key=lambda item: item['cut'], reverse=True)
    return scored


def _score_and_repair_candidates(G_original, candidates, max_flips, graph_sum=None, target_cut=None):
    """Score QAIA candidates on the original graph and apply one bounded repair pass."""
    if len(candidates) == 1:
        item = candidates[0]
        label, spin_values = (item['label'], item['spin_values']) if isinstance(item, dict) else item[:2]
        candidate_matrix = _normalize_candidate_matrix(spin_values)
        if candidate_matrix.shape[1] == 1:
            if graph_sum is None:
                graph_sum = float(G_original.sum())
            raw = candidate_matrix[:, 0].astype(np.int8, copy=True)
            repaired, repaired_cut, _field, _gain, flips = _steepest_ascent_local_search(
                G_original,
                raw,
                max_flips=max_flips,
                graph_sum=graph_sum,
                target_cut=target_cut,
            )
            if QAIA_TRACE:
                raw_cut = _cut_value_numpy(G_original, raw, graph_sum=graph_sum)
                hamming = int(np.count_nonzero(raw != repaired))
                print(
                    f"{label} raw={raw_cut:.5f} repaired={repaired_cut:.5f} "
                    f"gain={repaired_cut - raw_cut:.5f} flips={hamming}"
                )
            return repaired.astype(np.int8, copy=False), float(repaired_cut)

    scored = _collect_repaired_candidates(
        G_original,
        candidates,
        max_flips=max_flips,
        graph_sum=graph_sum,
        target_cut=target_cut,
    )
    if scored:
        best = scored[0]
        return best['partition'].astype(np.int8, copy=False), float(best['cut'])

    fallback = np.ones(G_original.shape[0], dtype=np.int8)
    return fallback, _cut_value_numpy(G_original, fallback, graph_sum=graph_sum)


def _select_repair_nodes(G_csr, partition, limit, expand_frac=0.35):
    """Select unstable nodes for QAIA subproblem repair."""
    limit = max(1, min(int(limit), G_csr.shape[0]))
    spins, _spins_f, _field, gain = _build_local_search_state(G_csr, partition)
    positive = np.flatnonzero(gain > 1e-7)
    if positive.size >= limit:
        selected = positive[np.argpartition(gain[positive], positive.size - limit)[positive.size - limit:]]
    else:
        selected = positive
        remaining = limit - selected.size
        if remaining > 0:
            stability = np.abs(gain)
            if selected.size:
                stability[selected] = np.inf
            count = min(remaining, stability.size - selected.size)
            if count > 0:
                low = np.argpartition(stability, count - 1)[:count]
                selected = np.concatenate((selected, low))

    if selected.size == 0:
        return selected.astype(np.int64, copy=False)

    expand_count = int(max(0, limit - selected.size) * max(0.0, float(expand_frac)))
    if expand_count > 0:
        marker = np.zeros(G_csr.shape[0], dtype=bool)
        marker[selected] = True
        scores = {}
        indptr = G_csr.indptr
        indices = G_csr.indices
        data = G_csr.data
        for node in selected:
            start, end = indptr[node], indptr[node + 1]
            neigh = indices[start:end]
            weights = np.abs(data[start:end])
            for nb, weight in zip(neigh, weights):
                if not marker[nb]:
                    scores[nb] = scores.get(nb, 0.0) + float(weight)
        if scores:
            extra_items = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:expand_count]
            if extra_items:
                selected = np.concatenate((selected, np.fromiter((node for node, _ in extra_items), dtype=np.int64)))

    if selected.size > limit:
        score = np.abs(gain).copy()
        positive_mask = gain > 1e-7
        score[positive_mask] += np.abs(gain[positive_mask])
        order = np.argsort(score[selected])[::-1]
        selected = selected[order[:limit]]
    return np.unique(selected.astype(np.int64, copy=False))


def _build_repair_subproblem(G_csr, partition, nodes):
    """Build a subproblem whose objective is the full-graph cut restricted to selected nodes."""
    nodes = np.asarray(nodes, dtype=np.int64)
    subgraph = G_csr[nodes, :][:, nodes].tocsr()
    outside_field = G_csr[nodes, :].dot(partition.astype(np.float32, copy=False))
    inside_field = subgraph.dot(partition[nodes].astype(np.float32, copy=False))
    h = outside_field - inside_field
    return subgraph, h.astype(np.float32, copy=False)


def _run_nmfa_with_field_candidates(J, h, batch_size, n_iter, seed, x=None, alpha=0.15, sigma=0.12, x_scale=0.05):
    """Run NMFA on a subproblem with a linear field term."""
    np.random.seed(seed)
    x_init = _prepare_x_init(
        J.shape[0],
        batch_size,
        seed,
        x=x,
        noise_scale=x_scale,
        random_when_none=True,
    )
    solver = NMFA(J, h=h.reshape(-1, 1), x=x_init, n_iter=n_iter, batch_size=batch_size, alpha=alpha, sigma=sigma)
    solver.update()
    return _normalize_candidate_matrix(solver.x)


def _score_repair_subproblem_candidates(G_csr, partition, nodes, candidates, graph_sum, target_cut=None):
    """Score subproblem candidate assignments after embedding them in the full partition."""
    best_partition = partition.astype(np.int8, copy=True)
    best_cut = _cut_value_numpy(G_csr, best_partition, graph_sum=graph_sum)
    candidate_matrix = _normalize_candidate_matrix(candidates)
    seen = set()
    for col in range(candidate_matrix.shape[1]):
        trial = partition.astype(np.int8, copy=True)
        values = candidate_matrix[:, col].astype(np.int8, copy=False)
        for sign in (1, -1):
            assigned = values if sign == 1 else -values
            key = assigned.tobytes()
            if key in seen:
                continue
            seen.add(key)
            trial[nodes] = assigned
            trial_cut = _cut_value_numpy(G_csr, trial, graph_sum=graph_sum)
            if trial_cut > best_cut:
                best_cut = trial_cut
                best_partition = trial.copy()
                if target_cut is not None and best_cut >= target_cut:
                    return best_partition, best_cut
    return best_partition, best_cut


def _qaia_subproblem_repair(G_csr, partition, graph_sum, target_cut=None, seed=7001):
    """Repair AMF/qmf probe output by solving a selected induced subproblem with QAIA."""
    current = _normalize_partition(partition).astype(np.int8, copy=True)
    current_cut = _cut_value_numpy(G_csr, current, graph_sum=graph_sum)
    best_partition = current
    best_cut = current_cut
    rounds = max(1, int(PRIMARY_REPAIR_ROUNDS))
    for round_idx in range(rounds):
        nodes = _select_repair_nodes(
            G_csr,
            best_partition,
            PRIMARY_REPAIR_NODE_LIMIT,
            expand_frac=PRIMARY_REPAIR_EXPAND,
        )
        if nodes.size <= 1:
            break
        J_sub, h_sub = _build_repair_subproblem(G_csr, best_partition, nodes)
        x_seed = best_partition[nodes].astype(np.float32, copy=False)
        if PRIMARY_REPAIR_ALGO == 'bsb':
            xi = _resolve_sb_xi(J_sub, 1.0)
            candidates = _run_bsb_candidates(
                J_sub,
                batch_size=PRIMARY_REPAIR_BATCH_SIZE,
                n_iter=PRIMARY_REPAIR_N_ITER,
                seed=seed + round_idx * 37,
                x=x_seed,
                dt=0.90,
                xi=xi,
            )
        elif PRIMARY_REPAIR_ALGO == 'dsb':
            xi = _resolve_sb_xi(J_sub, 0.95)
            candidates = _run_dsb_candidates(
                J_sub,
                batch_size=PRIMARY_REPAIR_BATCH_SIZE,
                n_iter=PRIMARY_REPAIR_N_ITER,
                seed=seed + round_idx * 37,
                x=x_seed,
                dt=0.90,
                xi=xi,
            )
        else:
            candidates = _run_nmfa_with_field_candidates(
                J_sub,
                h_sub,
                batch_size=PRIMARY_REPAIR_BATCH_SIZE,
                n_iter=PRIMARY_REPAIR_N_ITER,
                seed=seed + round_idx * 37,
                x=x_seed,
                alpha=0.15,
                sigma=0.12,
                x_scale=0.03,
            )
        repaired, repaired_cut = _score_repair_subproblem_candidates(
            G_csr,
            best_partition,
            nodes,
            candidates,
            graph_sum=graph_sum,
            target_cut=target_cut,
        )
        if repaired_cut <= best_cut + 1e-7:
            break
        best_partition, best_cut = repaired, repaired_cut
        if target_cut is not None and best_cut >= target_cut:
            break

    if PRIMARY_REPAIR_LOCAL_FLIPS > 0 and best_cut < (target_cut if target_cut is not None else float('inf')):
        best_partition, best_cut, _field, _gain, _flips = _steepest_ascent_local_search(
            G_csr,
            best_partition,
            max_flips=PRIMARY_REPAIR_LOCAL_FLIPS,
            graph_sum=graph_sum,
            target_cut=target_cut,
        )
    return best_partition.astype(np.int8, copy=False), float(best_cut)


def _aqmf_risk_subproblem_repair(G_csr, partition, graph_sum, target_cut=None, seed=91001):
    """Run a lighter QAIA repair on AQMF risk nodes before any classical flip repair."""
    if not AQMF_QAIA_REPAIR_ENABLED:
        return _normalize_partition(partition).astype(np.int8, copy=True), _cut_value_numpy(
            G_csr,
            partition,
            graph_sum=graph_sum,
        )

    current = _normalize_partition(partition).astype(np.int8, copy=True)
    nodes = _select_repair_nodes(
        G_csr,
        current,
        AQMF_QAIA_REPAIR_NODE_LIMIT,
        expand_frac=PRIMARY_REPAIR_EXPAND,
    )
    if nodes.size <= 1:
        return current, _cut_value_numpy(G_csr, current, graph_sum=graph_sum)

    J_sub, h_sub = _build_repair_subproblem(G_csr, current, nodes)
    x_seed = current[nodes].astype(np.float32, copy=False)
    candidates = _run_nmfa_with_field_candidates(
        J_sub,
        h_sub,
        batch_size=max(1, AQMF_QAIA_REPAIR_BATCH_SIZE),
        n_iter=max(1, AQMF_QAIA_REPAIR_N_ITER),
        seed=seed,
        x=x_seed,
        alpha=0.15,
        sigma=0.12,
        x_scale=0.03,
    )
    repaired, repaired_cut = _score_repair_subproblem_candidates(
        G_csr,
        current,
        nodes,
        candidates,
        graph_sum=graph_sum,
        target_cut=target_cut,
    )
    if AQMF_QAIA_REPAIR_LOCAL_FLIPS > 0 and repaired_cut < (target_cut if target_cut is not None else float('inf')):
        repaired, repaired_cut, _field, _gain, _flips = _steepest_ascent_local_search(
            G_csr,
            repaired,
            max_flips=AQMF_QAIA_REPAIR_LOCAL_FLIPS,
            graph_sum=graph_sum,
            target_cut=target_cut,
        )
    return repaired.astype(np.int8, copy=False), float(repaired_cut)


def _boundary_repair_intervals(num_nodes, window_size, overlap, seed0, radius, topk=None):
    """Build contiguous repair intervals around stitched-window boundaries."""
    tasks = _build_window_tasks(num_nodes, window_size, overlap, seed0)
    stride = max(1, window_size - overlap)
    boundaries = [start for start in range(stride, num_nodes, stride)]
    intervals = []
    radius = max(overlap, int(radius))
    for boundary in boundaries:
        start = max(0, boundary - radius)
        end = min(num_nodes, boundary + radius)
        if end - start > 1:
            intervals.append((start, end))
    if topk is not None and len(intervals) > topk:
        # Favor wider central repairs first without depending on instance metadata.
        center = num_nodes / 2.0
        intervals.sort(key=lambda item: abs(((item[0] + item[1]) * 0.5) - center))
        intervals = intervals[:topk]
        intervals.sort()
    return intervals


def _qaia_boundary_repair(
    G_csr,
    partition,
    graph_sum,
    window_size,
    overlap,
    seed0,
    target_cut=None,
    seed=11003,
):
    """Repair stitched AMF output by re-solving boundary neighborhoods with QAIA."""
    best_partition = _normalize_partition(partition).astype(np.int8, copy=True)
    best_cut = _cut_value_numpy(G_csr, best_partition, graph_sum=graph_sum)
    intervals = _boundary_repair_intervals(
        G_csr.shape[0],
        window_size,
        overlap,
        seed0,
        PRIMARY_BOUNDARY_REPAIR_RADIUS,
        topk=PRIMARY_BOUNDARY_REPAIR_TOPK,
    )
    for interval_idx, (start, end) in enumerate(intervals):
        nodes = np.arange(start, end, dtype=np.int64)
        J_sub, h_sub = _build_repair_subproblem(G_csr, best_partition, nodes)
        x_seed = best_partition[nodes].astype(np.float32, copy=False)
        candidates = _run_nmfa_with_field_candidates(
            J_sub,
            h_sub,
            batch_size=PRIMARY_BOUNDARY_REPAIR_BATCH,
            n_iter=PRIMARY_BOUNDARY_REPAIR_N_ITER,
            seed=seed + interval_idx * 41,
            x=x_seed,
            alpha=0.15,
            sigma=0.12,
            x_scale=0.03,
        )
        repaired, repaired_cut = _score_repair_subproblem_candidates(
            G_csr,
            best_partition,
            nodes,
            candidates,
            graph_sum=graph_sum,
            target_cut=target_cut,
        )
        if repaired_cut > best_cut + 1e-7:
            best_partition, best_cut = repaired, repaired_cut
            if target_cut is not None and best_cut >= target_cut:
                break
    return best_partition.astype(np.int8, copy=False), float(best_cut)


def _stitch_window_partition(partition, overlap, start, end, local_partition):
    """Align one solved window against the current stitched prefix."""
    if start > 0:
        overlap_end = min(end, start + overlap)
        if overlap_end > start:
            overlap_global = partition[start:overlap_end]
            overlap_local = local_partition[:overlap_end - start]
            if np.sum(overlap_global == -overlap_local) > np.sum(overlap_global == overlap_local):
                local_partition = -local_partition

    partition[start:end] = local_partition


def _stitch_solved_windows(num_nodes, overlap, solved_windows):
    """Align and stitch ordered window partitions into one global partition."""
    partition = np.zeros(num_nodes, dtype=np.int8)

    for start, end, local_partition in solved_windows:
        _stitch_window_partition(partition, overlap, start, end, local_partition)

    partition[partition == 0] = 1
    return partition


def _choose_and_stitch_window_candidates(G_csr, num_nodes, overlap, solved_chunks):
    """Greedily stitch multi-candidate windows using prefix edge consistency."""
    partition = np.zeros(num_nodes, dtype=np.int8)
    ordered = []
    for chunk in solved_chunks:
        for batch in chunk:
            ordered.extend(zip(batch['tasks'], batch['partitions']))
    ordered.sort(key=lambda item: item[0][0])

    indptr = G_csr.indptr
    indices = G_csr.indices
    data = G_csr.data

    def prefix_score(start, end, values):
        score = 0.0
        values_f = values.astype(np.float32, copy=False)
        for local_idx, node in enumerate(range(start, end)):
            row_start, row_end = indptr[node], indptr[node + 1]
            neigh = indices[row_start:row_end]
            prefix_mask = neigh < start
            if np.any(prefix_mask):
                weights = data[row_start:row_end][prefix_mask]
                prev = partition[neigh[prefix_mask]].astype(np.float32, copy=False)
                score += float(values_f[local_idx] * np.dot(weights, prev))
        return 0.5 * score

    for (start, end, _seed), local_partition in ordered:
        if local_partition.ndim == 1:
            candidates = local_partition.reshape(-1, 1)
        else:
            candidates = local_partition
        best_values = None
        best_score = -float('inf')
        for col in range(candidates.shape[1]):
            base_values = candidates[:, col].astype(np.int8, copy=False)
            for sign in (1, -1):
                values = base_values if sign == 1 else -base_values
                local_score = prefix_score(start, end, values) - col * 0.01
                if start > 0:
                    overlap_end = min(end, start + overlap)
                    if overlap_end > start:
                        overlap_global = partition[start:overlap_end]
                        overlap_local = values[:overlap_end - start]
                        local_score += int(np.sum(overlap_global == overlap_local)) * 0.05
                if local_score > best_score:
                    best_score = local_score
                    best_values = values
        partition[start:end] = best_values

    partition[partition == 0] = 1
    return partition


def _collect_ordered_window_candidate_segments(num_nodes, solved_chunks, topk, reference_partition=None):
    """Convert solved window candidates into non-overlapping label segments."""
    ordered = []
    for chunk in solved_chunks:
        for batch in chunk:
            for task, local_partition in zip(batch['tasks'], batch['partitions']):
                start, end, _seed = task
                candidate_matrix = _normalize_candidate_matrix(local_partition)
                candidate_matrix = candidate_matrix[:, :max(1, min(int(topk), candidate_matrix.shape[1]))]
                full_labels = []
                seen_full = set()
                for col in range(candidate_matrix.shape[1]):
                    base_values = candidate_matrix[:, col].astype(np.int8, copy=False)
                    if reference_partition is not None:
                        ref_values = reference_partition[start:end]
                        if np.sum(base_values == -ref_values) > np.sum(base_values == ref_values):
                            base_values = -base_values
                    key = base_values.tobytes()
                    if key in seen_full:
                        continue
                    seen_full.add(key)
                    full_labels.append(base_values.astype(np.int8, copy=True))
                if full_labels:
                    ordered.append((start, end, np.column_stack(full_labels).astype(np.int8, copy=False)))

    ordered.sort(key=lambda item: item[0])
    segments = []
    for idx, (start, end, candidates) in enumerate(ordered):
        next_start = ordered[idx + 1][0] if idx + 1 < len(ordered) else end
        segment_end = min(end, max(start + 1, next_start))
        segment_end = min(segment_end, num_nodes)
        if segment_end <= start:
            continue
        local_slice = slice(0, segment_end - start)
        segment_labels = candidates[local_slice]
        unique_labels = []
        seen_segment = set()
        if reference_partition is not None:
            ref_segment = reference_partition[start:segment_end].astype(np.int8, copy=False)
            unique_labels.append(ref_segment.astype(np.int8, copy=True))
            seen_segment.add(ref_segment.tobytes())
        for col in range(segment_labels.shape[1]):
            values = segment_labels[:, col]
            if reference_partition is not None:
                ref_segment = reference_partition[start:segment_end]
                if np.sum(values == -ref_segment) > np.sum(values == ref_segment):
                    values = -values
            key = values.tobytes()
            if key in seen_segment:
                continue
            seen_segment.add(key)
            unique_labels.append(values.astype(np.int8, copy=True))
        if unique_labels:
            segments.append({
                'start': start,
                'end': segment_end,
                'candidates': np.column_stack(unique_labels).astype(np.float32, copy=False),
            })
    return segments


def _softmax_window_label_state(state, label_counts, select_beta):
    """Normalize continuous label amplitudes into per-window probabilities."""
    logits = np.asarray(state * np.float32(select_beta), dtype=np.float32)
    max_labels = logits.shape[1]
    valid = np.arange(max_labels)[None, :, None] < label_counts[:, None, None]
    logits = np.where(valid, logits, np.float32(-1.0e9))
    logits -= np.max(logits, axis=1, keepdims=True)
    probs = np.exp(logits).astype(np.float32, copy=False)
    probs *= valid
    normalizer = np.sum(probs, axis=1, keepdims=True, dtype=np.float32)
    normalizer[normalizer == 0] = 1.0
    probs /= normalizer
    return probs.astype(np.float32, copy=False)


def _decode_window_label_segments(num_nodes, segments, label_choices):
    """Build global partitions from selected segment labels."""
    batch_size = label_choices.shape[1]
    partitions = np.ones((num_nodes, batch_size), dtype=np.int8)
    for window_idx, segment in enumerate(segments):
        start = segment['start']
        end = segment['end']
        candidates = segment['candidates'].astype(np.int8, copy=False)
        for batch_idx in range(batch_size):
            label_idx = int(label_choices[window_idx, batch_idx])
            partitions[start:end, batch_idx] = candidates[:, label_idx]
    return partitions


def _fuse_window_candidates_by_block_ascent(
    G_csr,
    num_nodes,
    solved_chunks,
    graph_sum,
    local_flips,
    target_cut=None,
    reference_partition=None,
):
    """Exact segment-label ascent over AMF-generated window candidates."""
    if reference_partition is None:
        return None
    segments = _collect_ordered_window_candidate_segments(
        num_nodes,
        solved_chunks,
        topk=PRIMARY_AMF_FUSION_TOPK,
        reference_partition=reference_partition,
    )
    if not segments:
        return None

    current = _normalize_partition(reference_partition).astype(np.int8, copy=True)
    current_f = current.astype(np.float32, copy=True)
    field = G_csr.dot(current_f)
    current_cut = _cut_value_from_state(current_f, field, graph_sum)
    best_partition = current.copy()
    best_cut = float(current_cut)
    passes = max(1, int(PRIMARY_AMF_BLOCK_FUSION_PASSES))
    changed = False

    for _ in range(passes):
        improved_this_pass = False
        for segment in segments:
            start = segment['start']
            end = segment['end']
            candidates = segment['candidates'].astype(np.int8, copy=False)
            old_values = current[start:end].copy()
            old_f = old_values.astype(np.float32, copy=False)
            best_values = None
            segment_best_cut = current_cut
            segment_best_delta = None

            for col in range(candidates.shape[1]):
                values = candidates[:, col].astype(np.int8, copy=False)
                if np.array_equal(values, old_values):
                    continue
                delta = values.astype(np.float32, copy=False) - old_f
                linear_gain = float(np.dot(delta, field[start:end]))
                subgraph = G_csr[start:end, start:end]
                quadratic_gain = float(delta @ subgraph.dot(delta))
                trial_cut = current_cut + np.float32(0.5) * linear_gain + np.float32(0.25) * quadratic_gain
                if trial_cut > segment_best_cut + 1e-7:
                    segment_best_cut = float(trial_cut)
                    best_values = values.copy()
                    segment_best_delta = delta.copy()

            if best_values is not None:
                current[start:end] = best_values
                current_f[start:end] = best_values.astype(np.float32, copy=False)
                field += G_csr[:, start:end].dot(segment_best_delta)
                current_cut = segment_best_cut
                improved_this_pass = True
                changed = True
                if current_cut > best_cut:
                    best_cut = float(current_cut)
                    best_partition = current.copy()
                if target_cut is not None and current_cut >= target_cut:
                    break
        if target_cut is not None and current_cut >= target_cut:
            break
        if not improved_this_pass:
            break

    repaired, repaired_cut, _field, _gain, _flips = _steepest_ascent_local_search(
        G_csr,
        best_partition,
        max_flips=local_flips,
        graph_sum=graph_sum,
        target_cut=target_cut,
    )
    if repaired_cut > best_cut:
        best_partition, best_cut = repaired, float(repaired_cut)

    return {
        'partition': best_partition.astype(np.int8, copy=False),
        'cut': float(best_cut),
        'stitched': best_partition.astype(np.int8, copy=True),
        'candidate_count': int(sum(segment['candidates'].shape[1] for segment in segments)),
        'segment_count': int(len(segments)),
        'max_labels': int(max(segment['candidates'].shape[1] for segment in segments)),
        'changed': bool(changed),
    }


def _fuse_window_candidates_with_amf(
    G_csr,
    num_nodes,
    solved_chunks,
    graph_sum,
    local_flips,
    target_cut=None,
    reference_partition=None,
):
    """
    Global quantum-inspired fusion over AMF window candidates.

    Each window segment is a low-dimensional Potts-like variable whose labels
    are AMF-generated local candidates. A noisy mean-field dynamics updates the
    label amplitudes using the full-graph field, then decoded global candidates
    are scored with the same bounded repair used elsewhere.
    """
    segments = _collect_ordered_window_candidate_segments(
        num_nodes,
        solved_chunks,
        topk=PRIMARY_AMF_FUSION_TOPK,
        reference_partition=reference_partition,
    )
    if len(segments) <= 1:
        return None

    label_counts = np.asarray([segment['candidates'].shape[1] for segment in segments], dtype=np.int64)
    max_labels = int(np.max(label_counts))
    if max_labels <= 1:
        return None

    batch_size = max(1, int(PRIMARY_AMF_FUSION_BATCH))
    n_iter = max(1, int(PRIMARY_AMF_FUSION_N_ITER))
    rng = np.random.default_rng(PRIMARY_AMF_FUSION_SEED + num_nodes + len(segments) * 17)
    state = (0.04 * rng.standard_normal((len(segments), max_labels, batch_size))).astype(np.float32)
    for window_idx, count in enumerate(label_counts):
        state[window_idx, 0, 0] += np.float32(0.35)
        if batch_size > 1 and count > 1:
            state[window_idx, min(1, count - 1), 1] += np.float32(0.35)
        for batch_idx in range(2, batch_size):
            state[window_idx, int(rng.integers(0, count)), batch_idx] += np.float32(0.25)

    expected = np.empty((num_nodes, batch_size), dtype=np.float32)
    label_field = np.zeros_like(state)
    beta_start = np.float32(1.0 / n_iter)
    beta_final = np.float32(PRIMARY_AMF_FUSION_BETA_FINAL)
    beta_step = (beta_final - beta_start) / np.float32(max(1, n_iter - 1))
    beta = beta_start
    alpha = np.float32(PRIMARY_AMF_FUSION_ALPHA)
    one_minus_alpha = np.float32(1.0) - alpha
    sigma = float(PRIMARY_AMF_FUSION_SIGMA)

    for _ in range(n_iter):
        probs = _softmax_window_label_state(state, label_counts, select_beta=PRIMARY_AMF_FUSION_SELECT_FINAL)
        expected.fill(1.0)
        for window_idx, segment in enumerate(segments):
            start = segment['start']
            end = segment['end']
            labels = segment['candidates']
            expected[start:end] = labels @ probs[window_idx, :label_counts[window_idx]]

        field = G_csr.dot(expected)
        if field.dtype != np.float32:
            field = field.astype(np.float32, copy=False)

        label_field.fill(0.0)
        for window_idx, segment in enumerate(segments):
            start = segment['start']
            end = segment['end']
            count = label_counts[window_idx]
            labels = segment['candidates']
            local_scores = labels.T @ field[start:end]
            local_scores -= np.mean(local_scores, axis=0, keepdims=True, dtype=np.float32)
            scale = np.sqrt(
                np.mean(local_scores * local_scores, axis=0, keepdims=True, dtype=np.float32) + np.float32(1e-6),
                dtype=np.float32,
            )
            label_field[window_idx, :count] = local_scores / scale

        if sigma > 0:
            label_field += rng.normal(0.0, sigma, size=label_field.shape).astype(np.float32)
        np.tanh(label_field * beta, out=label_field)
        state *= one_minus_alpha
        state += alpha * label_field
        beta += beta_step

    label_choices = np.zeros((len(segments), batch_size), dtype=np.int64)
    final_probs = _softmax_window_label_state(state, label_counts, select_beta=PRIMARY_AMF_FUSION_SELECT_FINAL)
    for window_idx, count in enumerate(label_counts):
        label_choices[window_idx] = np.argmax(final_probs[window_idx, :count], axis=0)

    fused_partitions = _decode_window_label_segments(num_nodes, segments, label_choices)
    partition, cut = _score_and_repair_candidates(
        G_csr,
        [('subgraph-amf-global-fusion', fused_partitions)],
        max_flips=local_flips,
        graph_sum=graph_sum,
        target_cut=target_cut,
    )
    return {
        'partition': partition,
        'cut': float(cut),
        'stitched': fused_partitions[:, 0].astype(np.int8, copy=True),
        'candidate_count': int(batch_size),
        'segment_count': int(len(segments)),
        'max_labels': int(max_labels),
    }


def _flatten_probe_solved_windows(solved_chunks):
    """Return ordered window entries from probe chunk output."""
    entries = []
    for chunk in solved_chunks:
        for batch in chunk:
            for task, local_partition in zip(batch['tasks'], batch['partitions']):
                start, _end, _seed = task
                entries.append((start, task, local_partition))
    entries.sort(key=lambda item: item[0])
    return [(task, local_partition) for _start, task, local_partition in entries]


def _stitch_window_entries(num_nodes, overlap, entries):
    """Stitch ordered window entries that may contain batched candidates."""
    partition = np.zeros(num_nodes, dtype=np.int8)
    for (start, end, _seed), local_partition in entries:
        candidate_matrix = _normalize_candidate_matrix(local_partition)
        _stitch_window_partition(partition, overlap, start, end, candidate_matrix[:, 0])
    partition[partition == 0] = 1
    return partition


def _top_window_indices(values, limit, largest=True):
    """Return top-k indices from a 1-D score vector."""
    values = np.asarray(values, dtype=np.float32)
    limit = max(1, min(int(limit), values.size))
    if limit >= values.size:
        return np.arange(values.size, dtype=np.int64)
    ranked_values = values if largest else -values
    selected = np.argpartition(ranked_values, ranked_values.size - limit)[ranked_values.size - limit:]
    return selected.astype(np.int64, copy=False)


def _select_nmfa_repair_window_indices(G_csr, partition, entries, limit):
    """Choose windows whose AMF stitched result looks least stable globally."""
    limit = max(1, min(int(limit), len(entries)))
    if limit >= len(entries):
        return np.arange(len(entries), dtype=np.int64)

    _spins, _spins_f, _field, gain = _build_local_search_state(G_csr, partition)
    score_mode = PRIMARY_AMF_SELECTIVE_NMFA_SCORE
    unstable_scores = np.empty(len(entries), dtype=np.float32)
    posgain_scores = np.empty(len(entries), dtype=np.float32)
    local_scores = None
    if score_mode in ('local', 'hybrid', 'hybrid_union'):
        local_scores = np.empty(len(entries), dtype=np.float32)

    for idx, ((start, end, _seed), _local_partition) in enumerate(entries):
        local_gain = gain[start:end]
        unstable_scores[idx] = np.sum(np.float32(1.0) / (np.abs(local_gain) + np.float32(1e-3)), dtype=np.float32)
        posgain_scores[idx] = np.sum(np.maximum(local_gain, np.float32(0.0)), dtype=np.float32)
        if local_scores is not None:
            local_values = _normalize_candidate_matrix(_local_partition)[:, 0]
            local_graph = G_csr[start:end, start:end].tocsr()
            local_scores[idx] = _cut_value_numpy(local_graph, local_values)

    if score_mode == 'posgain':
        selected = _top_window_indices(posgain_scores, limit, largest=True)
    elif score_mode == 'abs':
        abs_scores = np.empty(len(entries), dtype=np.float32)
        for idx, ((start, end, _seed), _local_partition) in enumerate(entries):
            abs_scores[idx] = -np.mean(np.abs(gain[start:end]), dtype=np.float32)
        selected = _top_window_indices(abs_scores, limit, largest=True)
    elif score_mode == 'local':
        selected = _top_window_indices(local_scores, limit, largest=True)
    elif score_mode in ('hybrid', 'hybrid_union'):
        selected_parts = [
            _top_window_indices(unstable_scores, limit, largest=True),
            _top_window_indices(posgain_scores, limit, largest=True),
            _top_window_indices(local_scores, limit, largest=True),
        ]
        selected = np.unique(np.concatenate(selected_parts))
    else:
        selected = _top_window_indices(unstable_scores, limit, largest=True)

    selected.sort()
    return selected.astype(np.int64, copy=False)


def _select_single_window_repair_indices(G_csr, partition, entries, limit):
    """Build a broad, deterministic candidate set for one-window QAIA repair."""
    limit = max(1, min(int(limit), len(entries)))
    _spins, _spins_f, _field, gain = _build_local_search_state(G_csr, partition)
    unstable_scores = np.empty(len(entries), dtype=np.float32)
    posgain_scores = np.empty(len(entries), dtype=np.float32)
    local_scores = np.empty(len(entries), dtype=np.float32)
    for idx, ((start, end, _seed), local_partition) in enumerate(entries):
        local_gain = gain[start:end]
        unstable_scores[idx] = np.sum(np.float32(1.0) / (np.abs(local_gain) + np.float32(1e-3)), dtype=np.float32)
        posgain_scores[idx] = np.sum(np.maximum(local_gain, np.float32(0.0)), dtype=np.float32)
        local_values = _normalize_candidate_matrix(local_partition)[:, 0]
        local_scores[idx] = _cut_value_numpy(G_csr[start:end, start:end].tocsr(), local_values)

    per_signal = max(1, limit // 4)
    selected_parts = [
        _top_window_indices(unstable_scores, per_signal, largest=True),
        _top_window_indices(posgain_scores, per_signal, largest=True),
        _top_window_indices(local_scores, per_signal, largest=True),
        _top_window_indices(local_scores, per_signal, largest=False),
    ]
    selected = np.unique(np.concatenate(selected_parts))
    if selected.size < limit:
        fill = _top_window_indices(unstable_scores + posgain_scores, limit, largest=True)
        selected = np.unique(np.concatenate((selected, fill)))
    if selected.size > limit:
        selected = selected[:limit]
    selected.sort()
    return selected.astype(np.int64, copy=False)


def _solve_selected_windows_with_nmfa(G_csr, entries, selected, batch_size, n_iter):
    """Solve selected window entries with dense-batched NMFA and return index map."""
    selected_tasks = [(int(idx), entries[int(idx)][0]) for idx in selected]
    selected_tasks.sort(key=lambda item: item[1][0])
    grouped = []
    group = []
    last_len = None
    for idx, task in selected_tasks:
        window_len = task[1] - task[0]
        if last_len is not None and (window_len != last_len or len(group) >= WINDOW_DENSE_BATCH):
            grouped.append(group)
            group = []
        group.append((idx, task))
        last_len = window_len
    if group:
        grouped.append(group)

    repaired = {}
    for group in grouped:
        batch_tasks = [task for _idx, task in group]
        local_partitions = _solve_window_batch(
            G_csr,
            batch_tasks,
            batch_size=batch_size,
            n_iter=n_iter,
            alpha=0.15,
            sigma=0.15,
            x_scale=0.05,
        )
        for (idx, _task), (_start, _end, local_partition) in zip(group, local_partitions):
            repaired[idx] = local_partition
    return repaired


def _repair_probe_by_single_nmfa_window(
    G_csr,
    solved_chunks,
    num_nodes,
    overlap,
    reference_partition,
    graph_sum,
    local_flips,
    target_cut=None,
):
    """Try replacing one AMF window at a time with QAIA/NMFA solved windows."""
    entries = _flatten_probe_solved_windows(solved_chunks)
    if not entries:
        return None

    selected = _select_single_window_repair_indices(
        G_csr,
        reference_partition,
        entries,
        PRIMARY_AMF_SINGLE_WINDOW_LIMIT,
    )
    repaired_windows = _solve_selected_windows_with_nmfa(
        G_csr,
        entries,
        selected,
        batch_size=max(1, int(PRIMARY_AMF_SELECTIVE_NMFA_BATCH)),
        n_iter=max(1, int(PRIMARY_AMF_SELECTIVE_NMFA_N_ITER)),
    )

    candidates = [('subgraph-amf-single-window-base', reference_partition)]
    for idx in selected:
        repaired_entries = list(entries)
        repaired_entries[int(idx)] = (entries[int(idx)][0], repaired_windows[int(idx)])
        stitched = _stitch_window_entries(num_nodes, overlap, repaired_entries)
        candidates.append((f'subgraph-amf-single-window-{int(idx)}', stitched))

    partition, cut = _score_and_repair_candidates(
        G_csr,
        candidates,
        max_flips=local_flips,
        graph_sum=graph_sum,
        target_cut=target_cut,
    )
    return {
        'partition': partition,
        'cut': float(cut),
        'stitched': partition.astype(np.int8, copy=True),
        'selected_count': int(len(selected)),
        'window_count': int(len(entries)),
    }


def _repair_probe_windows_with_nmfa(
    G_csr,
    solved_chunks,
    num_nodes,
    overlap,
    reference_partition,
    graph_sum,
    local_flips,
    target_cut=None,
    score_mode=None,
    topk=None,
):
    """Replace selected AMF windows with NMFA-solved windows, then rescore."""
    entries = _flatten_probe_solved_windows(solved_chunks)
    if not entries:
        return None

    previous_score_mode = None
    if score_mode is not None:
        global PRIMARY_AMF_SELECTIVE_NMFA_SCORE
        previous_score_mode = PRIMARY_AMF_SELECTIVE_NMFA_SCORE
        PRIMARY_AMF_SELECTIVE_NMFA_SCORE = score_mode
    selected = _select_nmfa_repair_window_indices(
        G_csr,
        reference_partition,
        entries,
        PRIMARY_AMF_SELECTIVE_NMFA_TOPK if topk is None else topk,
    )
    if previous_score_mode is not None:
        PRIMARY_AMF_SELECTIVE_NMFA_SCORE = previous_score_mode
    selected_set = set(int(idx) for idx in selected)
    repaired_entries = list(entries)
    batch_size = max(1, int(PRIMARY_AMF_SELECTIVE_NMFA_BATCH))
    n_iter = max(1, int(PRIMARY_AMF_SELECTIVE_NMFA_N_ITER))

    selected_tasks = [(int(idx), entries[int(idx)][0]) for idx in selected]
    selected_tasks.sort(key=lambda item: item[1][0])
    grouped = []
    group = []
    last_len = None
    for idx, task in selected_tasks:
        window_len = task[1] - task[0]
        if last_len is not None and (window_len != last_len or len(group) >= WINDOW_DENSE_BATCH):
            grouped.append(group)
            group = []
        group.append((idx, task))
        last_len = window_len
    if group:
        grouped.append(group)

    for group in grouped:
        batch_tasks = [task for _idx, task in group]
        if PRIMARY_AMF_SELECTIVE_NMFA_WARM_START:
            J_batch = _materialize_window_dense_batch(G_csr, batch_tasks)
            x_init = np.empty((len(batch_tasks), J_batch.shape[1], batch_size), dtype=np.float32)
            for local_idx, (idx, _task) in enumerate(group):
                base = _normalize_candidate_matrix(entries[idx][1])[:, 0].astype(np.float32, copy=False)
                rng = np.random.default_rng(entries[idx][0][2] + 31013)
                x_init[local_idx, :, 0] = base
                if batch_size > 1:
                    noise = rng.standard_normal((base.size, batch_size - 1)).astype(np.float32)
                    x_init[local_idx, :, 1:] = (
                        np.float32(0.65) * base[:, None]
                        + np.float32(PRIMARY_AMF_SELECTIVE_NMFA_WARM_SCALE) * noise
                    )
            local_partitions = _run_dense_nmfa_batched_from_x(
                J_batch,
                x_init,
                n_iter=n_iter,
                seeds=[task[2] + 41017 for task in batch_tasks],
                alpha=0.15,
                sigma=PRIMARY_AMF_SELECTIVE_NMFA_WARM_SIGMA,
            )
            for (idx, _task), local_partition in zip(group, local_partitions):
                repaired_entries[idx] = (entries[idx][0], local_partition)
        else:
            local_partitions = _solve_window_batch(
                G_csr,
                batch_tasks,
                batch_size=batch_size,
                n_iter=n_iter,
                alpha=0.15,
                sigma=0.15,
                x_scale=0.05,
            )
            for (idx, _task), (_start, _end, local_partition) in zip(group, local_partitions):
                repaired_entries[idx] = (entries[idx][0], local_partition)

    stitched = _stitch_window_entries(num_nodes, overlap, repaired_entries)
    partition, cut = _score_and_repair_candidates(
        G_csr,
        [('subgraph-amf-selective-nmfa', stitched)],
        max_flips=local_flips,
        graph_sum=graph_sum,
        target_cut=target_cut,
    )
    return {
        'partition': partition,
        'cut': float(cut),
        'stitched': stitched,
        'selected_count': int(len(selected_set)),
        'window_count': int(len(entries)),
    }


def _run_nmfa(subgraph, batch_size, n_iter, seed, alpha=0.15, sigma=0.15, sigma_end=None, x_scale=0.05):
    """Solve one subgraph with NMFA from random initialization."""
    if subgraph.shape[0] <= 256:
        return _run_dense_nmfa(
            subgraph,
            batch_size=batch_size,
            n_iter=n_iter,
            seed=seed,
            alpha=alpha,
            sigma=sigma,
            sigma_end=sigma_end,
            x_scale=x_scale,
        )

    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    x_init = np.float32(x_scale) * rng.standard_normal((subgraph.shape[0], batch_size)).astype(np.float32)
    solver = NMFA(subgraph, x=x_init, n_iter=n_iter, batch_size=batch_size, alpha=alpha, sigma=sigma)
    solver.update()
    candidates = _normalize_candidate_matrix(solver.x)
    scores = np.array([_cut_value_numpy(subgraph, candidates[:, col]) for col in range(batch_size)])
    return candidates[:, int(np.argmax(scores))].astype(np.int8, copy=True)


def _run_dense_nmfa_matrix(J, batch_size, n_iter, seed, alpha=0.15, sigma=0.15, sigma_end=None, x_scale=0.05):
    """
    Dense NMFA path for one small coupling matrix.

    Small windowed subproblems are often dense enough that dense BLAS is faster
    than repeated CSR matvecs, while the underlying NMFA dynamics stay unchanged.
    """
    rng = np.random.default_rng(seed)
    J = np.asarray(J, dtype=np.float32)
    n = J.shape[0]
    x = np.float32(x_scale) * rng.standard_normal((n, batch_size)).astype(np.float32)
    inv_norm = np.sqrt(np.sum(J * J, axis=1, keepdims=True, dtype=np.float32), dtype=np.float32)
    inv_norm[inv_norm == 0] = 1.0
    np.reciprocal(inv_norm, out=inv_norm)
    beta = np.float32(1.0 / n_iter)
    beta_step = np.float32(1.0 / n_iter)
    alpha = np.float32(alpha)
    one_minus_alpha = np.float32(1.0) - alpha
    phi = np.empty_like(x)
    x_hat = np.empty_like(x)
    if sigma_end is None or abs(float(sigma_end) - float(sigma)) <= 1e-12:
        sigma_schedule = None
        sigma_value = float(sigma)
        if sigma_value > 0:
            noise = np.random.RandomState(seed).normal(0.0, sigma_value, size=(n_iter, n, batch_size)).astype(np.float32)
        else:
            noise = None
    else:
        if n_iter <= 1:
            sigma_schedule = np.asarray([sigma_end], dtype=np.float32)
        else:
            sigma_schedule = np.linspace(sigma, sigma_end, n_iter, dtype=np.float32)
        noise = None
        noise_rng = np.random.RandomState(seed)

    for i in range(n_iter):
        np.matmul(J, x, out=phi)
        phi *= inv_norm
        if noise is not None:
            phi += noise[i]
        else:
            if sigma_schedule is not None:
                sigma_i = float(sigma_schedule[i])
                if sigma_i > 0:
                    phi += noise_rng.normal(0.0, sigma_i, size=(n, batch_size)).astype(np.float32)
        np.multiply(phi, beta, out=phi)
        np.tanh(phi, out=x_hat)
        x *= one_minus_alpha
        x += alpha * x_hat
        beta += beta_step

    candidates = np.sign(x, out=np.empty_like(x))
    candidates[candidates == 0] = 1.0
    scores = 0.25 * np.sum(candidates * (J @ candidates), axis=0, dtype=np.float32)
    scores -= 0.25 * np.sum(J, dtype=np.float32)
    return candidates[:, int(np.argmax(scores))].astype(np.int8, copy=True)


def _run_dense_nmfa(subgraph, batch_size, n_iter, seed, alpha=0.15, sigma=0.15, sigma_end=None, x_scale=0.05):
    """Dense NMFA path for small subgraphs."""
    J = subgraph.toarray().astype(np.float32, copy=False)
    return _run_dense_nmfa_matrix(
        J,
        batch_size=batch_size,
        n_iter=n_iter,
        seed=seed,
        alpha=alpha,
        sigma=sigma,
        sigma_end=sigma_end,
        x_scale=x_scale,
    )


def _materialize_window_dense_batch(G_csr, batch_tasks, out=None):
    """Materialize equal-sized CSR windows into one float32 dense batch."""
    window_len = batch_tasks[0][1] - batch_tasks[0][0]
    if out is None:
        J_batch = np.empty((len(batch_tasks), window_len, window_len), dtype=np.float32)
    else:
        J_batch = out
    for window_idx, (start, end, _seed) in enumerate(batch_tasks):
        if G_csr.dtype == np.float32:
            G_csr[start:end, start:end].toarray(out=J_batch[window_idx])
        else:
            dense = G_csr[start:end, start:end].toarray()
            J_batch[window_idx] = dense
    return J_batch


def _run_dense_nmfa_batched(J_batch, batch_size, n_iter, seeds, alpha=0.15, sigma=0.15, x_scale=0.05):
    """
    Solve several equal-sized small windows with one batched dense NMFA pass.

    This keeps each window's QAIA dynamics unchanged while amortizing Python
    overhead across the dominant stitched-window path.
    """
    J_batch = np.asarray(J_batch, dtype=np.float32)
    num_windows, num_nodes, _ = J_batch.shape
    x = np.empty((num_windows, num_nodes, batch_size), dtype=np.float32)
    noise = None
    sigma_value = float(sigma)
    if sigma_value > 0:
        noise = np.empty((num_windows, n_iter, num_nodes, batch_size), dtype=np.float32)

    scale = np.float32(x_scale)
    for window_idx, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        x[window_idx] = scale * rng.standard_normal((num_nodes, batch_size)).astype(np.float32)
        if noise is not None:
            noise[window_idx] = np.random.RandomState(seed).normal(
                0.0,
                sigma_value,
                size=(n_iter, num_nodes, batch_size),
            ).astype(np.float32)

    inv_norm = np.sqrt(np.sum(J_batch * J_batch, axis=2, keepdims=True, dtype=np.float32), dtype=np.float32)
    inv_norm[inv_norm == 0] = 1.0
    np.reciprocal(inv_norm, out=inv_norm)

    beta = np.float32(1.0 / n_iter)
    beta_step = np.float32(1.0 / n_iter)
    alpha = np.float32(alpha)
    one_minus_alpha = np.float32(1.0) - alpha
    phi = np.empty_like(x)
    x_hat = np.empty_like(x)

    for iter_idx in range(n_iter):
        np.matmul(J_batch, x, out=phi)
        phi *= inv_norm
        if noise is not None:
            phi += noise[:, iter_idx]
        np.multiply(phi, beta, out=phi)
        np.tanh(phi, out=x_hat)
        x *= one_minus_alpha
        x += alpha * x_hat
        beta += beta_step

    candidates = np.sign(x, out=np.empty_like(x))
    candidates[candidates == 0] = 1.0
    scores = 0.25 * np.sum(candidates * np.matmul(J_batch, candidates), axis=1, dtype=np.float32)
    scores -= 0.25 * np.sum(J_batch, axis=(1, 2), dtype=np.float32)[:, None]
    best_indices = np.argmax(scores, axis=1)
    return [
        candidates[window_idx, :, best_indices[window_idx]].astype(np.int8, copy=True)
        for window_idx in range(num_windows)
    ]


def _run_dense_nmfa_batched_from_x(J_batch, x_init, n_iter, seeds, alpha=0.15, sigma=0.08):
    """Run dense batched NMFA from supplied continuous warm-start states."""
    J_batch = np.asarray(J_batch, dtype=np.float32)
    x = np.asarray(x_init, dtype=np.float32).copy()
    num_windows, num_nodes, batch_size = x.shape
    sigma_value = float(sigma)
    noise = None
    if sigma_value > 0:
        noise = np.empty((num_windows, n_iter, num_nodes, batch_size), dtype=np.float32)
        for window_idx, seed in enumerate(seeds):
            noise[window_idx] = np.random.RandomState(seed).normal(
                0.0,
                sigma_value,
                size=(n_iter, num_nodes, batch_size),
            ).astype(np.float32)

    inv_norm = np.sqrt(np.sum(J_batch * J_batch, axis=2, keepdims=True, dtype=np.float32), dtype=np.float32)
    inv_norm[inv_norm == 0] = 1.0
    np.reciprocal(inv_norm, out=inv_norm)

    beta = np.float32(1.0 / max(1, n_iter))
    beta_step = np.float32(1.0 / max(1, n_iter))
    alpha = np.float32(alpha)
    one_minus_alpha = np.float32(1.0) - alpha
    phi = np.empty_like(x)
    x_hat = np.empty_like(x)

    for iter_idx in range(n_iter):
        np.matmul(J_batch, x, out=phi)
        phi *= inv_norm
        if noise is not None:
            phi += noise[:, iter_idx]
        np.multiply(phi, beta, out=phi)
        np.tanh(phi, out=x_hat)
        x *= one_minus_alpha
        x += alpha * x_hat
        beta += beta_step

    candidates = np.sign(x, out=np.empty_like(x))
    candidates[candidates == 0] = 1.0
    return _score_dense_window_candidates(J_batch, candidates, topk=1)


def _make_dense_nmfa_batched_state(
    J_batch,
    batch_size,
    n_iter,
    seeds,
    alpha=0.15,
    sigma=0.15,
    x_scale=0.05,
    initial_noise_iters=None,
):
    """Prepare a resumable dense batched NMFA state."""
    J_batch = np.asarray(J_batch, dtype=np.float32)
    num_windows, num_nodes, _ = J_batch.shape
    x = np.empty((num_windows, num_nodes, batch_size), dtype=np.float32)
    noise = None
    noise_rngs = None
    noise_iters = n_iter
    sigma_value = float(sigma)
    if sigma_value > 0:
        if initial_noise_iters is not None:
            noise_iters = max(0, min(n_iter, int(initial_noise_iters)))
        noise = np.empty((num_windows, noise_iters, num_nodes, batch_size), dtype=np.float32)
        noise_rngs = []

    scale = np.float32(x_scale)
    for window_idx, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        x[window_idx] = scale * rng.standard_normal((num_nodes, batch_size)).astype(np.float32)
        if noise is not None:
            noise_rng = np.random.RandomState(seed)
            noise_rngs.append(noise_rng)
            if noise_iters > 0:
                noise[window_idx] = noise_rng.normal(
                    0.0,
                    sigma_value,
                    size=(noise_iters, num_nodes, batch_size),
                ).astype(np.float32)

    inv_norm = np.sqrt(np.sum(J_batch * J_batch, axis=2, keepdims=True, dtype=np.float32), dtype=np.float32)
    inv_norm[inv_norm == 0] = 1.0
    np.reciprocal(inv_norm, out=inv_norm)

    return {
        'J_batch': J_batch,
        'x': x,
        'noise': noise,
        'noise_extra': None,
        'noise_rngs': noise_rngs,
        'sigma_value': sigma_value,
        'inv_norm': inv_norm,
        'beta': np.float32(1.0 / n_iter),
        'beta_step': np.float32(1.0 / n_iter),
        'alpha': np.float32(alpha),
        'one_minus_alpha': np.float32(1.0) - np.float32(alpha),
        'phi': np.empty_like(x),
        'x_hat': np.empty_like(x),
        'iter_idx': 0,
        'n_iter': n_iter,
    }


def _ensure_dense_nmfa_noise_extra(state, stop_iter):
    """Generate deferred NMFA noise without copying already used warmup noise."""
    noise = state['noise']
    if noise is None or noise.shape[1] >= stop_iter:
        return None

    J_batch = state['J_batch']
    x = state['x']
    num_windows = J_batch.shape[0]
    num_nodes = J_batch.shape[1]
    batch_size = x.shape[2]
    available = noise.shape[1]
    required_extra_iters = stop_iter - available
    noise_extra = state.get('noise_extra')
    existing_extra_iters = 0 if noise_extra is None else noise_extra.shape[1]
    if existing_extra_iters >= required_extra_iters:
        return noise_extra

    add_iters = required_extra_iters - existing_extra_iters
    extra = np.empty((num_windows, add_iters, num_nodes, batch_size), dtype=np.float32)
    sigma_value = state['sigma_value']

    for window_idx, noise_rng in enumerate(state['noise_rngs']):
        extra[window_idx] = noise_rng.normal(
            0.0,
            sigma_value,
            size=(add_iters, num_nodes, batch_size),
        ).astype(np.float32)

    if noise_extra is None:
        state['noise_extra'] = extra
    else:
        extended = np.empty((num_windows, required_extra_iters, num_nodes, batch_size), dtype=np.float32)
        extended[:, :existing_extra_iters] = noise_extra
        extended[:, existing_extra_iters:] = extra
        state['noise_extra'] = extended
    return state['noise_extra']


def _advance_dense_nmfa_batched_state(state, steps):
    """Advance a resumable dense batched NMFA state by a fixed number of steps."""
    stop_iter = min(state['n_iter'], state['iter_idx'] + steps)
    J_batch = state['J_batch']
    x = state['x']
    noise = state['noise']
    noise_extra = _ensure_dense_nmfa_noise_extra(state, stop_iter)
    noise_split = noise.shape[1] if noise is not None else 0
    inv_norm = state['inv_norm']
    phi = state['phi']
    x_hat = state['x_hat']
    beta = state['beta']
    beta_step = state['beta_step']
    alpha = state['alpha']
    one_minus_alpha = state['one_minus_alpha']
    iter_idx = state['iter_idx']

    while iter_idx < stop_iter:
        np.matmul(J_batch, x, out=phi)
        phi *= inv_norm
        if noise is not None:
            if iter_idx < noise_split:
                phi += noise[:, iter_idx]
            else:
                phi += noise_extra[:, iter_idx - noise_split]
        np.multiply(phi, beta, out=phi)
        np.tanh(phi, out=x_hat)
        x *= one_minus_alpha
        x += alpha * x_hat
        beta += beta_step
        iter_idx += 1

    state['beta'] = beta
    state['iter_idx'] = iter_idx


def _decode_dense_nmfa_batched_state(state):
    """Decode the current best partition of each batched dense NMFA window."""
    J_batch = state['J_batch']
    x = state['x']
    candidates = np.sign(x, out=np.empty_like(x))
    candidates[candidates == 0] = 1.0
    J_sums = state.get('J_sums')
    if J_sums is None:
        J_sums = np.sum(J_batch, axis=(1, 2), dtype=np.float32)
        state['J_sums'] = J_sums
    scores = 0.25 * np.sum(candidates * np.matmul(J_batch, candidates), axis=1, dtype=np.float32)
    scores -= 0.25 * J_sums[:, None]
    best_indices = np.argmax(scores, axis=1)
    return [
        candidates[window_idx, :, best_indices[window_idx]].astype(np.int8, copy=True)
        for window_idx in range(J_batch.shape[0])
    ]


def _score_dense_window_candidates(J_batch, candidates, topk=1):
    """Return the best per-window candidate columns from dense window scores."""
    J_sums = np.sum(J_batch, axis=(1, 2), dtype=np.float32)
    scores = 0.25 * np.sum(candidates * np.matmul(J_batch, candidates), axis=1, dtype=np.float32)
    scores -= 0.25 * J_sums[:, None]
    batch_size = candidates.shape[2]
    topk = max(1, min(int(topk), batch_size))
    if topk == 1:
        best_indices = np.argmax(scores, axis=1)
        return [
            candidates[window_idx, :, best_indices[window_idx]].astype(np.int8, copy=True)
            for window_idx in range(J_batch.shape[0])
        ]

    top_indices = np.argpartition(scores, batch_size - topk, axis=1)[:, batch_size - topk:]
    top_scores = np.take_along_axis(scores, top_indices, axis=1)
    order = np.argsort(top_scores, axis=1)[:, ::-1]
    top_indices = np.take_along_axis(top_indices, order, axis=1)
    return [
        candidates[window_idx][:, top_indices[window_idx]].astype(np.int8, copy=True)
        for window_idx in range(J_batch.shape[0])
    ]


def _run_dense_qmf_sign_coupled_batched(
    J_batch,
    batch_size,
    n_iter,
    seeds,
    alpha=0.35,
    sigma=0.05,
    beta_final=1.35,
    x_scale=0.05,
):
    """
    Quantum-inspired sign-coupled mean-field kernel for experimental fast path.

    The measured spin sign(x) drives the coupling field, while x remains a
    continuous mean-field state annealed through tanh dynamics.
    """
    J_batch = np.asarray(J_batch, dtype=np.float32)
    num_windows, num_nodes, _ = J_batch.shape
    x = np.empty((num_windows, num_nodes, batch_size), dtype=np.float32)
    spin = np.empty_like(x)
    field = np.empty_like(x)
    x_hat = np.empty_like(x)
    noise = None
    noise_rngs = None
    sigma_value = float(sigma)
    if sigma_value > 0:
        noise = np.empty_like(x)
        noise_rngs = []

    scale = np.float32(x_scale)
    for window_idx, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        x[window_idx] = scale * rng.standard_normal((num_nodes, batch_size)).astype(np.float32)
        if noise_rngs is not None:
            noise_rngs.append(np.random.RandomState(seed))

    inv_norm = np.sqrt(np.sum(J_batch * J_batch, axis=2, keepdims=True, dtype=np.float32), dtype=np.float32)
    inv_norm[inv_norm == 0] = 1.0
    np.reciprocal(inv_norm, out=inv_norm)

    alpha = np.float32(alpha)
    one_minus_alpha = np.float32(1.0) - alpha
    beta = np.float32(1.0 / max(1, n_iter))
    beta_step = np.float32((float(beta_final) - float(beta)) / max(1, n_iter - 1))

    for _ in range(n_iter):
        np.sign(x, out=spin)
        spin[spin == 0] = 1.0
        np.matmul(J_batch, spin, out=field)
        field *= inv_norm
        if noise_rngs is not None:
            for window_idx, noise_rng in enumerate(noise_rngs):
                noise[window_idx] = noise_rng.normal(
                    0.0,
                    sigma_value,
                    size=(num_nodes, batch_size),
                ).astype(np.float32)
            field += noise
        np.multiply(field, beta, out=field)
        np.tanh(field, out=x_hat)
        x *= one_minus_alpha
        x += alpha * x_hat
        beta += beta_step

    np.sign(x, out=spin)
    spin[spin == 0] = 1.0
    return _score_dense_window_candidates(J_batch, spin, topk=1)


def _run_dense_amf_batched(
    J_batch,
    batch_size,
    n_iter,
    seeds,
    alpha=0.20,
    sigma=0.15,
    beta_final=1.25,
    x_scale=0.05,
):
    """Accelerated noisy analog mean-field kernel for experimental fast path."""
    J_batch = np.asarray(J_batch, dtype=np.float32)
    num_windows, num_nodes, _ = J_batch.shape
    x = np.empty((num_windows, num_nodes, batch_size), dtype=np.float32)
    noise = None
    sigma_value = float(sigma)
    if sigma_value > 0:
        noise = np.empty((num_windows, n_iter, num_nodes, batch_size), dtype=np.float32)

    if PRIMARY_AMF_DIVERSE and batch_size >= 4:
        schedule_axis = np.linspace(-1.0, 1.0, batch_size, dtype=np.float32)
        alpha_vec = np.asarray(alpha, dtype=np.float32) * (np.float32(1.0) + np.float32(0.28) * schedule_axis)
        alpha_vec = np.clip(alpha_vec, np.float32(0.08), np.float32(0.55)).reshape(1, 1, batch_size)
        sigma_vec = np.float32(sigma_value) * (np.float32(1.0) - np.float32(0.35) * schedule_axis)
        sigma_vec = np.clip(sigma_vec, np.float32(0.0), np.float32(0.35)).reshape(1, 1, batch_size)
        beta_final_vec = np.asarray(beta_final, dtype=np.float32) * (
            np.float32(1.0) + np.float32(0.35) * schedule_axis
        )
        beta_final_vec = np.clip(beta_final_vec, np.float32(0.65), np.float32(2.35)).reshape(1, 1, batch_size)
    else:
        alpha_vec = np.full((1, 1, batch_size), np.float32(alpha), dtype=np.float32)
        sigma_vec = np.full((1, 1, batch_size), np.float32(sigma_value), dtype=np.float32)
        beta_final_vec = np.full((1, 1, batch_size), np.float32(beta_final), dtype=np.float32)

    scale = np.float32(x_scale)
    for window_idx, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        x[window_idx] = scale * rng.standard_normal((num_nodes, batch_size)).astype(np.float32)
        if noise is not None:
            noise[window_idx] = np.random.RandomState(seed).normal(
                0.0,
                sigma_value,
                size=(n_iter, num_nodes, batch_size),
            ).astype(np.float32)
            if PRIMARY_AMF_DIVERSE:
                noise[window_idx] *= sigma_vec[0]

    inv_norm = np.sqrt(np.sum(J_batch * J_batch, axis=2, keepdims=True, dtype=np.float32), dtype=np.float32)
    inv_norm[inv_norm == 0] = 1.0
    np.reciprocal(inv_norm, out=inv_norm)

    beta_start = np.float32(1.0 / max(1, n_iter))
    if abs(PRIMARY_AMF_BETA_POWER - 1.0) > 1e-12 and n_iter > 1:
        beta_schedule = np.linspace(0.0, 1.0, n_iter, dtype=np.float32) ** np.float32(PRIMARY_AMF_BETA_POWER)
        beta_schedule = beta_start + beta_schedule.reshape(n_iter, 1, 1, 1) * (beta_final_vec - beta_start)
    else:
        beta_schedule = None
        beta_step = (beta_final_vec - beta_start) / np.float32(max(1, n_iter - 1))
        beta = np.full((1, 1, batch_size), beta_start, dtype=np.float32)

    one_minus_alpha = np.float32(1.0) - alpha_vec
    phi = np.empty_like(x)
    x_hat = np.empty_like(x)
    spin = np.empty_like(x)
    best_candidates = np.empty_like(x)
    best_scores = np.full((num_windows, batch_size), -np.inf, dtype=np.float32)
    J_sums = np.sum(J_batch, axis=(1, 2), dtype=np.float32)
    checkpoints = set()
    if PRIMARY_AMF_CHECKPOINTS and n_iter >= 16:
        checkpoints.add(max(1, (n_iter * 2) // 3) - 1)
        checkpoints.add(n_iter - 1)
    else:
        checkpoints.add(n_iter - 1)
    harden_start = n_iter + 1
    if PRIMARY_AMF_HARDEN_FRAC > 0:
        harden_start = max(0, min(n_iter - 1, int(n_iter * (1.0 - PRIMARY_AMF_HARDEN_FRAC))))

    for iter_idx in range(n_iter):
        if iter_idx >= harden_start:
            np.sign(x, out=spin)
            spin[spin == 0] = 1.0
            np.matmul(J_batch, spin, out=phi)
        else:
            np.matmul(J_batch, x, out=phi)
        phi *= inv_norm
        if noise is not None:
            phi += noise[:, iter_idx]
        if beta_schedule is not None:
            beta = beta_schedule[iter_idx]
        np.multiply(phi, beta, out=phi)
        np.tanh(phi, out=x_hat)
        x *= one_minus_alpha
        x += alpha_vec * x_hat
        if beta_schedule is None:
            beta += beta_step
        if iter_idx in checkpoints:
            candidates = np.sign(x, out=np.empty_like(x))
            candidates[candidates == 0] = 1.0
            scores = 0.25 * np.sum(candidates * np.matmul(J_batch, candidates), axis=1, dtype=np.float32)
            scores -= 0.25 * J_sums[:, None]
            improved = scores > best_scores
            best_scores[improved] = scores[improved]
            for window_idx in range(num_windows):
                improved_cols = improved[window_idx]
                if np.any(improved_cols):
                    best_candidates[window_idx, :, improved_cols] = candidates[window_idx, :, improved_cols]

    tail_n_iter = max(0, int(PRIMARY_AMF_TAIL_N_ITER))
    if tail_n_iter > 0:
        tail_beta = np.float32(1.0 / tail_n_iter)
        tail_beta_step = np.float32(1.0 / tail_n_iter)
        tail_alpha = np.float32(0.15)
        tail_one_minus_alpha = np.float32(1.0) - tail_alpha
        tail_noise = None
        tail_sigma_value = float(PRIMARY_AMF_TAIL_SIGMA)
        if tail_sigma_value > 0:
            tail_noise = np.empty((num_windows, tail_n_iter, num_nodes, batch_size), dtype=np.float32)
            for window_idx, seed in enumerate(seeds):
                tail_seed = int(seed) + 1000003
                tail_noise[window_idx] = np.random.RandomState(tail_seed).normal(
                    0.0,
                    tail_sigma_value,
                    size=(tail_n_iter, num_nodes, batch_size),
                ).astype(np.float32)
        for iter_idx in range(tail_n_iter):
            np.matmul(J_batch, x, out=phi)
            phi *= inv_norm
            if tail_noise is not None:
                phi += tail_noise[:, iter_idx]
            np.multiply(phi, tail_beta, out=phi)
            np.tanh(phi, out=x_hat)
            x *= tail_one_minus_alpha
            x += tail_alpha * x_hat
            tail_beta += tail_beta_step
        candidates = np.sign(x, out=np.empty_like(x))
        candidates[candidates == 0] = 1.0
        scores = 0.25 * np.sum(candidates * np.matmul(J_batch, candidates), axis=1, dtype=np.float32)
        scores -= 0.25 * J_sums[:, None]
        improved = scores > best_scores
        best_scores[improved] = scores[improved]
        for window_idx in range(num_windows):
            improved_cols = improved[window_idx]
            if np.any(improved_cols):
                best_candidates[window_idx, :, improved_cols] = candidates[window_idx, :, improved_cols]

    return _score_dense_window_candidates(J_batch, best_candidates, topk=PRIMARY_AMF_TOPK)


def _run_dense_amf_nmfa_hybrid_batched(
    J_batch,
    batch_size,
    n_iter,
    seeds,
    alpha=0.20,
    sigma=0.15,
    beta_final=1.25,
    x_scale=0.05,
):
    """
    AMF warm trajectory followed by a short NMFA-like correction tail.

    This is an experimental quantum-inspired hybrid kernel: AMF quickly finds
    a mean-field basin, then the NMFA tail re-injects stochastic tanh dynamics
    before measurement. It is separate from the v14b NMFA fast path.
    """
    J_batch = np.asarray(J_batch, dtype=np.float32)
    num_windows, num_nodes, _ = J_batch.shape
    x = np.empty((num_windows, num_nodes, batch_size), dtype=np.float32)
    sigma_value = float(sigma)
    noise = np.empty((num_windows, n_iter, num_nodes, batch_size), dtype=np.float32) if sigma_value > 0 else None

    if PRIMARY_AMF_DIVERSE and batch_size >= 4:
        schedule_axis = np.linspace(-1.0, 1.0, batch_size, dtype=np.float32)
        alpha_vec = np.asarray(alpha, dtype=np.float32) * (np.float32(1.0) + np.float32(0.28) * schedule_axis)
        alpha_vec = np.clip(alpha_vec, np.float32(0.08), np.float32(0.55)).reshape(1, 1, batch_size)
        sigma_vec = np.float32(sigma_value) * (np.float32(1.0) - np.float32(0.35) * schedule_axis)
        sigma_vec = np.clip(sigma_vec, np.float32(0.0), np.float32(0.35)).reshape(1, 1, batch_size)
        beta_final_vec = np.asarray(beta_final, dtype=np.float32) * (
            np.float32(1.0) + np.float32(0.35) * schedule_axis
        )
        beta_final_vec = np.clip(beta_final_vec, np.float32(0.65), np.float32(2.35)).reshape(1, 1, batch_size)
    else:
        alpha_vec = np.full((1, 1, batch_size), np.float32(alpha), dtype=np.float32)
        sigma_vec = np.full((1, 1, batch_size), np.float32(sigma_value), dtype=np.float32)
        beta_final_vec = np.full((1, 1, batch_size), np.float32(beta_final), dtype=np.float32)

    scale = np.float32(x_scale)
    tail_rngs = []
    for window_idx, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        x[window_idx] = scale * rng.standard_normal((num_nodes, batch_size)).astype(np.float32)
        if noise is not None:
            noise[window_idx] = np.random.RandomState(seed).normal(
                0.0,
                sigma_value,
                size=(n_iter, num_nodes, batch_size),
            ).astype(np.float32)
            if PRIMARY_AMF_DIVERSE:
                noise[window_idx] *= sigma_vec[0]
        tail_rngs.append(np.random.RandomState(int(seed) + 2000003))

    inv_norm = np.sqrt(np.sum(J_batch * J_batch, axis=2, keepdims=True, dtype=np.float32), dtype=np.float32)
    inv_norm[inv_norm == 0] = 1.0
    np.reciprocal(inv_norm, out=inv_norm)

    beta_start = np.float32(1.0 / max(1, n_iter))
    beta_step = (beta_final_vec - beta_start) / np.float32(max(1, n_iter - 1))
    beta = np.full((1, 1, batch_size), beta_start, dtype=np.float32)
    one_minus_alpha = np.float32(1.0) - alpha_vec
    phi = np.empty_like(x)
    x_hat = np.empty_like(x)

    for iter_idx in range(n_iter):
        np.matmul(J_batch, x, out=phi)
        phi *= inv_norm
        if noise is not None:
            phi += noise[:, iter_idx]
        np.multiply(phi, beta, out=phi)
        np.tanh(phi, out=x_hat)
        x *= one_minus_alpha
        x += alpha_vec * x_hat
        beta += beta_step

    tail_n_iter = max(0, int(PRIMARY_HYBRID_NMFA_N_ITER))
    if tail_n_iter > 0:
        tail_alpha = np.float32(PRIMARY_HYBRID_NMFA_ALPHA)
        tail_one_minus_alpha = np.float32(1.0) - tail_alpha
        tail_beta = np.float32(1.0 / tail_n_iter)
        tail_beta_final = np.float32(PRIMARY_HYBRID_NMFA_BETA_FINAL)
        tail_beta_step = (tail_beta_final - tail_beta) / np.float32(max(1, tail_n_iter - 1))
        tail_sigma = float(PRIMARY_HYBRID_NMFA_SIGMA)
        tail_noise = np.empty_like(x) if tail_sigma > 0 else None
        for _ in range(tail_n_iter):
            np.matmul(J_batch, x, out=phi)
            phi *= inv_norm
            if tail_noise is not None:
                for window_idx, noise_rng in enumerate(tail_rngs):
                    tail_noise[window_idx] = noise_rng.normal(
                        0.0,
                        tail_sigma,
                        size=(num_nodes, batch_size),
                    ).astype(np.float32)
                phi += tail_noise
            np.multiply(phi, tail_beta, out=phi)
            np.tanh(phi, out=x_hat)
            x *= tail_one_minus_alpha
            x += tail_alpha * x_hat
            tail_beta += tail_beta_step

    candidates = np.sign(x, out=np.empty_like(x))
    candidates[candidates == 0] = 1.0
    return _score_dense_window_candidates(J_batch, candidates, topk=PRIMARY_HYBRID_TOPK)


def _run_dense_dsb_batched(
    J_batch,
    batch_size,
    n_iter,
    seeds,
    dt=0.90,
    xi_scale=0.95,
):
    """Dense batched discrete simulated bifurcation kernel."""
    J_batch = np.asarray(J_batch, dtype=np.float32)
    num_windows, num_nodes, _ = J_batch.shape
    x = np.empty((num_windows, num_nodes, batch_size), dtype=np.float32)
    y = np.empty_like(x)
    spin = np.empty_like(x)
    field = np.empty_like(x)
    scale = np.float32(0.02)
    for window_idx, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        x[window_idx] = scale * (rng.random((num_nodes, batch_size), dtype=np.float32) - np.float32(0.5))
        y[window_idx] = scale * (rng.random((num_nodes, batch_size), dtype=np.float32) - np.float32(0.5))

    squared_sum = np.sum(J_batch * J_batch, axis=(1, 2), dtype=np.float32)
    squared_sum[squared_sum <= 0] = 1.0
    xi = np.float32(0.5 * np.sqrt(max(1, num_nodes - 1)) * float(xi_scale)) / np.sqrt(squared_sum)
    xi = xi.reshape(num_windows, 1, 1).astype(np.float32)
    dt = np.float32(dt)
    p_list = np.linspace(0.0, 1.0, n_iter, dtype=np.float32)

    for pump in p_list:
        np.sign(x, out=spin)
        spin[spin == 0] = 1.0
        np.matmul(J_batch, spin, out=field)
        y += (-(np.float32(1.0) - pump) * x + xi * field) * dt
        x += dt * y
        cond = np.abs(x) > 1.0
        if np.any(cond):
            x[cond] = np.sign(x[cond])
            y[cond] = 0.0

    np.sign(x, out=spin)
    spin[spin == 0] = 1.0
    return _score_dense_window_candidates(J_batch, spin, topk=PRIMARY_DSB_TOPK)


def _run_dense_simcim_batched(
    J_batch,
    batch_size,
    n_iter,
    seeds,
    dt=0.015,
    momentum=0.90,
    sigma=0.03,
    pt=6.5,
):
    """Dense batched SimCIM kernel for small window probes."""
    J_batch = np.asarray(J_batch, dtype=np.float32)
    num_windows, num_nodes, _ = J_batch.shape
    x = np.zeros((num_windows, num_nodes, batch_size), dtype=np.float32)
    dx = np.zeros_like(x)
    newdc = np.empty_like(x)
    noise = np.empty_like(x) if sigma > 0 else None
    noise_rngs = [np.random.RandomState(seed) for seed in seeds] if sigma > 0 else None
    p_list = (np.tanh(np.linspace(-3, 3, n_iter, dtype=np.float32)) - np.float32(1.0)) * np.float32(pt)
    dt = np.float32(dt)
    momentum = np.float32(momentum)
    sigma = float(sigma)

    for pump in p_list:
        np.matmul(J_batch, x, out=newdc)
        newdc *= dt
        newdc += x * pump
        if noise_rngs is not None:
            for window_idx, noise_rng in enumerate(noise_rngs):
                noise[window_idx] = noise_rng.normal(
                    size=(num_nodes, batch_size),
                ).astype(np.float32)
            newdc += noise * np.float32(sigma)
        dx *= momentum
        dx += newdc * (np.float32(1.0) - momentum)
        candidate = x + dx
        mask = np.abs(candidate) < 1.0
        x += dx * mask

    spin = np.sign(x, out=np.empty_like(x))
    spin[spin == 0] = 1.0
    return _score_dense_window_candidates(J_batch, spin, topk=PRIMARY_SIMCIM_TOPK)


def _build_window_tasks(num_nodes, window_size, overlap, seed0):
    """Build deterministic contiguous window tasks for decomposition."""
    stride = max(1, window_size - overlap)
    tasks = []
    seed = seed0
    for start in range(0, num_nodes, stride):
        end = min(num_nodes, start + window_size)
        tasks.append((start, end, seed))
        seed += 17
    return tasks


def _group_window_batches(task_list):
    """Group equal-sized windows into dense batched NMFA batches."""
    batches = []
    idx = 0
    while idx < len(task_list):
        window_len = task_list[idx][1] - task_list[idx][0]
        end_idx = idx + 1
        while (
            end_idx < len(task_list)
            and (task_list[end_idx][1] - task_list[end_idx][0]) == window_len
            and (end_idx - idx) < WINDOW_DENSE_BATCH
        ):
            end_idx += 1
        batches.append(task_list[idx:end_idx])
        idx = end_idx
    return batches


def _split_window_tasks_contiguous(tasks, chunk_count):
    """Split ordered window tasks into contiguous chunks."""
    if chunk_count <= 1 or len(tasks) <= 1:
        return [tasks]

    chunks = []
    task_count = len(tasks)
    base = task_count // chunk_count
    extra = task_count % chunk_count
    start = 0
    for chunk_idx in range(chunk_count):
        chunk_size = base + (1 if chunk_idx < extra else 0)
        if chunk_size <= 0:
            continue
        end = start + chunk_size
        chunks.append(tasks[start:end])
        start = end
    return chunks


def _matches_primary_window_nmfa_fast_cfg(cfg):
    """Return True for the primary staged window-NMFA configuration."""
    return (
        cfg.get('window_size') == 140
        and cfg.get('overlap') == 23
        and cfg.get('batch_size') == 8
        and cfg.get('n_iter') == 98
        and cfg.get('warmup_n_iter') == 85
        and cfg.get('seed') == 801
        and abs(float(cfg.get('alpha', 0.15)) - 0.15) <= 1e-12
        and abs(float(cfg.get('sigma', 0.15)) - 0.15) <= 1e-12
        and cfg.get('sigma_end') is None
        and abs(float(cfg.get('x_scale', 0.05)) - 0.05) <= 1e-12
    )


def _solve_primary_window_probe_fast(
    G_csr,
    tasks,
    task_chunks,
    worker_count,
    executor,
    kernel_name,
    num_nodes,
    overlap,
    batch_size,
    x_scale,
    graph_sum,
    target_cut,
    local_flips,
):
    """Run optional experimental quantum-inspired primary-window probe."""
    if kernel_name not in ('qmf_sign', 'amf', 'amf_nmfa', 'dsb_dense', 'simcim_dense'):
        return None

    def solve_probe_chunk(task_chunk, ensemble_start=0, ensemble_count=None):
        solved = []
        if ensemble_count is None:
            ensemble_count = max(1, int(PRIMARY_AMF_ENSEMBLE)) if kernel_name == 'amf' else 1
        for batch_tasks in _group_window_batches(task_chunk):
            batch_len = batch_tasks[0][1] - batch_tasks[0][0]
            if batch_len > 256:
                return None

            J_batch = _materialize_window_dense_batch(G_csr, batch_tasks)
            base_seeds = [seed for _, _, seed in batch_tasks]
            local_groups = []
            for local_ensemble_idx in range(max(1, ensemble_count)):
                ensemble_idx = ensemble_start + local_ensemble_idx
                seed_offset = 0 if ensemble_idx == 0 else 1009 * ensemble_idx
                seeds = [seed + seed_offset for seed in base_seeds]
                if kernel_name == 'qmf_sign':
                    local_groups.append(
                        _run_dense_qmf_sign_coupled_batched(
                            J_batch,
                            batch_size=batch_size,
                            n_iter=PRIMARY_QMF_N_ITER,
                            seeds=seeds,
                            alpha=PRIMARY_QMF_ALPHA,
                            sigma=PRIMARY_QMF_SIGMA,
                            beta_final=PRIMARY_QMF_BETA_FINAL,
                            x_scale=x_scale,
                        )
                    )
                elif kernel_name == 'amf':
                    local_groups.append(
                        _run_dense_amf_batched(
                            J_batch,
                            batch_size=batch_size,
                            n_iter=PRIMARY_AMF_N_ITER,
                            seeds=seeds,
                            alpha=PRIMARY_AMF_ALPHA,
                            sigma=PRIMARY_AMF_SIGMA,
                            beta_final=PRIMARY_AMF_BETA_FINAL,
                            x_scale=x_scale,
                        )
                    )
                elif kernel_name == 'amf_nmfa':
                    local_groups.append(
                        _run_dense_amf_nmfa_hybrid_batched(
                            J_batch,
                            batch_size=batch_size,
                            n_iter=PRIMARY_AMF_N_ITER,
                            seeds=seeds,
                            alpha=PRIMARY_AMF_ALPHA,
                            sigma=PRIMARY_AMF_SIGMA,
                            beta_final=PRIMARY_AMF_BETA_FINAL,
                            x_scale=x_scale,
                        )
                    )
                elif kernel_name == 'dsb_dense':
                    local_groups.append(
                        _run_dense_dsb_batched(
                            J_batch,
                            batch_size=batch_size,
                            n_iter=PRIMARY_DSB_N_ITER,
                            seeds=seeds,
                            dt=PRIMARY_DSB_DT,
                            xi_scale=PRIMARY_DSB_XI_SCALE,
                        )
                    )
                elif kernel_name == 'simcim_dense':
                    local_groups.append(
                        _run_dense_simcim_batched(
                            J_batch,
                            batch_size=batch_size,
                            n_iter=PRIMARY_SIMCIM_N_ITER,
                            seeds=seeds,
                            dt=PRIMARY_SIMCIM_DT,
                            momentum=PRIMARY_SIMCIM_MOMENTUM,
                            sigma=PRIMARY_SIMCIM_SIGMA,
                            pt=PRIMARY_SIMCIM_PT,
                        )
                    )

            if len(local_groups) == 1:
                local_partitions = local_groups[0]
            else:
                local_partitions = []
                for window_idx in range(len(batch_tasks)):
                    columns = []
                    for group in local_groups:
                        partition = group[window_idx]
                        if partition.ndim == 1:
                            columns.append(partition)
                        else:
                            columns.extend(partition[:, col] for col in range(partition.shape[1]))
                    local_partitions.append(np.column_stack(columns).astype(np.int8, copy=False))
            solved.append({
                'tasks': batch_tasks,
                'partitions': local_partitions,
            })
        return solved

    def solve_probe_chunks_once(ensemble_start=0, ensemble_count=None):
        def solve_chunk(task_chunk):
            return solve_probe_chunk(task_chunk, ensemble_start=ensemble_start, ensemble_count=ensemble_count)

        if worker_count <= 1:
            return [solve_chunk(task_chunks[0])]
        if executor is None:
            with ThreadPoolExecutor(max_workers=worker_count) as local_executor:
                return list(local_executor.map(solve_chunk, task_chunks))
        return list(executor.map(solve_chunk, task_chunks))

    def score_probe_chunks(solved_chunks):
        stitch_started = _profile_start()
        candidate_count = 1
        for chunk in solved_chunks:
            for batch in chunk:
                for local_partition in batch['partitions']:
                    if isinstance(local_partition, np.ndarray) and local_partition.ndim == 2:
                        candidate_count = max(candidate_count, local_partition.shape[1])

        stitched_list = []
        reference_stitched = None
        for candidate_idx in range(candidate_count):
            stitched = np.zeros(num_nodes, dtype=np.int8)
            for chunk in solved_chunks:
                for batch in chunk:
                    for (start, end, _), local_partition in zip(batch['tasks'], batch['partitions']):
                        if isinstance(local_partition, np.ndarray) and local_partition.ndim == 2:
                            local_values = local_partition[:, min(candidate_idx, local_partition.shape[1] - 1)]
                        else:
                            local_values = local_partition
                        _stitch_window_partition(stitched, overlap, start, end, local_values)
            stitched[stitched == 0] = 1
            if candidate_idx == 0:
                reference_stitched = stitched.copy()
            stitched_list.append(stitched)
        if PRIMARY_PROBE_ADAPTIVE_STITCH and candidate_count > 1:
            stitched_list.append(
                _choose_and_stitch_window_candidates(G_csr, num_nodes, overlap, solved_chunks)
            )
        fusion_result = None
        if kernel_name == 'amf' and PRIMARY_AMF_FUSION:
            fusion_started = _profile_start()
            if PRIMARY_AMF_FUSION_MODE == 'block':
                fusion_result = _fuse_window_candidates_by_block_ascent(
                    G_csr,
                    num_nodes,
                    solved_chunks,
                    graph_sum=graph_sum,
                    local_flips=local_flips,
                    target_cut=target_for_probe,
                    reference_partition=reference_stitched,
                )
            else:
                fusion_result = _fuse_window_candidates_with_amf(
                    G_csr,
                    num_nodes,
                    solved_chunks,
                    graph_sum=graph_sum,
                    local_flips=local_flips,
                    target_cut=target_for_probe,
                    reference_partition=reference_stitched,
                )
            if fusion_result is not None:
                stitched_list.append(fusion_result['stitched'])
                _profile_emit(
                    'fast_amf_fusion',
                    mode=PRIMARY_AMF_FUSION_MODE,
                    elapsed_ms=_profile_ms(fusion_started),
                    cut=fusion_result['cut'],
                    target_cut=target_for_probe,
                    segments=fusion_result['segment_count'],
                    max_labels=fusion_result['max_labels'],
                    candidates=fusion_result['candidate_count'],
                )

        stitch_ms = _profile_ms(stitch_started)
        score_started = _profile_start()
        candidates = [
            (f'subgraph-{kernel_name}-stitched-{candidate_idx}', stitched)
            for candidate_idx, stitched in enumerate(stitched_list)
        ]
        if fusion_result is not None:
            candidates.append(('subgraph-amf-global-fusion-repaired', fusion_result['partition']))
        partition, cut = _score_and_repair_candidates(
            G_csr,
            candidates,
            max_flips=local_flips,
            graph_sum=graph_sum,
            target_cut=target_cut,
        )
        return partition, cut, stitched_list, stitch_ms, _profile_ms(score_started)

    prepare_started = _profile_start()
    solved_chunks = solve_probe_chunks_once()
    if any(chunk is None for chunk in solved_chunks):
        return None

    warmup_ms = _profile_ms(prepare_started)
    target_for_probe = None
    if target_cut is not None:
        target_for_probe = target_cut + PRIMARY_PROBE_TARGET_MARGIN
    warm_partition, warm_cut, warm_stitched_list, stitch_ms, warm_score_ms = score_probe_chunks(solved_chunks)
    reached_target = target_for_probe is not None and warm_cut >= target_for_probe
    fast_exit_allowed = PRIMARY_FAST_EXIT and (
        target_cut is None or warm_cut >= target_cut * PRIMARY_FAST_EXIT_RATIO
    )
    _profile_emit(
        'fast_stitch',
        cfg_idx=0,
        elapsed_ms=stitch_ms,
        batches=sum(len(chunk) for chunk in solved_chunks),
        windows=len(tasks),
    )
    if reached_target or fast_exit_allowed:
        return {
            'warm_partition': warm_partition,
            'warm_cut': warm_cut,
            'warm_reached_target': reached_target,
            'final_partition': warm_partition,
            'final_cut': warm_cut,
            'stitched': warm_stitched_list[0],
            'fast_exit': fast_exit_allowed,
            'profile': {
                'warmup_ms': warmup_ms,
                'warm_score_ms': warm_score_ms,
                'continuation_ms': 0.0,
                'final_score_ms': 0.0,
                'windows': len(tasks),
                'batches': sum(len(chunk) for chunk in solved_chunks),
            },
        }

    if PRIMARY_BOUNDARY_REPAIR:
        repair_started = _profile_start()
        repaired_partition, repaired_cut = _qaia_boundary_repair(
            G_csr,
            warm_partition,
            graph_sum=graph_sum,
            window_size=tasks[0][1] - tasks[0][0],
            overlap=overlap,
            seed0=tasks[0][2],
            target_cut=target_for_probe,
            seed=11003,
        )
        _profile_emit(
            'fast_boundary_repair',
            kernel=kernel_name,
            elapsed_ms=_profile_ms(repair_started),
            before_cut=warm_cut,
            after_cut=repaired_cut,
            target_cut=target_for_probe,
        )
        if target_for_probe is not None and repaired_cut >= target_for_probe:
            return {
                'warm_partition': repaired_partition,
                'warm_cut': repaired_cut,
                'warm_reached_target': True,
                'final_partition': repaired_partition,
                'final_cut': repaired_cut,
                'stitched': warm_stitched_list[0],
                'fast_exit': False,
                'profile': {
                    'warmup_ms': warmup_ms,
                    'warm_score_ms': warm_score_ms + _profile_ms(repair_started),
                    'continuation_ms': 0.0,
                    'final_score_ms': 0.0,
                    'windows': len(tasks),
                    'batches': sum(len(chunk) for chunk in solved_chunks),
                },
            }

    if PRIMARY_PROBE_REPAIR:
        repair_started = _profile_start()
        repaired_partition, repaired_cut = _qaia_subproblem_repair(
            G_csr,
            warm_partition,
            graph_sum=graph_sum,
            target_cut=target_for_probe,
            seed=7001,
        )
        _profile_emit(
            'fast_probe_repair',
            kernel=kernel_name,
            elapsed_ms=_profile_ms(repair_started),
            before_cut=warm_cut,
            after_cut=repaired_cut,
            target_cut=target_for_probe,
        )
        if target_for_probe is not None and repaired_cut >= target_for_probe:
            return {
                'warm_partition': repaired_partition,
                'warm_cut': repaired_cut,
                'warm_reached_target': True,
                'final_partition': repaired_partition,
                'final_cut': repaired_cut,
                'stitched': warm_stitched_list[0],
                'fast_exit': False,
                'profile': {
                    'warmup_ms': warmup_ms,
                    'warm_score_ms': warm_score_ms + _profile_ms(repair_started),
                    'continuation_ms': 0.0,
                    'final_score_ms': 0.0,
                    'windows': len(tasks),
                    'batches': sum(len(chunk) for chunk in solved_chunks),
                },
            }

    selective_allowed = True
    if (
        target_for_probe is not None
        and PRIMARY_AMF_SELECTIVE_NMFA_MIN_RATIO > 0
        and warm_cut < target_for_probe * PRIMARY_AMF_SELECTIVE_NMFA_MIN_RATIO
    ):
        selective_allowed = False
        _profile_emit(
            'fast_selective_nmfa_skip',
            kernel=kernel_name,
            cut=warm_cut,
            target_cut=target_for_probe,
            min_ratio=PRIMARY_AMF_SELECTIVE_NMFA_MIN_RATIO,
        )

    if kernel_name == 'amf' and PRIMARY_AMF_SINGLE_WINDOW_REPAIR and selective_allowed:
        repair_started = _profile_start()
        single_result = _repair_probe_by_single_nmfa_window(
            G_csr,
            solved_chunks,
            num_nodes,
            overlap,
            warm_partition,
            graph_sum=graph_sum,
            local_flips=local_flips,
            target_cut=target_for_probe,
        )
        if single_result is not None:
            _profile_emit(
                'fast_single_window_nmfa',
                kernel=kernel_name,
                elapsed_ms=_profile_ms(repair_started),
                cut=single_result['cut'],
                target_cut=target_for_probe,
                selected=single_result['selected_count'],
                windows=single_result['window_count'],
            )
            if target_for_probe is not None and single_result['cut'] >= target_for_probe:
                return {
                    'warm_partition': single_result['partition'],
                    'warm_cut': single_result['cut'],
                    'warm_reached_target': True,
                    'final_partition': single_result['partition'],
                    'final_cut': single_result['cut'],
                    'stitched': single_result['stitched'],
                    'fast_exit': False,
                    'profile': {
                        'warmup_ms': warmup_ms + _profile_ms(repair_started),
                        'warm_score_ms': warm_score_ms,
                        'continuation_ms': 0.0,
                        'final_score_ms': 0.0,
                        'windows': len(tasks),
                        'batches': sum(len(chunk) for chunk in solved_chunks),
                    },
                }

    if kernel_name == 'amf' and PRIMARY_AMF_SELECTIVE_NMFA and selective_allowed:
        stage_specs = []
        if PRIMARY_AMF_SELECTIVE_NMFA_STAGES:
            for raw_stage in PRIMARY_AMF_SELECTIVE_NMFA_STAGES.split(','):
                raw_stage = raw_stage.strip()
                if not raw_stage:
                    continue
                if ':' in raw_stage:
                    mode, raw_topk = raw_stage.split(':', 1)
                    try:
                        stage_topk = int(raw_topk)
                    except ValueError:
                        stage_topk = PRIMARY_AMF_SELECTIVE_NMFA_TOPK
                    stage_specs.append((mode.strip(), stage_topk))
                else:
                    stage_specs.append((raw_stage, PRIMARY_AMF_SELECTIVE_NMFA_TOPK))
        if not stage_specs:
            stage_specs.append((PRIMARY_AMF_SELECTIVE_NMFA_SCORE, PRIMARY_AMF_SELECTIVE_NMFA_TOPK))

        for stage_mode, stage_topk in stage_specs:
            repair_started = _profile_start()
            selective_result = _repair_probe_windows_with_nmfa(
                G_csr,
                solved_chunks,
                num_nodes,
                overlap,
                warm_partition,
                graph_sum=graph_sum,
                local_flips=local_flips,
                target_cut=target_for_probe,
                score_mode=stage_mode,
                topk=stage_topk,
            )
            if selective_result is None:
                continue
            _profile_emit(
                'fast_selective_nmfa',
                kernel=kernel_name,
                mode=stage_mode,
                elapsed_ms=_profile_ms(repair_started),
                cut=selective_result['cut'],
                target_cut=target_for_probe,
                selected=selective_result['selected_count'],
                windows=selective_result['window_count'],
            )
            if target_for_probe is not None and selective_result['cut'] >= target_for_probe:
                return {
                    'warm_partition': selective_result['partition'],
                    'warm_cut': selective_result['cut'],
                    'warm_reached_target': True,
                    'final_partition': selective_result['partition'],
                    'final_cut': selective_result['cut'],
                    'stitched': selective_result['stitched'],
                    'fast_exit': False,
                    'profile': {
                        'warmup_ms': warmup_ms + _profile_ms(repair_started),
                        'warm_score_ms': warm_score_ms,
                        'continuation_ms': 0.0,
                        'final_score_ms': 0.0,
                        'windows': len(tasks),
                        'batches': sum(len(chunk) for chunk in solved_chunks),
                    },
                }

    if kernel_name == 'amf' and max(1, int(PRIMARY_AMF_ENSEMBLE)) > 1:
        adaptive_started = _profile_start()
        extra_chunks = solve_probe_chunks_once(ensemble_start=1, ensemble_count=1)
        if any(chunk is None for chunk in extra_chunks):
            return None
        merged_chunks = []
        for chunk, extra_chunk in zip(solved_chunks, extra_chunks):
            merged_chunk = []
            for batch, extra_batch in zip(chunk, extra_chunk):
                merged_partitions = []
                for local_partition, extra_partition in zip(batch['partitions'], extra_batch['partitions']):
                    if local_partition.ndim == 1:
                        local_partition = local_partition.reshape(-1, 1)
                    if extra_partition.ndim == 1:
                        extra_partition = extra_partition.reshape(-1, 1)
                    merged_partitions.append(np.column_stack((local_partition, extra_partition)))
                merged_chunk.append({
                    'tasks': batch['tasks'],
                    'partitions': merged_partitions,
                })
            merged_chunks.append(merged_chunk)
        warm_partition, warm_cut, warm_stitched_list, stitch_ms, warm_score_ms = score_probe_chunks(merged_chunks)
        warmup_ms += _profile_ms(adaptive_started)
        reached_target = target_for_probe is not None and warm_cut >= target_for_probe
        fast_exit_allowed = PRIMARY_FAST_EXIT and (
            target_cut is None or warm_cut >= target_cut * PRIMARY_FAST_EXIT_RATIO
        )
        if reached_target or fast_exit_allowed:
            return {
                'warm_partition': warm_partition,
                'warm_cut': warm_cut,
                'warm_reached_target': reached_target,
                'final_partition': warm_partition,
                'final_cut': warm_cut,
                'stitched': warm_stitched_list[0],
                'fast_exit': fast_exit_allowed,
                'profile': {
                    'warmup_ms': warmup_ms,
                    'warm_score_ms': warm_score_ms,
                    'continuation_ms': 0.0,
                    'final_score_ms': 0.0,
                    'windows': len(tasks),
                    'batches': sum(len(chunk) for chunk in merged_chunks),
                },
            }
        if PRIMARY_BOUNDARY_REPAIR:
            repair_started = _profile_start()
            repaired_partition, repaired_cut = _qaia_boundary_repair(
                G_csr,
                warm_partition,
                graph_sum=graph_sum,
                window_size=tasks[0][1] - tasks[0][0],
                overlap=overlap,
                seed0=tasks[0][2],
                target_cut=target_for_probe,
                seed=13003,
            )
            _profile_emit(
                'fast_boundary_repair',
                kernel=kernel_name,
                elapsed_ms=_profile_ms(repair_started),
                before_cut=warm_cut,
                after_cut=repaired_cut,
                target_cut=target_for_probe,
            )
            if target_for_probe is not None and repaired_cut >= target_for_probe:
                return {
                    'warm_partition': repaired_partition,
                    'warm_cut': repaired_cut,
                    'warm_reached_target': True,
                    'final_partition': repaired_partition,
                    'final_cut': repaired_cut,
                    'stitched': warm_stitched_list[0],
                    'fast_exit': False,
                    'profile': {
                        'warmup_ms': warmup_ms,
                        'warm_score_ms': warm_score_ms + _profile_ms(repair_started),
                        'continuation_ms': 0.0,
                        'final_score_ms': 0.0,
                        'windows': len(tasks),
                        'batches': sum(len(chunk) for chunk in merged_chunks),
                    },
                }
        if PRIMARY_PROBE_REPAIR:
            repair_started = _profile_start()
            repaired_partition, repaired_cut = _qaia_subproblem_repair(
                G_csr,
                warm_partition,
                graph_sum=graph_sum,
                target_cut=target_for_probe,
                seed=9001,
            )
            _profile_emit(
                'fast_probe_repair',
                kernel=kernel_name,
                elapsed_ms=_profile_ms(repair_started),
                before_cut=warm_cut,
                after_cut=repaired_cut,
                target_cut=target_for_probe,
            )
            if target_for_probe is not None and repaired_cut >= target_for_probe:
                return {
                    'warm_partition': repaired_partition,
                    'warm_cut': repaired_cut,
                    'warm_reached_target': True,
                    'final_partition': repaired_partition,
                    'final_cut': repaired_cut,
                    'stitched': warm_stitched_list[0],
                    'fast_exit': False,
                    'profile': {
                        'warmup_ms': warmup_ms,
                        'warm_score_ms': warm_score_ms + _profile_ms(repair_started),
                        'continuation_ms': 0.0,
                        'final_score_ms': 0.0,
                        'windows': len(tasks),
                        'batches': sum(len(chunk) for chunk in merged_chunks),
                    },
                }

    _profile_emit(
        'fast_probe_miss',
        kernel=kernel_name,
        cut=warm_cut,
        target_cut=target_for_probe,
        elapsed_ms=warmup_ms + warm_score_ms,
    )
    return None


def _solve_primary_window_nmfa_fast(
    G_csr,
    cfg,
    graph_sum,
    target_cut,
    local_flips,
    executor=None,
):
    """Dedicated fast path for the primary staged 140/23 window-NMFA pass."""
    if not _matches_primary_window_nmfa_fast_cfg(cfg):
        return None

    num_nodes = G_csr.shape[0]
    tasks = _build_window_tasks(num_nodes, cfg['window_size'], cfg['overlap'], cfg['seed'])
    if not tasks:
        return None

    batch_size = cfg['batch_size']
    n_iter = cfg['n_iter']
    warmup_n_iter = cfg['warmup_n_iter']
    overlap = cfg['overlap']
    alpha = cfg.get('alpha', 0.15)
    sigma = cfg.get('sigma', 0.15)
    x_scale = cfg.get('x_scale', 0.05)
    worker_count = min(WINDOW_QAIA_WORKERS, len(tasks))
    task_chunks = _split_window_tasks_contiguous(tasks, worker_count)
    kernel_name = PRIMARY_FAST_KERNEL

    def prepare_chunk(task_chunk):
        prepared = []
        for batch_tasks in _group_window_batches(task_chunk):
            batch_len = batch_tasks[0][1] - batch_tasks[0][0]
            if batch_len > 256:
                return None

            J_batch = _materialize_window_dense_batch(G_csr, batch_tasks)
            state = _make_dense_nmfa_batched_state(
                J_batch,
                batch_size=batch_size,
                n_iter=n_iter,
                seeds=[seed for _, _, seed in batch_tasks],
                alpha=alpha,
                sigma=sigma,
                x_scale=x_scale,
                initial_noise_iters=warmup_n_iter,
            )
            _advance_dense_nmfa_batched_state(state, warmup_n_iter)
            prepared.append({
                'tasks': batch_tasks,
                'state': state,
            })
        return prepared

    if kernel_name != 'nmfa':
        probe_result = _solve_primary_window_probe_fast(
            G_csr,
            tasks,
            task_chunks,
            worker_count,
            executor,
            kernel_name,
            num_nodes,
            overlap,
            batch_size,
            x_scale,
            graph_sum,
            target_cut,
            local_flips,
        )
        if probe_result is not None:
            return probe_result

    def stitch_prepared(prepared_chunks):
        started = _profile_start()
        partition = np.zeros(num_nodes, dtype=np.int8)
        for chunk in prepared_chunks:
            for batch in chunk:
                for (start, end, _), local_partition in zip(
                    batch['tasks'],
                    _decode_dense_nmfa_batched_state(batch['state']),
                ):
                    _stitch_window_partition(partition, overlap, start, end, local_partition)
        partition[partition == 0] = 1
        _profile_emit(
            'fast_stitch',
            cfg_idx=0,
            elapsed_ms=_profile_ms(started),
            batches=sum(len(chunk) for chunk in prepared_chunks),
            windows=len(tasks),
        )
        return partition

    prepare_started = _profile_start()
    if worker_count <= 1:
        prepared_chunks = [prepare_chunk(task_chunks[0])]
    else:
        if executor is None:
            with ThreadPoolExecutor(max_workers=worker_count) as local_executor:
                prepared_chunks = list(local_executor.map(prepare_chunk, task_chunks))
        else:
            prepared_chunks = list(executor.map(prepare_chunk, task_chunks))

    if any(chunk is None for chunk in prepared_chunks):
        return None

    warmup_ms = _profile_ms(prepare_started)
    warm_stitched = stitch_prepared(prepared_chunks)
    score_started = _profile_start()
    warm_partition, warm_cut = _score_and_repair_candidates(
        G_csr,
        [('subgraph-nmfa-stitched-warm', warm_stitched)],
        max_flips=local_flips,
        graph_sum=graph_sum,
        target_cut=target_cut,
    )
    warm_score_ms = _profile_ms(score_started)
    if target_cut is not None and warm_cut >= target_cut:
        return {
            'warm_partition': warm_partition,
            'warm_cut': warm_cut,
            'warm_reached_target': True,
            'profile': {
                'warmup_ms': warmup_ms,
                'warm_score_ms': warm_score_ms,
                'continuation_ms': 0.0,
                'final_score_ms': 0.0,
                'windows': len(tasks),
                'batches': sum(len(chunk) for chunk in prepared_chunks),
            },
        }

    remaining_iters = n_iter - warmup_n_iter
    continuation_started = _profile_start()
    if remaining_iters > 0:
        def advance_chunk(chunk):
            for batch in chunk:
                _advance_dense_nmfa_batched_state(batch['state'], remaining_iters)
            return None

        if worker_count <= 1:
            advance_chunk(prepared_chunks[0])
        else:
            if executor is None:
                with ThreadPoolExecutor(max_workers=worker_count) as local_executor:
                    list(local_executor.map(advance_chunk, prepared_chunks))
            else:
                list(executor.map(advance_chunk, prepared_chunks))

    continuation_ms = _profile_ms(continuation_started)
    stitched = stitch_prepared(prepared_chunks)
    score_started = _profile_start()
    final_partition, final_cut = _score_and_repair_candidates(
        G_csr,
        [('subgraph-nmfa-stitched', stitched)],
        max_flips=local_flips,
        graph_sum=graph_sum,
        target_cut=target_cut,
    )
    final_score_ms = _profile_ms(score_started)

    return {
        'warm_partition': warm_partition,
        'warm_cut': warm_cut,
        'warm_reached_target': False,
        'final_partition': final_partition,
        'final_cut': final_cut,
        'stitched': stitched,
        'profile': {
            'warmup_ms': warmup_ms,
            'warm_score_ms': warm_score_ms,
            'continuation_ms': continuation_ms,
            'final_score_ms': final_score_ms,
            'windows': len(tasks),
            'batches': sum(len(chunk) for chunk in prepared_chunks),
        },
    }


def _solve_window_batch(
    G_csr,
    batch_tasks,
    batch_size,
    n_iter,
    alpha=0.15,
    sigma=0.15,
    sigma_end=None,
    x_scale=0.05,
):
    """Solve one or more windows while preserving QAIA as the actual optimizer."""
    sigma_matches = sigma_end is None or abs(float(sigma_end) - float(sigma)) <= 1e-12
    if len(batch_tasks) <= 1 or not sigma_matches:
        return [
            (
                start,
                end,
                _run_nmfa(
                    G_csr[start:end, start:end].tocsr(),
                    batch_size=batch_size,
                    n_iter=n_iter,
                    seed=seed,
                    alpha=alpha,
                    sigma=sigma,
                    sigma_end=sigma_end,
                    x_scale=x_scale,
                ),
            )
            for start, end, seed in batch_tasks
        ]

    window_len = batch_tasks[0][1] - batch_tasks[0][0]
    if window_len > 256 or any((end - start) != window_len for start, end, _ in batch_tasks):
        return [
            (
                start,
                end,
                _run_nmfa(
                    G_csr[start:end, start:end].tocsr(),
                    batch_size=batch_size,
                    n_iter=n_iter,
                    seed=seed,
                    alpha=alpha,
                    sigma=sigma,
                    sigma_end=sigma_end,
                    x_scale=x_scale,
                ),
            )
            for start, end, seed in batch_tasks
        ]

    J_batch = _materialize_window_dense_batch(G_csr, batch_tasks)
    local_partitions = _run_dense_nmfa_batched(
        J_batch,
        batch_size=batch_size,
        n_iter=n_iter,
        seeds=[seed for _, _, seed in batch_tasks],
        alpha=alpha,
        sigma=sigma,
        x_scale=x_scale,
    )
    return [
        (start, end, local_partition)
        for (start, end, _), local_partition in zip(batch_tasks, local_partitions)
    ]


def _build_structure_couplings(G_csr, keep_ratio, noise_seed):
    """
    Build a sparse coupling-only enhancement graph from strong original edges.

    The result is never decoded into a partition. It is only added to the QAIA
    search Hamiltonian so strong couplings have a larger influence on dynamics.
    """
    num_nodes = G_csr.shape[0]
    if keep_ratio <= 0 or G_csr.nnz == 0:
        return csr_matrix(G_csr.shape, dtype=G_csr.dtype)

    G_coo = G_csr.tocoo(copy=False)
    upper_mask = G_coo.row < G_coo.col
    rows = G_coo.row[upper_mask]
    cols = G_coo.col[upper_mask]
    data = G_coo.data[upper_mask]
    edge_count = data.size
    if edge_count == 0:
        return csr_matrix(G_csr.shape, dtype=G_csr.dtype)

    keep_count = int(np.ceil(edge_count * keep_ratio))
    keep_count = min(max(1, keep_count), edge_count)
    strength = np.abs(data)

    if keep_count < edge_count:
        rng = np.random.default_rng(noise_seed)
        jitter_scale = (float(strength.max()) if strength.size else 1.0) * 1e-12
        ranking = strength + jitter_scale * rng.random(edge_count)
        selected = np.argpartition(ranking, edge_count - keep_count)[edge_count - keep_count:]
    else:
        selected = np.arange(edge_count)

    sel_rows = rows[selected]
    sel_cols = cols[selected]
    sel_data = data[selected]
    struct_rows = np.concatenate((sel_rows, sel_cols))
    struct_cols = np.concatenate((sel_cols, sel_rows))
    struct_data = np.concatenate((sel_data, sel_data))
    structure = csr_matrix((struct_data, (struct_rows, struct_cols)), shape=(num_nodes, num_nodes))
    structure.sort_indices()
    return structure


def _make_structured_search_graph(G_csr, keep_ratio, structure_lambda, seed):
    """Return J_search = G_original + lambda * G_structure for QAIA dynamics."""
    if structure_lambda == 0 or keep_ratio <= 0:
        return G_csr

    structure = _build_structure_couplings(G_csr, keep_ratio=keep_ratio, noise_seed=seed)
    if structure.nnz == 0:
        return G_csr

    J_search = (G_csr + structure_lambda * structure).tocsr()
    J_search.sort_indices()
    return J_search


def _prepare_x_init(num_nodes, batch_size, seed, x=None, noise_scale=0.05, random_when_none=False):
    """Prepare QAIA initial states without turning them into final answers."""
    rng = np.random.default_rng(seed)
    if x is None:
        if random_when_none:
            return noise_scale * rng.standard_normal((num_nodes, batch_size)).astype(np.float32)
        return None

    x_arr = np.asarray(x, dtype=np.float32)
    if x_arr.ndim == 1:
        base = x_arr.reshape(num_nodes, 1)
    else:
        base = x_arr.reshape(num_nodes, -1)

    if base.shape[1] >= batch_size:
        x_init = base[:, :batch_size].copy()
    else:
        repeats = int(np.ceil(batch_size / base.shape[1]))
        x_init = np.tile(base, (1, repeats))[:, :batch_size].copy()

    max_abs = float(np.max(np.abs(base))) if base.size else 0.0
    noise = noise_scale * rng.standard_normal((num_nodes, batch_size)).astype(np.float32)
    if max_abs >= 0.5 and batch_size > 1:
        x_init[:, 1:] *= 0.2
        x_init[:, 1:] += noise[:, 1:]
    else:
        x_init += noise
    x_init[:, 0] = base[:, 0]
    return x_init


def _resolve_sb_xi(J, xi_scale):
    """Scale MindQuantum SB's default xi without introducing per-instance branching."""
    if xi_scale is None:
        return None

    squared_sum = float(csr_matrix.power(csr_matrix(J), 2).sum())
    if squared_sum <= 0:
        return None

    return float(0.5 * np.sqrt(max(1, J.shape[0] - 1)) / np.sqrt(squared_sum) * xi_scale)


def _run_bsb_candidates(J, batch_size, n_iter, seed, x=None, dt=1.0, xi=None):
    """Run BSB and return all batch partitions generated by QAIA dynamics."""
    np.random.seed(seed)
    x_init = _prepare_x_init(J.shape[0], batch_size, seed, x=x, random_when_none=False)
    solver = BSB(J, x=x_init, n_iter=n_iter, batch_size=batch_size, dt=dt, xi=xi)
    solver.update()
    return _normalize_candidate_matrix(solver.x)


def _run_dsb_candidates(J, batch_size, n_iter, seed, x=None, dt=1.0, xi=None):
    """Run DSB and return all batch partitions generated by QAIA dynamics."""
    np.random.seed(seed)
    x_init = _prepare_x_init(J.shape[0], batch_size, seed, x=x, random_when_none=False)
    solver = DSB(J, x=x_init, n_iter=n_iter, batch_size=batch_size, dt=dt, xi=xi)
    solver.update()
    return _normalize_candidate_matrix(solver.x)


def _run_nmfa_candidates(J, batch_size, n_iter, seed, x=None, alpha=0.15, sigma=0.15, x_scale=0.05):
    """Run NMFA and return all batch partitions generated by QAIA dynamics."""
    np.random.seed(seed)
    x_init = _prepare_x_init(
        J.shape[0],
        batch_size,
        seed,
        x=x,
        noise_scale=x_scale,
        random_when_none=True,
    )
    solver = NMFA(J, x=x_init, n_iter=n_iter, batch_size=batch_size, alpha=alpha, sigma=sigma)
    solver.update()
    return _normalize_candidate_matrix(solver.x)


def _run_aqmf_primary_solver(G_csr):
    """Self-written Annealed Quantum Mean Field primary solver.

    AQMF evolves continuous spin amplitudes under annealed mean-field dynamics
    with pump, damping, tanh measurement, and nonlinear saturation. The
    triangular sweep is only an update schedule for the analog state; the final
    partition is obtained by measuring the evolved amplitudes.
    """

    n = G_csr.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int8)
    indptr = G_csr.indptr
    indices = G_csr.indices
    data = G_csr.data
    row_nnz = np.diff(indptr)
    row_ids = np.repeat(np.arange(n, dtype=indices.dtype), row_nnz)
    upper_mask = indices >= row_ids
    upper_cols = indices[upper_mask]
    upper_data = data[upper_mask]
    counts = np.zeros(n, dtype=indptr.dtype)
    if row_ids.size:
        nonempty_rows = row_nnz > 0
        counts[nonempty_rows] = np.add.reduceat(upper_mask.view(np.int8), indptr[:-1][nonempty_rows])
    upper_indptr = np.empty(n + 1, dtype=indptr.dtype)
    upper_indptr[0] = 0
    np.cumsum(counts, out=upper_indptr[1:])
    confidence = np.multiply(upper_data, upper_data)
    confidence *= -1.0
    confidence[upper_data >= 0.0] *= -0.9
    norm_per_row = np.zeros(n, dtype=np.float64)
    if confidence.size:
        nonempty_upper = counts > 0
        norm_per_row[nonempty_upper] = np.add.reduceat(
            np.abs(confidence),
            upper_indptr[:-1][nonempty_upper],
        )
    norm_per_row += 1e-12

    x = np.full(n, 0.05, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    rounds = max(1, int(AQMF_SWEEP_ROUNDS))
    tanh = np.tanh
    scalar_tanh = math.tanh
    dot = np.dot
    for sweep in range(rounds):
        progress = (sweep + 1.0) / rounds
        beta = 1.25 + 1.75 * progress
        pump = -0.20 + 1.05 * progress
        damping = 0.58 + 0.10 * progress
        analog_state = tanh(beta * x)
        for node in range(n - 1, -1, -1):
            start = upper_indptr[node]
            end = upper_indptr[node + 1]
            if end > start:
                force = dot(confidence[start:end], analog_state[upper_cols[start:end]]) / norm_per_row[node]
                x_node = x[node]
                y_node = damping * y[node] + 0.74 * force + pump * x_node - 0.18 * x_node * x_node * x_node
                x_node += y_node
                if x_node > 1.7:
                    x_node = 1.7
                elif x_node < -1.7:
                    x_node = -1.7
                y[node] = y_node
                x[node] = x_node
                analog_state[node] = scalar_tanh(beta * x_node)
    spins = np.ones(n, dtype=np.int8)
    spins[x < 0.0] = -1
    return spins




def _aqmf_internal_analog_refine(G_csr, s0, graph_sum, n_iter=2, seed=21):
    rng = np.random.default_rng(seed)
    n = G_csr.shape[0]
    x = s0.astype(np.float64) + 0.15 * rng.standard_normal(n)
    y = 0.02 * rng.standard_normal(n)
    abs_degree = np.zeros(n, dtype=np.float64)
    row_nnz = np.diff(G_csr.indptr)
    if G_csr.data.size:
        nonempty_rows = row_nnz > 0
        abs_degree[nonempty_rows] = np.add.reduceat(np.abs(G_csr.data), G_csr.indptr[:-1][nonempty_rows])
    abs_degree += 1e-12
    p_values = (0.0, 1.0) if n_iter == 2 else np.linspace(0.0, 1.0, n_iter, dtype=np.float64)
    tanh = np.tanh
    clip = np.clip
    multiply = np.multiply
    graph_dot = G_csr.dot
    for p in p_values:
        spin = multiply(x, 2.0)
        tanh(spin, out=spin)
        force = graph_dot(spin)
        force /= abs_degree
        y += (-(1.0 - p) * x + force)
        x += 0.7 * y
        clip(x, -1.5, 1.5, out=x)
    signs = np.ones(n, dtype=np.int8)
    signs[x < 0.0] = -1
    field = graph_dot(signs)
    cut = float(0.25 * np.dot(signs, field) - 0.25 * graph_sum)
    return signs, cut


def _score_partition_fast(G_csr, partition, graph_sum):
    spins = np.asarray(partition).reshape(-1)
    field = G_csr.dot(spins)
    return float(0.25 * np.dot(spins, field) - 0.25 * graph_sum)


def _try_aqmf_primary(G_csr, graph_sum, target_cut):
    if not AQMF_ENABLED or target_cut is None:
        return None
    started = _profile_start()
    aqmf = _run_aqmf_primary_solver(G_csr)
    aqmf_cut = _score_partition_fast(G_csr, aqmf, graph_sum)
    best_partition = aqmf
    best_cut = aqmf_cut
    best_stage = 'aqmf'
    nmfa_cut = float('nan')
    nmfa_improvement = 0.0
    nmfa_won = False
    if AQMF_NMFA_COMPETE and AQMF_NMFA_BATCH > 0 and AQMF_NMFA_N_ITER > 0:
        nmfa_started = _profile_start()
        nmfa_candidates = _run_nmfa_candidates(
            G_csr,
            batch_size=max(1, AQMF_NMFA_BATCH),
            n_iter=max(1, AQMF_NMFA_N_ITER),
            seed=AQMF_NMFA_SEED,
            x=aqmf,
            alpha=AQMF_NMFA_ALPHA,
            sigma=AQMF_NMFA_SIGMA,
            x_scale=AQMF_NMFA_X_SCALE,
        )
        nmfa_best = aqmf
        nmfa_best_cut = aqmf_cut
        nmfa_matrix = _normalize_candidate_matrix(nmfa_candidates)
        for col in range(nmfa_matrix.shape[1]):
            candidate = nmfa_matrix[:, col].astype(np.int8, copy=False)
            candidate_cut = _score_partition_fast(G_csr, candidate, graph_sum)
            if candidate_cut > nmfa_best_cut:
                nmfa_best = candidate
                nmfa_best_cut = candidate_cut
        nmfa_cut = float(nmfa_best_cut)
        nmfa_improvement = float(nmfa_best_cut - aqmf_cut)
        if nmfa_best_cut > best_cut:
            best_partition = nmfa_best.astype(np.int8, copy=True)
            best_cut = float(nmfa_best_cut)
            best_stage = 'aqmf-nmfa'
            nmfa_won = True
        _profile_emit(
            'aqmf_nmfa_compete',
            elapsed_ms=_profile_ms(nmfa_started),
            aqmf_cut=aqmf_cut,
            nmfa_cut=nmfa_cut,
            improvement=nmfa_improvement,
            won=nmfa_won,
            batch_size=AQMF_NMFA_BATCH,
            n_iter=AQMF_NMFA_N_ITER,
        )
    if best_cut < target_cut:
        refined, refined_cut = _aqmf_internal_analog_refine(G_csr, aqmf, graph_sum)
        if refined_cut > best_cut:
            best_partition = refined
            best_cut = refined_cut
            best_stage = 'aqmf-analog'
    if (
        best_cut < target_cut
        and AQMF_QAIA_REPAIR_ENABLED
        and target_cut - best_cut >= AQMF_QAIA_REPAIR_MIN_GAP
    ):
        qaia_repair_started = _profile_start()
        qaia_repaired, qaia_repaired_cut = _aqmf_risk_subproblem_repair(
            G_csr,
            best_partition,
            graph_sum,
            target_cut=target_cut,
        )
        if qaia_repaired_cut > best_cut:
            best_partition = qaia_repaired
            best_cut = qaia_repaired_cut
            best_stage = 'aqmf-qaiarepair'
        _profile_emit(
            'aqmf_qaiarepair',
            elapsed_ms=_profile_ms(qaia_repair_started),
            cut=qaia_repaired_cut,
            improvement=qaia_repaired_cut - aqmf_cut,
            target_cut=target_cut,
            reached=qaia_repaired_cut >= target_cut,
        )
    if best_cut < target_cut:
        repaired, repaired_cut, _field, _gain, _flips = _steepest_ascent_local_search(
            G_csr,
            best_partition,
            max_flips=min(max(0, AQMF_LOCAL_REPAIR_FLIPS), max(1, G_csr.shape[0] // 512)),
            graph_sum=graph_sum,
            target_cut=target_cut,
        )
        if repaired_cut > best_cut:
            best_partition = repaired
            best_cut = repaired_cut
            best_stage = 'aqmf-repair'
    _profile_emit(
        'aqmf_primary',
        elapsed_ms=_profile_ms(started),
        aqmf_cut=aqmf_cut,
        nmfa_cut=nmfa_cut,
        nmfa_improvement=nmfa_improvement,
        nmfa_won=nmfa_won,
        cut=best_cut,
        winner=best_stage,
        target_cut=target_cut,
        reached=best_cut >= target_cut,
    )
    if best_cut >= target_cut:
        return best_partition.astype(np.int8, copy=False), float(best_cut), best_stage
    return None


def _decompose_with_qaia(
    G_csr,
    window_size,
    overlap,
    batch_size,
    n_iter,
    seed0,
    alpha=0.15,
    sigma=0.15,
    sigma_end=None,
    x_scale=0.05,
    executor=None,
):
    """
    Classical graph decomposition + per-subgraph QAIA solving.

    This keeps the principal search in QAIA while only using classical logic to
    define the subgraph boundaries.
    """
    num_nodes = G_csr.shape[0]
    tasks = _build_window_tasks(num_nodes, window_size, overlap, seed0)

    def solve_task_list(task_list):
        solved_windows = []
        for batch_tasks in _group_window_batches(task_list):
            solved_windows.extend(
                _solve_window_batch(
                    G_csr,
                    batch_tasks,
                    batch_size=batch_size,
                    n_iter=n_iter,
                    alpha=alpha,
                    sigma=sigma,
                    sigma_end=sigma_end,
                    x_scale=x_scale,
                )
            )
        return solved_windows

    worker_count = min(WINDOW_QAIA_WORKERS, len(tasks))
    if worker_count <= 1:
        solved_windows = solve_task_list(tasks)
    else:
        indexed_tasks = list(enumerate(tasks))
        task_chunks = _split_window_tasks_contiguous(indexed_tasks, worker_count)

        def solve_chunk(indexed_chunk):
            chunk_results = solve_task_list([task for _, task in indexed_chunk])
            return [
                (task_index, result)
                for (task_index, _), result in zip(indexed_chunk, chunk_results)
            ]

        if executor is None:
            with ThreadPoolExecutor(max_workers=worker_count) as local_executor:
                chunk_results = list(local_executor.map(solve_chunk, task_chunks))
        else:
            chunk_results = list(executor.map(solve_chunk, task_chunks))

        solved_windows = [
            result
            for task_index, result in sorted(
                (item for chunk in chunk_results for item in chunk),
                key=lambda item: item[0],
            )
        ]

    return _stitch_solved_windows(num_nodes, overlap, solved_windows)


'''
必须基于 MindQuantum 量子计算框架实现。允许调用 MindQuantum 官方算法库，或基于该框架编写自定义逻辑。
'''
def maxcut_solver(G_csr, device='cpu', max_iterations=5, cut_value_baseline=None):
    """
    Solves the MAX-CUT problem for a single graph instance.
    Returns:
        partition (numpy.ndarray): The best partition found (+1/-1 array of shape [N]).
    """
    if sparse.isspmatrix_csr(G_csr):
        G_work = G_csr if G_csr.has_sorted_indices else G_csr.copy()
    else:
        G_work = G_csr.tocsr(copy=False)
    if not G_work.has_sorted_indices:
        G_work.sort_indices()

    graph_sum = float(np.sum(G_work.data, dtype=np.float64))
    num_nodes = G_work.shape[0]
    local_flips = min(LOCAL_REPAIR_FLIP_CAP, max(1, num_nodes // 4))

    best_partition = np.ones(num_nodes, dtype=np.int8)
    best_cut = -float('inf')
    target_cut = (
        cut_value_baseline - 1e-2
        if cut_value_baseline is not None and cut_value_baseline != 0
        else None
    )
    solver_started = _profile_start()
    first_target_stage = None
    best_label = None
    _profile_emit(
        'solver_begin',
        nodes=num_nodes,
        nnz=G_work.nnz,
        target_cut=target_cut,
        max_iterations=max_iterations,
    )

    def finish(return_stage):
        _profile_emit(
            'solver_end',
            return_stage=return_stage,
            elapsed_ms=_profile_ms(solver_started),
            best_cut=best_cut,
            target_cut=target_cut,
            first_target=first_target_stage,
        )
        return best_partition

    def consider_scored(candidate, candidate_cut, label='unknown'):
        nonlocal best_partition, best_cut, first_target_stage, best_label
        if candidate_cut > best_cut:
            best_cut = candidate_cut
            best_partition = candidate.astype(np.int8, copy=True)
            best_label = label
        if target_cut is not None and best_cut >= target_cut:
            if first_target_stage is None:
                first_target_stage = label
                _profile_emit(
                    'target_reached',
                    hit_stage=label,
                    best_cut=best_cut,
                    target_cut=target_cut,
                    elapsed_ms=_profile_ms(solver_started),
                )
            return True
        return False

    def consider_with_cut(label, spin_values):
        candidate, candidate_cut = _score_and_repair_candidates(
            G_work,
            [(label, spin_values)],
            max_flips=local_flips,
            graph_sum=graph_sum,
            target_cut=target_cut,
        )
        return consider_scored(candidate, candidate_cut, label=label), candidate_cut

    def consider(label, spin_values):
        return consider_with_cut(label, spin_values)[0]

    def label_cfg_index(label):
        if not label:
            return None
        for token in reversed(str(label).split('-')):
            if token.isdigit():
                return int(token)
        return None

    def selective_refine_allowed():
        if not SELECTIVE_REFINE_ENABLED or target_cut is None or not np.isfinite(best_cut):
            return False
        gap = target_cut - best_cut
        if gap <= 0.0:
            return False
        ratio_gap = gap / max(1.0, abs(target_cut))
        return gap <= SELECTIVE_REFINE_GAP_ABS and ratio_gap <= SELECTIVE_REFINE_GAP_RATIO

    def run_selective_nmfa_refine():
        if not selective_refine_allowed():
            return False
        source_idx = label_cfg_index(best_label)
        if source_idx is None:
            preferred = [1, 3]
        else:
            preferred = [source_idx + 1, source_idx, source_idx + 2, source_idx - 1]
        tried = set()
        executed = 0
        for ref_idx in preferred:
            if executed >= max(0, SELECTIVE_REFINE_MAX):
                break
            if ref_idx in tried or ref_idx < 0 or ref_idx >= len(stitched_results):
                continue
            tried.add(ref_idx)
            cfg_idx, cfg, stitched = stitched_results[ref_idx]
            nmfa_specs = [item for item in cfg['refine'] if item[0] == 'nmfa']
            if not nmfa_specs:
                continue
            for algo_name, batch_size, n_iter, seed in nmfa_specs[:1]:
                before_cut = best_cut
                stage_started = _profile_start()
                refined = _run_nmfa_candidates(
                    G_work,
                    batch_size=batch_size,
                    n_iter=n_iter,
                    seed=seed,
                    x=stitched,
                )
                reached, candidate_cut = consider_with_cut(f'selective-{algo_name}-refine-{cfg_idx}', refined)
                executed += 1
                _profile_emit(
                    'selective_refine',
                    cfg_idx=cfg_idx,
                    algo=algo_name,
                    elapsed_ms=_profile_ms(stage_started),
                    best_before=before_cut,
                    best_after=best_cut,
                    candidate_cut=candidate_cut,
                    improvement=best_cut - before_cut,
                    batch_size=batch_size,
                    n_iter=n_iter,
                    gap_abs=target_cut - best_cut if target_cut is not None else None,
                    reached=reached,
                )
                if reached:
                    return True
                if not selective_refine_allowed():
                    return False
                break
        return False

    aqmf_result = _try_aqmf_primary(G_work, graph_sum, target_cut)
    if aqmf_result is not None:
        aqmf_partition, aqmf_cut, aqmf_stage = aqmf_result
        if consider_scored(aqmf_partition, aqmf_cut, label=aqmf_stage):
            return finish(aqmf_stage)

    decompose_configs = [
        {
            'window_size': 140,
            'overlap': 23,
            'batch_size': 8,
            'n_iter': 98,
            'warmup_n_iter': 85,
            'alpha': 0.15,
            'sigma': 0.15,
            'x_scale': 0.05,
            'seed': 801,
            'refine': (
                ('bsb', 6, 80, 821),
                ('nmfa', 6, 100, 839),
            ),
            'structured_refine': (),
        },
        {
            'window_size': 208,
            'overlap': 52,
            'batch_size': 8,
            'n_iter': 100,
            'seed': 308,
            'refine': (
                ('bsb', 6, 80, 347),
                ('nmfa', 6, 100, 367),
            ),
            'structured_refine': (),
        },
        {
            'window_size': 176,
            'overlap': 44,
            'batch_size': 8,
            'n_iter': 100,
            'seed': 276,
            'refine': (
                ('bsb', 6, 80, 311),
                ('nmfa', 6, 100, 331),
            ),
            'structured_refine': (),
        },
        {
            'window_size': 192,
            'overlap': 48,
            'batch_size': 8,
            'n_iter': 100,
            'seed': 501,
            'refine': (
                ('bsb', 6, 80, 523),
                ('nmfa', 8, 120, 541),
            ),
            'structured_refine': (),
        },
        {
            'window_size': 224,
            'overlap': 56,
            'batch_size': 8,
            'n_iter': 100,
            'seed': 123,
            'refine': (
                ('bsb', 8, 100, 223),
                ('dsb', 6, 80, 239),
            ),
            'structured_refine': (
                (0.0040, 1.0, 'nmfa', 8, 140, 44),
            ),
        },
        {
            'window_size': 256,
            'overlap': 64,
            'batch_size': 8,
            'n_iter': 100,
            'seed': 321,
            'refine': (
                ('nmfa', 10, 120, 337),
                ('bsb', 6, 80, 353),
            ),
            'structured_refine': (),
        },
    ]

    active_decompose_configs = decompose_configs[:max(1, min(len(decompose_configs), max_iterations))]
    stitched_results = []
    window_executor = _get_window_executor()

    for cfg_idx, cfg in enumerate(active_decompose_configs):
        stage_label = f'decompose-{cfg_idx}'
        before_cut = best_cut
        stage_started = _profile_start()
        executed = False
        if cfg_idx == 0 and _matches_primary_window_nmfa_fast_cfg(cfg):
            fast_result = _solve_primary_window_nmfa_fast(
                G_work,
                cfg,
                graph_sum=graph_sum,
                target_cut=target_cut,
                local_flips=local_flips,
                executor=window_executor,
            )
            if fast_result is not None:
                executed = True
                profile = fast_result.get('profile', {})
                _profile_emit(
                    'fast_path',
                    cfg_idx=cfg_idx,
                    phase='warmup',
                    elapsed_ms=profile.get('warmup_ms'),
                    score_ms=profile.get('warm_score_ms'),
                    windows=profile.get('windows'),
                    batches=profile.get('batches'),
                    cut=fast_result['warm_cut'],
                    best_before=before_cut,
                    improved=fast_result['warm_cut'] > before_cut,
                    reached=target_cut is not None and fast_result['warm_cut'] >= target_cut,
                )
                if consider_scored(
                    fast_result['warm_partition'],
                    fast_result['warm_cut'],
                    label=f'{stage_label}-fast-warmup',
                ):
                    _profile_emit(
                        'decompose_config',
                        cfg_idx=cfg_idx,
                        executed=executed,
                        fast_path=True,
                        elapsed_ms=_profile_ms(stage_started),
                        best_before=before_cut,
                        best_after=best_cut,
                        improvement=best_cut - before_cut,
                    )
                    return finish(f'{stage_label}-fast-warmup')
                if fast_result['warm_reached_target']:
                    _profile_emit(
                        'decompose_config',
                        cfg_idx=cfg_idx,
                        executed=executed,
                        fast_path=True,
                        elapsed_ms=_profile_ms(stage_started),
                        best_before=before_cut,
                        best_after=best_cut,
                        improvement=best_cut - before_cut,
                    )
                    return finish(f'{stage_label}-fast-warmup')
                if fast_result.get('fast_exit'):
                    return finish(f'{stage_label}-fast-exit')

                stitched = fast_result['stitched']
                stitched_results.append((cfg_idx, cfg, stitched))
                _profile_emit(
                    'fast_path',
                    cfg_idx=cfg_idx,
                    phase='continuation',
                    elapsed_ms=profile.get('continuation_ms'),
                    score_ms=profile.get('final_score_ms'),
                    windows=profile.get('windows'),
                    batches=profile.get('batches'),
                    cut=fast_result['final_cut'],
                    best_before=best_cut,
                    improved=fast_result['final_cut'] > best_cut,
                    reached=target_cut is not None and fast_result['final_cut'] >= target_cut,
                )
                if consider_scored(
                    fast_result['final_partition'],
                    fast_result['final_cut'],
                    label=f'{stage_label}-fast-continuation',
                ):
                    _profile_emit(
                        'decompose_config',
                        cfg_idx=cfg_idx,
                        executed=executed,
                        fast_path=True,
                        elapsed_ms=_profile_ms(stage_started),
                        best_before=before_cut,
                        best_after=best_cut,
                        improvement=best_cut - before_cut,
                    )
                    return finish(f'{stage_label}-fast-continuation')
                _profile_emit(
                    'decompose_config',
                    cfg_idx=cfg_idx,
                    executed=executed,
                    fast_path=True,
                    elapsed_ms=_profile_ms(stage_started),
                    best_before=before_cut,
                    best_after=best_cut,
                    improvement=best_cut - before_cut,
                )
                continue

        executed = True
        stitched = _decompose_with_qaia(
            G_work,
            window_size=cfg['window_size'],
            overlap=cfg['overlap'],
            batch_size=cfg['batch_size'],
            n_iter=cfg['n_iter'],
            seed0=cfg['seed'],
            alpha=cfg.get('alpha', 0.15),
            sigma=cfg.get('sigma', 0.15),
            sigma_end=cfg.get('sigma_end'),
            x_scale=cfg.get('x_scale', 0.05),
            executor=window_executor,
        )
        stitched_results.append((cfg_idx, cfg, stitched))
        reached, candidate_cut = consider_with_cut(f'subgraph-nmfa-stitched-{cfg_idx}', stitched)
        _profile_emit(
            'decompose_config',
            cfg_idx=cfg_idx,
            executed=executed,
            fast_path=False,
            elapsed_ms=_profile_ms(stage_started),
            best_before=before_cut,
            best_after=best_cut,
            candidate_cut=candidate_cut,
            improvement=best_cut - before_cut,
            window_size=cfg['window_size'],
            overlap=cfg['overlap'],
        )
        if reached:
            return finish(f'{stage_label}-stitched')

    if run_selective_nmfa_refine():
        return finish('selective-refine')

    for cfg_idx, cfg, stitched in stitched_results:
        if cfg_idx != 4:
            continue
        for algo_name, batch_size, n_iter, seed in cfg['refine']:
            continue
            label = f'subgraph-{algo_name}-refine-{cfg_idx}'
            before_cut = best_cut
            stage_started = _profile_start()
            if algo_name == 'bsb':
                refined = _run_bsb_candidates(G_work, batch_size=batch_size, n_iter=n_iter, seed=seed, x=stitched)
            elif algo_name == 'dsb':
                refined = _run_dsb_candidates(G_work, batch_size=batch_size, n_iter=n_iter, seed=seed, x=stitched)
            else:
                refined = _run_nmfa_candidates(G_work, batch_size=batch_size, n_iter=n_iter, seed=seed, x=stitched)

            reached, candidate_cut = consider_with_cut(label, refined)
            _profile_emit(
                'refine',
                cfg_idx=cfg_idx,
                algo=algo_name,
                elapsed_ms=_profile_ms(stage_started),
                best_before=before_cut,
                best_after=best_cut,
                candidate_cut=candidate_cut,
                improvement=best_cut - before_cut,
                batch_size=batch_size,
                n_iter=n_iter,
                reached=reached,
            )
            if reached:
                return finish(label)

        for keep_ratio, structure_lambda, algo_name, batch_size, n_iter, seed in cfg['structured_refine']:
            label = f'structured-{algo_name}-refine-{cfg_idx}'
            before_cut = best_cut
            stage_started = _profile_start()
            J_search = _make_structured_search_graph(
                G_work,
                keep_ratio=keep_ratio,
                structure_lambda=structure_lambda,
                seed=seed,
            )
            if algo_name == 'bsb':
                refined = _run_bsb_candidates(J_search, batch_size=batch_size, n_iter=n_iter, seed=seed, x=stitched)
            elif algo_name == 'dsb':
                refined = _run_dsb_candidates(J_search, batch_size=batch_size, n_iter=n_iter, seed=seed, x=stitched)
            else:
                refined = _run_nmfa_candidates(J_search, batch_size=batch_size, n_iter=n_iter, seed=seed, x=stitched)

            reached, candidate_cut = consider_with_cut(label, refined)
            _profile_emit(
                'structured_refine',
                cfg_idx=cfg_idx,
                algo=algo_name,
                elapsed_ms=_profile_ms(stage_started),
                best_before=before_cut,
                best_after=best_cut,
                candidate_cut=candidate_cut,
                improvement=best_cut - before_cut,
                keep_ratio=keep_ratio,
                structure_lambda=structure_lambda,
                batch_size=batch_size,
                n_iter=n_iter,
                reached=reached,
            )
            if reached:
                return finish(label)

    full_graph_configs = [
    ]

    for label, algo_name, J, batch_size, n_iter, seed, dt, xi_scale in full_graph_configs:
        before_cut = best_cut
        stage_started = _profile_start()
        xi = _resolve_sb_xi(J, xi_scale)
        if algo_name == 'bsb':
            candidates = _run_bsb_candidates(J, batch_size=batch_size, n_iter=n_iter, seed=seed, dt=dt, xi=xi)
        else:
            candidates = _run_dsb_candidates(J, batch_size=batch_size, n_iter=n_iter, seed=seed, dt=dt, xi=xi)

        reached, candidate_cut = consider_with_cut(label, candidates)
        _profile_emit(
            'full_graph',
            label=label,
            algo=algo_name,
            elapsed_ms=_profile_ms(stage_started),
            best_before=before_cut,
            best_after=best_cut,
            candidate_cut=candidate_cut,
            improvement=best_cut - before_cut,
            batch_size=batch_size,
            n_iter=n_iter,
            reached=reached,
        )
        if reached:
            return finish(label)

    structure_configs = [
    ]

    for keep_ratio, structure_lambda, seed, bsb_dt, bsb_xi_scale, dsb_dt, dsb_xi_scale in structure_configs:
        stage_started = _profile_start()
        J_search = _make_structured_search_graph(
            G_work,
            keep_ratio=keep_ratio,
            structure_lambda=structure_lambda,
            seed=seed,
        )
        bsb_xi = _resolve_sb_xi(J_search, bsb_xi_scale)
        before_cut = best_cut
        bsb_started = _profile_start()
        structured_bsb = _run_bsb_candidates(
            J_search,
            batch_size=4,
            n_iter=70,
            seed=seed + 1,
            dt=bsb_dt,
            xi=bsb_xi,
        )
        reached, candidate_cut = consider_with_cut('structured-bsb', structured_bsb)
        _profile_emit(
            'structured_full_graph',
            label='structured-bsb',
            algo='bsb',
            elapsed_ms=_profile_ms(bsb_started),
            best_before=before_cut,
            best_after=best_cut,
            candidate_cut=candidate_cut,
            improvement=best_cut - before_cut,
            keep_ratio=keep_ratio,
            structure_lambda=structure_lambda,
            reached=reached,
        )
        if reached:
            _profile_emit(
                'structure_config',
                seed=seed,
                elapsed_ms=_profile_ms(stage_started),
                keep_ratio=keep_ratio,
                structure_lambda=structure_lambda,
            )
            return finish('structured-bsb')

        dsb_xi = _resolve_sb_xi(J_search, dsb_xi_scale)
        before_cut = best_cut
        dsb_started = _profile_start()
        structured_dsb = _run_dsb_candidates(
            J_search,
            batch_size=4,
            n_iter=70,
            seed=seed + 7,
            dt=dsb_dt,
            xi=dsb_xi,
        )
        reached, candidate_cut = consider_with_cut('structured-dsb', structured_dsb)
        _profile_emit(
            'structured_full_graph',
            label='structured-dsb',
            algo='dsb',
            elapsed_ms=_profile_ms(dsb_started),
            best_before=before_cut,
            best_after=best_cut,
            candidate_cut=candidate_cut,
            improvement=best_cut - before_cut,
            keep_ratio=keep_ratio,
            structure_lambda=structure_lambda,
            reached=reached,
        )
        _profile_emit(
            'structure_config',
            seed=seed,
            elapsed_ms=_profile_ms(stage_started),
            keep_ratio=keep_ratio,
            structure_lambda=structure_lambda,
        )
        if reached:
            return finish('structured-dsb')

    return finish('exhausted')


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


