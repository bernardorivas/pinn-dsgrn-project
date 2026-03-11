"""
DSGRN interface for parameter index computation and Morse graph comparison.

Wraps DSGRN/DSGRN_utils calls in try/except for graceful degradation
when the DSGRN package is not installed.
"""

import json
from pathlib import Path

import numpy as np

from network_parser import parse_net_spec

try:
    import DSGRN
    _HAS_DSGRN = True
except ImportError:
    _HAS_DSGRN = False

try:
    from DSGRN.ParameterFromSample import par_index_from_sample
    _HAS_PAR_INDEX = True
except ImportError:
    try:
        from DSGRN_utils.parameter_building import par_index_from_sample
        _HAS_PAR_INDEX = True
    except ImportError:
        _HAS_PAR_INDEX = False


def dsgrn_available():
    """Check if DSGRN and par_index_from_sample are available."""
    return _HAS_DSGRN and _HAS_PAR_INDEX


def _require_dsgrn():
    """Raise ImportError if DSGRN or DSGRN_utils are not available."""
    if not _HAS_DSGRN:
        raise ImportError(
            "DSGRN is required but not installed. "
            "Install with: pip install DSGRN"
        )
    if not _HAS_PAR_INDEX:
        raise ImportError(
            "par_index_from_sample not found in DSGRN.ParameterFromSample "
            "or DSGRN_utils.parameter_building. Update DSGRN or install DSGRN_utils."
        )


def compute_parameter_index(net_spec, L, U, T):
    """
    Compute the DSGRN parameter index for given L, U, T matrices.

    Args:
        net_spec: DSGRN network specification string
        L, U, T: 2D arrays of shape (n_nodes, n_nodes)

    Returns:
        int: DSGRN parameter index, or -1 if DSGRN is unavailable
    """
    if not dsgrn_available():
        print("Warning: DSGRN not available, skipping parameter index computation")
        return -1

    network = DSGRN.Network(net_spec)
    pg = DSGRN.ParameterGraph(network)

    L = np.asarray(L)
    U = np.asarray(U)
    T = np.asarray(T)

    par_index = par_index_from_sample(pg, L, U, T)
    return par_index


def compute_morse_graph(net_spec, par_index):
    """
    Compute the DSGRN Morse graph for a given parameter index.

    Args:
        net_spec: DSGRN network specification string
        par_index: integer parameter index

    Returns:
        (morse_graph, stg, graded_complex) or (None, None, None) if unavailable
    """
    if not _HAS_DSGRN or par_index < 0:
        return None, None, None

    network = DSGRN.Network(net_spec)
    pg = DSGRN.ParameterGraph(network)
    parameter = pg.parameter(par_index)
    domain_graph = DSGRN.DomainGraph(parameter)
    morse_decomposition = DSGRN.MorseDecomposition(domain_graph.digraph())
    morse_graph = DSGRN.MorseGraph(domain_graph, morse_decomposition)
    return morse_graph, domain_graph, morse_decomposition


def morse_graph_to_string(morse_graph):
    """Convert a DSGRN MorseGraph to a human-readable string."""
    if morse_graph is None:
        return "unavailable"
    try:
        return str(morse_graph)
    except Exception:
        return "error converting morse graph"


def parse_dsgrn_sample(sample_json, network):
    """
    Parse JSON from DSGRN.ParameterSampler.sample() into L, U, T arrays.

    Keys in the JSON are like "L[1->2]" with 1-based node name strings.
    Uses network.index(name) for 0-based conversion.

    Args:
        sample_json: JSON string from sampler.sample(parameter)
        network: DSGRN.Network object

    Returns:
        (L, U, T) as 2D numpy arrays of shape (n_nodes, n_nodes)
    """
    sample_dict = json.loads(sample_json)
    params = sample_dict["Parameter"]
    n = network.size()
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    T = np.zeros((n, n))

    for key, value in params.items():
        par_type = key[0]  # 'L', 'U', or 'T'
        # Extract node names from e.g. "L[1->2]"
        node_names = [name.strip() for name in key[2:-1].split('->')]
        src = network.index(node_names[0])
        tgt = network.index(node_names[1])
        if par_type == 'L':
            L[src, tgt] = value
        elif par_type == 'U':
            U[src, tgt] = value
        elif par_type == 'T':
            T[src, tgt] = value
        else:
            raise ValueError(f"Unexpected parameter type '{par_type}' in key '{key}'")

    return L, U, T


def sample_dsgrn_parameters(net_spec, par_index):
    """
    Draw one sample from DSGRN for a given par_index.

    Returns:
        (L, U, T) as 2D numpy arrays
    """
    _require_dsgrn()
    network = DSGRN.Network(net_spec)
    pg = DSGRN.ParameterGraph(network)
    parameter = pg.parameter(par_index)
    sampler = DSGRN.ParameterSampler(network)
    sample_json = sampler.sample(parameter)
    return parse_dsgrn_sample(sample_json, network)


def get_threshold_ordering(net_spec, par_index):
    """
    Extract threshold ordering per source node from parameter.inequalities().

    DSGRN orders thresholds on each source variable: for source s with
    outgoing edges to targets t1, t2, ..., the parameter region fixes
    an ordering T[s,t_i1] < T[s,t_i2] < ...

    Returns:
        dict[int, list[int]]: source_node -> list of target_nodes in ascending
        T[source, target] order
    """
    _require_dsgrn()
    network = DSGRN.Network(net_spec)
    pg = DSGRN.ParameterGraph(network)
    parameter = pg.parameter(par_index)

    ineq_json = json.loads(parameter.inequalities())
    ineq_str = ineq_json["inequalities"]

    import re
    ordering = {}
    for clause in ineq_str.split("&&"):
        clause = clause.strip()
        # Match pure T ordering chains: "0 < T[a->b] < T[c->d] < ..."
        parts = [p.strip() for p in clause.split("<")]
        t_parts = []
        is_t_chain = True
        for p in parts:
            if p == '0':
                continue
            m = re.match(r'^T\[(\d+)->(\d+)\]$', p)
            if m:
                t_parts.append((m.group(1), m.group(2)))
            else:
                is_t_chain = False
                break
        if is_t_chain and len(t_parts) >= 2:
            src_0 = network.index(t_parts[0][0])
            targets_0 = [network.index(t) for _, t in t_parts]
            ordering[src_0] = targets_0

    # Sources with only one outgoing edge have no ordering clause
    topology = parse_net_spec(net_spec)
    for edge in topology.edges:
        if edge.source not in ordering:
            ordering[edge.source] = [edge.target]

    return ordering


def generate_well_separated_T(net_spec, par_index, min_spacing=1.5,
                               min_T=1.0, max_T=6.0, seed=None,
                               global_min_spacing=0.0, max_attempts=10000):
    """
    Generate randomized well-separated T values respecting DSGRN ordering.

    For each source node with ordered thresholds T[s,t1] < T[s,t2] < ...,
    samples T values uniformly in [min_T, max_T] with at least min_spacing
    between consecutive thresholds.

    Args:
        net_spec: DSGRN network specification string
        par_index: DSGRN parameter index
        min_spacing: minimum gap between consecutive thresholds on same source
        min_T: minimum threshold value
        max_T: maximum threshold value
        seed: random seed
        global_min_spacing: minimum gap between ANY pair of T values across
            all edges. 0 disables cross-source checking.
        max_attempts: max rejection sampling attempts for global constraint

    Returns:
        T as 2D numpy array
    """
    ordering = get_threshold_ordering(net_spec, par_index)
    topology = parse_net_spec(net_spec)

    for attempt in range(max_attempts):
        rng = np.random.default_rng(seed + attempt if seed is not None else None)
        T = np.zeros((topology.n_nodes, topology.n_nodes))

        for source, targets in ordering.items():
            n = len(targets)
            range_needed = (n - 1) * min_spacing
            if max_T - min_T < range_needed:
                raise ValueError(
                    f"Cannot fit {n} thresholds for source {source} in "
                    f"[{min_T}, {max_T}] with spacing {min_spacing}. "
                    f"Need at least {range_needed + min_T} for max_T."
                )
            available = max_T - min_T - range_needed
            raw = np.sort(rng.uniform(0, available, size=n))
            for rank, tgt in enumerate(targets):
                T[source, tgt] = min_T + raw[rank] + rank * min_spacing

        if global_min_spacing <= 0:
            return T

        # Check all pairs of T values across edges
        all_vals = [T[s, t] for s, t in topology.edge_list]
        ok = True
        for i in range(len(all_vals)):
            for j in range(i + 1, len(all_vals)):
                if abs(all_vals[i] - all_vals[j]) < global_min_spacing:
                    ok = False
                    break
            if not ok:
                break

        if ok:
            return T

    raise RuntimeError(
        f"Could not generate well-separated T values in {max_attempts} attempts. "
        f"Try relaxing global_min_spacing={global_min_spacing} or widening "
        f"[min_T={min_T}, max_T={max_T}]."
    )


def sample_with_fixed_T(net_spec, par_index, T, U_margin=5.0,
                         max_iter=100_000, seed=None):
    """
    Rejection-sample L, U that land in the given par_index with fixed T.

    For each edge (s, t):
        L[s,t] ~ Uniform(0, T[s,t])
        U[s,t] ~ Uniform(T[s,t], T[s,t] + U_margin)

    Args:
        net_spec: DSGRN network specification string
        par_index: target DSGRN parameter index
        T: 2D array of thresholds
        U_margin: max offset above T for U sampling
        max_iter: max rejection sampling iterations
        seed: random seed

    Returns:
        (L, U) as 2D numpy arrays

    Raises:
        RuntimeError: if no valid sample found within max_iter
    """
    _require_dsgrn()
    rng = np.random.default_rng(seed)
    topology = parse_net_spec(net_spec)
    network = DSGRN.Network(net_spec)
    pg = DSGRN.ParameterGraph(network)
    n = topology.n_nodes
    edge_list = topology.edge_list

    for i in range(max_iter):
        L = np.zeros((n, n))
        U = np.zeros((n, n))
        for s, t in edge_list:
            L[s, t] = rng.uniform(1e-3, T[s, t])
            U[s, t] = rng.uniform(T[s, t], T[s, t] + U_margin)

        idx = par_index_from_sample(pg, L, U, T)
        if idx == par_index:
            return L, U

    raise RuntimeError(
        f"No valid (L, U) found in {max_iter} iterations for par_index={par_index}. "
        f"Try increasing max_iter, U_margin, or adjusting T values."
    )


def generate_parameters(net_spec, par_index, T=None, gamma=None,
                         min_spacing=1.5, min_T=1.0, max_T=6.0,
                         U_margin=5.0, max_iter=100_000, seed=None,
                         global_min_spacing=0.0):
    """
    Generate DSGRN-compatible parameters for a given par_index.

    If T is None, auto-generates randomized well-separated T values.
    If gamma is None, defaults to all ones.

    Args:
        net_spec: DSGRN network specification string
        par_index: target DSGRN parameter index
        T: 2D array of thresholds, or None for auto-generation
        gamma: 1D array of decay rates, or None for default
        min_spacing: minimum gap between thresholds (auto-T only)
        min_T: minimum threshold value (auto-T only)
        max_T: maximum threshold value (auto-T only)
        U_margin: max offset above T for U sampling
        max_iter: max rejection sampling iterations
        seed: random seed
        global_min_spacing: minimum gap between ANY pair of T values

    Returns:
        dict with keys: L, U, T, gamma (all numpy arrays)
    """
    topology = parse_net_spec(net_spec)

    if T is None:
        T = generate_well_separated_T(net_spec, par_index,
                                       min_spacing=min_spacing, min_T=min_T,
                                       max_T=max_T, seed=seed,
                                       global_min_spacing=global_min_spacing)
    T = np.asarray(T, dtype=float)

    if gamma is None:
        gamma = np.ones(topology.n_nodes)
    gamma = np.asarray(gamma, dtype=float)

    L, U = sample_with_fixed_T(net_spec, par_index, T,
                                U_margin=U_margin, max_iter=max_iter, seed=seed)

    return {'L': L, 'U': U, 'T': T, 'gamma': gamma}


def generate_dsgrn_figures(net_spec, par_index, output_dir):
    """
    Generate DSGRN Morse graph and Morse set figures for a par_index.

    Saves morse_graph.png (graphviz) and morse_sets.png (matplotlib).
    """
    _require_dsgrn()
    import DSGRN_utils
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    network = DSGRN.Network(net_spec)
    pg = DSGRN.ParameterGraph(network)
    parameter = pg.parameter(par_index)

    morse_graph, stg, graded_complex = DSGRN_utils.ConleyMorseGraph(parameter)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Morse graph (graphviz)
    mg_plot = DSGRN_utils.PlotMorseGraph(morse_graph)
    mg_plot.render(str(output_dir / "morse_graph"), format="png", cleanup=True)

    # Morse sets (matplotlib)
    DSGRN_utils.PlotMorseSets(
        morse_graph, stg, graded_complex,
        fig_fname=str(output_dir / "morse_sets.png"),
    )
    plt.close('all')
    print(f"  Saved dsgrn_figs: morse_graph.png, morse_sets.png")


def compare_dynamics(net_spec, L_true, U_true, T_true, L_hat, U_hat, T_hat):
    """
    Compare ground truth and recovered parameters via DSGRN.

    Returns:
        dict with keys: gt_index, rec_index, same_region,
                       gt_morse_str, rec_morse_str
    """
    gt_idx = compute_parameter_index(net_spec, L_true, U_true, T_true)
    rec_idx = compute_parameter_index(net_spec, L_hat, U_hat, T_hat)

    gt_mg, _, _ = compute_morse_graph(net_spec, gt_idx)
    rec_mg, _, _ = compute_morse_graph(net_spec, rec_idx)

    return {
        'gt_index': gt_idx,
        'rec_index': rec_idx,
        'same_region': gt_idx == rec_idx and gt_idx >= 0,
        'gt_morse_str': morse_graph_to_string(gt_mg),
        'rec_morse_str': morse_graph_to_string(rec_mg),
    }
