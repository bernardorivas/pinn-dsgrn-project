"""
Network topology parser for DSGRN net_spec strings.

Parses DSGRN network specifications into structured topology representations
with logic trees describing multilinear polynomial combinations of sigma functions.
"""

from dataclasses import dataclass
from typing import Union
import numpy as np


@dataclass
class InputEdge:
    """A directed edge in the regulatory network."""
    source: int  # 0-based node index
    target: int  # 0-based node index
    sign: str    # '+' (activation) or '-' (repression)

    def __repr__(self):
        return f"Edge({self.source}->{self.target}, {self.sign})"


@dataclass
class LogicNode:
    """A node in the logic tree representing sum or product of inputs."""
    op: str   # 'sum' or 'product'
    children: list  # list of LogicNode | InputEdge

    def __repr__(self):
        return f"LogicNode({self.op}, {self.children})"


LogicTree = Union[LogicNode, InputEdge]


class NetworkTopology:
    """Parsed representation of a DSGRN regulatory network."""

    def __init__(self, n_nodes, node_logic, edges, net_spec):
        self.n_nodes = n_nodes
        self.node_logic = node_logic  # dict[int, LogicTree]
        self.edges = edges            # list[InputEdge]
        self.net_spec = net_spec

    @property
    def n_edges(self):
        return len(self.edges)

    @property
    def edge_list(self):
        """Ordered list of (source, target) tuples."""
        return [(e.source, e.target) for e in self.edges]

    def edge_sign(self, source, target):
        """Get the sign of the edge from source to target."""
        for e in self.edges:
            if e.source == source and e.target == target:
                return e.sign
        raise ValueError(f"Edge ({source}, {target}) not found")

    def edge_index(self, source, target):
        """Get the flat index of an edge in the edge list."""
        for i, (s, t) in enumerate(self.edge_list):
            if s == source and t == target:
                return i
        raise ValueError(f"Edge ({source}, {target}) not found")

    def trapping_box(self, U, gamma):
        """
        Compute trapping box upper bounds B_i per node.

        Both sigma_plus and sigma_minus have maximum value U[src, tgt],
        so the max of the logic tree is obtained by replacing each sigma
        with U and evaluating the sums/products.

        Args:
            U: array of shape (n_nodes, n_nodes), U[source, target]
            gamma: array of shape (n_nodes,)

        Returns:
            1D array of upper bounds B_i per node
        """
        U = np.asarray(U)
        gamma = np.asarray(gamma)
        bounds = np.zeros(self.n_nodes)
        for i in range(self.n_nodes):
            max_val = self._evaluate_max(self.node_logic[i], U)
            bounds[i] = max_val / gamma[i]
        return bounds

    def _evaluate_max(self, node, U):
        """Recursively compute max output value of a logic tree node."""
        if isinstance(node, InputEdge):
            return U[node.source, node.target]
        elif isinstance(node, LogicNode):
            child_vals = [self._evaluate_max(c, U) for c in node.children]
            if node.op == 'sum':
                return sum(child_vals)
            elif node.op == 'product':
                result = 1.0
                for v in child_vals:
                    result *= v
                return result
        raise ValueError(f"Unknown node type: {type(node)}")

    def __repr__(self):
        return (f"NetworkTopology(n_nodes={self.n_nodes}, n_edges={self.n_edges}, "
                f"net_spec='{self.net_spec}')")


def topology_slug(topology: NetworkTopology) -> str:
    """
    Generate a filesystem-safe network ID from topology structure.

    Per node: {target}-{logic_type}-{inputs} where inputs list source indices,
    prefixed with 'n' for repression. Nodes separated by '__'.

    Example for "1 : 1+2\n2 : (~1)2":
        Node 0: sum of (+0, +1)  -> "0-sum-0-1"
        Node 1: prod of (-0, +1) -> "1-prod-n0-1"
        Full slug: "0-sum-0-1__1-prod-n0-1"
    """
    parts = []
    for target in range(topology.n_nodes):
        tree = topology.node_logic[target]
        inputs = _collect_inputs(tree)
        logic_type = _top_level_op(tree).replace('product', 'prod')
        input_strs = []
        for source, sign in inputs:
            prefix = "n" if sign == '-' else ""
            input_strs.append(f"{prefix}{source}")
        parts.append(f"{target}-{logic_type}-{'-'.join(input_strs)}")
    return "__".join(parts)


def _collect_inputs(node) -> list[tuple[int, str]]:
    """Collect (source, sign) pairs from a logic tree in left-to-right order."""
    if isinstance(node, InputEdge):
        return [(node.source, node.sign)]
    elif isinstance(node, LogicNode):
        result = []
        for child in node.children:
            result.extend(_collect_inputs(child))
        return result
    raise ValueError(f"Unknown node type: {type(node)}")


def _top_level_op(node) -> str:
    """Get the top-level operation of a logic tree node."""
    if isinstance(node, InputEdge):
        return "single"
    elif isinstance(node, LogicNode):
        return node.op
    raise ValueError(f"Unknown node type: {type(node)}")


def parse_net_spec(net_spec: str) -> NetworkTopology:
    """
    Parse a DSGRN net_spec string into a NetworkTopology.

    Grammar for each node's expression (after the colon):
        expr   = term ('+' term)*        # sum
        term   = factor factor*           # product (juxtaposition)
        factor = '(' expr ')' | '~' NUM | NUM

    Node IDs in the spec are 1-based; converted to 0-based internally.

    Args:
        net_spec: e.g. "1 : 1+2\\n2 : (~1)2"

    Returns:
        NetworkTopology
    """
    lines = [line.strip() for line in net_spec.strip().split('\n') if line.strip()]

    node_specs = {}
    for line in lines:
        parts = line.split(':')
        node_id = int(parts[0].strip())
        expr_str = parts[1].strip()
        node_specs[node_id] = expr_str

    n_nodes = len(node_specs)
    node_logic = {}
    all_edges = []

    for node_id_1based in sorted(node_specs):
        target = node_id_1based - 1
        expr_str = node_specs[node_id_1based]
        tokens = _tokenize(expr_str)
        tree, edges, _ = _parse_expr(tokens, 0, target)
        node_logic[target] = tree
        all_edges.extend(edges)

    # Validate no duplicate (source, target) pairs
    seen = set()
    for e in all_edges:
        key = (e.source, e.target)
        if key in seen:
            raise ValueError(
                f"Duplicate edge ({e.source+1} -> {e.target+1}) in net_spec. "
                "Each (source, target) pair must appear exactly once."
            )
        seen.add(key)

    return NetworkTopology(
        n_nodes=n_nodes,
        node_logic=node_logic,
        edges=all_edges,
        net_spec=net_spec
    )


def _tokenize(expr_str):
    """Tokenize an expression string into a list of (type, value) tuples."""
    tokens = []
    s = expr_str.replace(' ', '')
    i = 0
    while i < len(s):
        if s[i] == '(':
            tokens.append(('LPAREN', '('))
            i += 1
        elif s[i] == ')':
            tokens.append(('RPAREN', ')'))
            i += 1
        elif s[i] == '+':
            tokens.append(('PLUS', '+'))
            i += 1
        elif s[i] == '~':
            tokens.append(('TILDE', '~'))
            i += 1
        elif s[i].isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            tokens.append(('NUMBER', int(s[i:j])))
            i = j
        else:
            raise ValueError(f"Unexpected character '{s[i]}' in '{expr_str}'")
    return tokens


def _parse_expr(tokens, pos, target):
    """Parse sum: term ('+' term)*. Returns (tree, edges, new_pos)."""
    tree, edges, pos = _parse_term(tokens, pos, target)
    terms = [tree]
    all_edges = list(edges)

    while pos < len(tokens) and tokens[pos][0] == 'PLUS':
        pos += 1
        tree, edges, pos = _parse_term(tokens, pos, target)
        terms.append(tree)
        all_edges.extend(edges)

    if len(terms) == 1:
        return terms[0], all_edges, pos
    return LogicNode('sum', terms), all_edges, pos


def _parse_term(tokens, pos, target):
    """Parse product: factor+. Returns (tree, edges, new_pos)."""
    tree, edges, pos = _parse_factor(tokens, pos, target)
    factors = [tree]
    all_edges = list(edges)

    while pos < len(tokens) and tokens[pos][0] in ('LPAREN', 'TILDE', 'NUMBER'):
        tree, edges, pos = _parse_factor(tokens, pos, target)
        factors.append(tree)
        all_edges.extend(edges)

    if len(factors) == 1:
        return factors[0], all_edges, pos
    return LogicNode('product', factors), all_edges, pos


def _parse_factor(tokens, pos, target):
    """Parse factor: '(' expr ')' | '~' NUMBER | NUMBER."""
    if tokens[pos][0] == 'LPAREN':
        pos += 1
        tree, edges, pos = _parse_expr(tokens, pos, target)
        if pos >= len(tokens) or tokens[pos][0] != 'RPAREN':
            raise ValueError("Unmatched parenthesis in expression")
        pos += 1
        return tree, edges, pos
    elif tokens[pos][0] == 'TILDE':
        pos += 1
        if pos >= len(tokens) or tokens[pos][0] != 'NUMBER':
            raise ValueError("Expected number after '~'")
        source = tokens[pos][1] - 1
        pos += 1
        edge = InputEdge(source, target, '-')
        return edge, [edge], pos
    elif tokens[pos][0] == 'NUMBER':
        source = tokens[pos][1] - 1
        pos += 1
        edge = InputEdge(source, target, '+')
        return edge, [edge], pos
    else:
        raise ValueError(f"Unexpected token: {tokens[pos]}")
