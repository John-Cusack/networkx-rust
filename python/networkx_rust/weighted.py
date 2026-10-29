"""Dispatch layer: fast Rust kernel on the stock-weight path, stock
networkx everywhere else.

Each wrapper mirrors its upstream preamble (validation, short-circuits)
line for line, then takes the kernel when the graph is not a multigraph
and `weight` is not callable. The kernel refuses non-numeric weights
(TypeError); on refusal the wrapper re-runs stock networkx, which then
produces whatever result or error upstream defines.
"""

from itertools import islice

import networkx as nx

from networkx_rust import _core

__all__ = [
    "single_source_dijkstra",
    "single_source_dijkstra_path",
    "single_source_dijkstra_path_length",
    "multi_source_dijkstra",
    "multi_source_dijkstra_path",
    "multi_source_dijkstra_path_length",
    "dijkstra_path",
    "dijkstra_path_length",
    "dijkstra_predecessor_and_distance",
    "all_pairs_dijkstra",
    "all_pairs_dijkstra_path",
    "all_pairs_dijkstra_path_length",
]

# Stock entrypoints, captured at import so the slow path stays stock even
# if someone shadows networkx attributes with these wrappers (as the
# upstream-suite gate does).
_stock = {name: getattr(nx, name) for name in __all__}


def _fast_path(G, weight):
    return _core is not None and not G.is_multigraph() and not callable(weight)


def _kernel_length(G, sources, weight, cutoff, target):
    """(dist,) via kernel; TypeError -> None (caller falls back)."""
    try:
        dist, _ = _core.dijkstra(
            G._adj, sources, weight, cutoff, target, 0
        )
    except TypeError:
        return None
    return dist


def _kernel_pred(G, sources, weight, cutoff, target, want):
    """(dist, pred) via kernel; TypeError -> (None, None)."""
    try:
        return _core.dijkstra(G._adj, sources, weight, cutoff, target, want)
    except TypeError:
        return None, None


def single_source_dijkstra_path_length(G, source, cutoff=None, weight="weight"):
    return multi_source_dijkstra_path_length(
        G, {source}, cutoff=cutoff, weight=weight
    )


def single_source_dijkstra_path(G, source, cutoff=None, weight="weight"):
    return multi_source_dijkstra_path(G, {source}, cutoff=cutoff, weight=weight)


def single_source_dijkstra(G, source, target=None, cutoff=None, weight="weight"):
    return multi_source_dijkstra(
        G, {source}, cutoff=cutoff, target=target, weight=weight
    )


def multi_source_dijkstra_path_length(G, sources, cutoff=None, weight="weight"):
    if not _fast_path(G, weight):
        return _stock["multi_source_dijkstra_path_length"](
            G, sources, cutoff=cutoff, weight=weight
        )
    if not sources:
        raise ValueError("sources must not be empty")
    for s in sources:
        if s not in G:
            raise nx.NodeNotFound(f"Node {s} not found in graph")
    dist = _kernel_length(G, list(sources), weight, cutoff, None)
    if dist is None:
        return _stock["multi_source_dijkstra_path_length"](
            G, sources, cutoff=cutoff, weight=weight
        )
    return dist


def multi_source_dijkstra(G, sources, target=None, cutoff=None, weight="weight"):
    if not _fast_path(G, weight):
        return _stock["multi_source_dijkstra"](
            G, sources, target=target, cutoff=cutoff, weight=weight
        )
    if not sources:
        raise ValueError("sources must not be empty")
    for s in sources:
        if s not in G:
            raise nx.NodeNotFound(f"Node {s} not found in graph")
    if target in sources:
        return (0, [target])
    srcs = list(sources)
    if target is None:
        got = _kernel_pred(G, srcs, weight, cutoff, None, 1)
        if got[0] is None:
            return _stock["multi_source_dijkstra"](
                G, sources, target=target, cutoff=cutoff, weight=weight
            )
        dist, pred = got
        paths = {source: [source] for source in srcs}
        # upstream skips len(seen) entries (unique sources settle first)
        for v in islice(dist, len(set(srcs)), None):
            paths[v] = paths[pred[v][0]] + [v]
        return (dist, paths)
    got = _kernel_pred(G, srcs, weight, cutoff, target, 1)
    if got[0] is None:
        return _stock["multi_source_dijkstra"](
            G, sources, target=target, cutoff=cutoff, weight=weight
        )
    dist, pred = got
    try:
        d = dist[target]
        p = pred[target]
    except KeyError as err:
        raise nx.NetworkXNoPath(f"No path to {target}.") from err
    path = [target]
    while (cp := pred.get(path[-1])) is not None and cp:
        path.append(cp[0])
    # upstream rebuilds from target backwards through first preds;
    # sources map to [source] via the paths dict, but the target branch
    # walks pred directly (pred has no source entries with preds).
    path.reverse()
    return (d, path)


def multi_source_dijkstra_path(G, sources, cutoff=None, weight="weight"):
    length, path = multi_source_dijkstra(G, sources, cutoff=cutoff, weight=weight)
    return path


def dijkstra_path(G, source, target, weight="weight"):
    (length, path) = single_source_dijkstra(
        G, source, target=target, weight=weight
    )
    return path


def dijkstra_path_length(G, source, target, weight="weight"):
    if not _fast_path(G, weight):
        return _stock["dijkstra_path_length"](G, source, target, weight=weight)
    if source not in G:
        raise nx.NodeNotFound(f"Node {source} not found in graph")
    if source == target:
        return 0
    dist = _kernel_length(G, [source], weight, None, target)
    if dist is None:
        return _stock["dijkstra_path_length"](G, source, target, weight=weight)
    try:
        return dist[target]
    except KeyError as err:
        raise nx.NetworkXNoPath(
            f"Node {target} not reachable from {source}"
        ) from err


def dijkstra_predecessor_and_distance(G, source, cutoff=None, weight="weight"):
    if not _fast_path(G, weight):
        return _stock["dijkstra_predecessor_and_distance"](
            G, source, cutoff=cutoff, weight=weight
        )
    if source not in G:
        raise nx.NodeNotFound(f"Node {source} is not found in the graph")
    got = _kernel_pred(G, [source], weight, cutoff, None, 2)
    if got[0] is None:
        return _stock["dijkstra_predecessor_and_distance"](
            G, source, cutoff=cutoff, weight=weight
        )
    dist, pred = got
    if source not in pred:
        pred[source] = []
    return (pred, dist)


def all_pairs_dijkstra(G, cutoff=None, weight="weight"):
    for n in G:
        yield (
            n,
            single_source_dijkstra(G, n, cutoff=cutoff, weight=weight),
        )


def all_pairs_dijkstra_path_length(G, cutoff=None, weight="weight"):
    for n in G:
        yield (
            n,
            single_source_dijkstra_path_length(G, n, cutoff=cutoff, weight=weight),
        )


def all_pairs_dijkstra_path(G, cutoff=None, weight="weight"):
    for n in G:
        yield (
            n,
            single_source_dijkstra_path(G, n, cutoff=cutoff, weight=weight),
        )
