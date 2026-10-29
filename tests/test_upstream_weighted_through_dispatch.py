"""Gate: run upstream networkx shortest-path weighted tests with every
Dijkstra-family entrypoint routed through networkx_rust dispatch.

The test functions are star-imported so they run verbatim; only the
`nx` module attributes they call are shadowed (monkeypatched at module
import, restored never — this module owns its pytest process).
"""

import networkx as nx

import networkx_rust as nr

_DISPATCHED = [
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

for _name in _DISPATCHED:
    setattr(nx, _name, getattr(nr, _name))

from networkx.algorithms.shortest_paths.tests.test_weighted import *  # noqa
