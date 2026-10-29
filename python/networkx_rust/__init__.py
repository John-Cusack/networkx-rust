"""networkx_rust: Rust-accelerated Dijkstra for networkx.

Drop-in replacements for the networkx Dijkstra shortest-path family.
Stock numeric edge weights on (non-multi) graphs take the Rust kernel
(whole-graph crossing, one call); custom weight callables, multigraphs,
and anything the kernel refuses take the pure-Python upstream path by
construction. Without the compiled extension every function delegates,
so the package imports and runs anywhere networkx does.
"""

try:
    from networkx_rust import _rust as _core
except ImportError:  # pure-Python fallback: no compiled extension
    _core = None

HAS_CORE = _core is not None

__version__ = "0.1.0"

from networkx_rust.weighted import (  # noqa: E402
    all_pairs_dijkstra,
    all_pairs_dijkstra_path,
    all_pairs_dijkstra_path_length,
    dijkstra_path,
    dijkstra_path_length,
    dijkstra_predecessor_and_distance,
    multi_source_dijkstra,
    multi_source_dijkstra_path,
    multi_source_dijkstra_path_length,
    single_source_dijkstra,
    single_source_dijkstra_path,
    single_source_dijkstra_path_length,
)

__all__ = [
    "HAS_CORE",
    "all_pairs_dijkstra",
    "all_pairs_dijkstra_path",
    "all_pairs_dijkstra_path_length",
    "dijkstra_path",
    "dijkstra_path_length",
    "dijkstra_predecessor_and_distance",
    "multi_source_dijkstra",
    "multi_source_dijkstra_path",
    "multi_source_dijkstra_path_length",
    "single_source_dijkstra",
    "single_source_dijkstra_path",
    "single_source_dijkstra_path_length",
]
