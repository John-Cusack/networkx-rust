# networkx-rust

Rust-accelerated Dijkstra shortest paths for
[networkx](https://github.com/networkx/networkx) (see NOTICE: no upstream
code is copied here; everything else in networkx stays verbatim upstream).

## Scope (deliberately narrow)

Only the Dijkstra shortest-path family is accelerated:

- `single_source_dijkstra{,_path,_path_length}`
- `multi_source_dijkstra{,_path,_path_length}`
- `dijkstra_path`, `dijkstra_path_length`
- `dijkstra_predecessor_and_distance`
- `all_pairs_dijkstra{,_path,_path_length}`

Everything else — graph classes, generators, Bellman–Ford, Johnson,
A\*, bidirectional Dijkstra, centrality, flow — is untouched upstream
networkx and is never reimplemented here.

## Routing rule

| condition | path |
|---|---|
| non-multigraph + non-callable `weight` | Rust kernel (whole graph crosses once per call) |
| custom weight callable, multigraph, or kernel refusal (non-numeric weight) | stock networkx, by construction |
| package imported without the compiled extension | pure-Python fallback: every function delegates to stock networkx |

A\* keeps the pure-Python path: the heuristic is a per-node user
callback, and per-item callbacks across the FFI boundary are the
documented losing shape (per-item work < ~1 µs is boundary-dominated).

## Equivalence

Distances (float-exact), dict iteration order, paths node-for-node,
predecessor lists including equal-cost multi-predecessors, tie-breaking
order, cutoffs, targets, and exception type/message (`ValueError`
contradictory-weights tripwire, `NodeNotFound`, `NetworkXNoPath`) are
identical to stock networkx. Gates:

- `tests/test_weighted_dispatch.py` — equivalence corpus + fuzz (30
  randomized graphs × full function surface)
- upstream `networkx/algorithms/shortest_paths/tests/` run verbatim with
  every Dijkstra entrypoint routed through this package: **654 passed**
- full upstream suite (8412 passed, 63 skipped, 741 xfailed) with the
  same routing as backstop — no regressions outside the kernel
| track | base (ms) | port (ms) | speedup | RSS base → port |
|---|---|---|---|---|
| rep | 220.5 | 61.3 | 3.60x | 47.1 → 49.5 MiB |
| path | 97.6 | 43.6 | 2.24x | 47.8 → 49.0 MiB |
| multi | 35.9 | 13.0 | 2.77x | 47.1 → 48.7 MiB |
| target | 77.4 | 37.1 | 2.09x | 47.2 → 48.4 MiB |
| x1 | 2.9 | 1.3 | 2.29x | 46.6 → 47.4 MiB |

(Filled by `bench/bench.py --driver`; numbers land in
`bench/bench_evidence.json`. Full lane record — workload, tier-1
scripts, equiv corpus, verification — lives in rustsmith
`data/rs-port-networkx/` alongside ADR-061. Tier-1 pure-Python attempts
for context:
per-call adjacency export 0.4–0.9x (measured loser — export costs more
than the `dict.get` + weight-lambda it removes); cross-query cached
export 1.4–1.7x (needs a caching-handle API, not drop-in).)

## Install

```
pip install networkx-rust
python -c "import networkx_rust as nr; print(nr.HAS_CORE)"
```

Replace calls selectively (`nr.single_source_dijkstra_path_length(G, s)`)
— same signatures, same results. No global monkeypatching: `import
networkx` behavior never changes.

## Build

```
pip install -e .   # builds crates/nx-dijkstra-core via setuptools-rust
```
