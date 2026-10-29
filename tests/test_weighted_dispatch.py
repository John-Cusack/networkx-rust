"""Equivalence: networkx_rust dispatch vs stock networkx.

Exact equality required: distances (float bits), dict iteration order,
paths node-for-node, predecessor lists, and exception type/message.
"""

import random

import networkx as nx
import pytest

import networkx_rust as nr
from networkx_rust import weighted as nrw


def ring(n, seed, extra_random=True, weight="weight"):
    rng = random.Random(seed)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        G.add_edge(i, (i + 1) % n, **{weight: 1.0 + (i % 9)})
        G.add_edge(i, (i * 37 + 11) % n, **{weight: 1.0 + ((i * 13) % 17)})
        if extra_random and rng.random() < 0.3:
            G.add_edge(i, rng.randrange(n),
                       **{weight: 1.0 + rng.random() * 9})
    return G


GRAPHS = {
    "R1": ring(200, 20261028),
    "R2": ring(500, 20261029, extra_random=False),
    "GRID": nx.grid_2d_graph(10, 10),
}
for _g in GRAPHS.values():
    for _u, _v in _g.edges():
        _g.edges[_u, _v].setdefault("weight", 2.5)


def pairs(G, k=4, seed=1):
    rng = random.Random(seed)
    nodes = list(G.nodes())
    return [(nodes[rng.randrange(len(nodes))],
             nodes[rng.randrange(len(nodes))]) for _ in range(k)]


def assert_same_dist(a, b):
    assert list(a.items()) == list(b.items())


@pytest.mark.parametrize("gid", sorted(GRAPHS))
def test_lengths_exact(gid):
    G = GRAPHS[gid]
    srcs = [list(G.nodes())[i * 37 % G.number_of_nodes()] for i in range(3)]
    for s in srcs:
        assert_same_dist(
            nx.single_source_dijkstra_path_length(G, s), nr.single_source_dijkstra_path_length(G, s))
        assert_same_dist(
            nx.multi_source_dijkstra_path_length(G, {s}), nr.multi_source_dijkstra_path_length(G, {s}))
    assert_same_dist(
        nx.multi_source_dijkstra_path_length(G, set(srcs)),
        nr.multi_source_dijkstra_path_length(G, set(srcs)))


@pytest.mark.parametrize("gid", sorted(GRAPHS))
def test_paths_exact(gid):
    G = GRAPHS[gid]
    for s, t in pairs(G):
        assert (nx.single_source_dijkstra(G, s, target=t)
                == nr.single_source_dijkstra(G, s, target=t))
        assert (nx.single_source_dijkstra(G, s) == nr.single_source_dijkstra(G, s))
        assert nx.dijkstra_path(G, s, t) == nr.dijkstra_path(G, s, t)
        assert (nx.dijkstra_path_length(G, s, t)
                == nr.dijkstra_path_length(G, s, t))
    assert (nx.multi_source_dijkstra(G, set(list(G.nodes())[:3]))
            == nr.multi_source_dijkstra(G, set(list(G.nodes())[:3])))
    assert (nx.multi_source_dijkstra_path(G, set(list(G.nodes())[:3]))
            == nr.multi_source_dijkstra_path(G, set(list(G.nodes())[:3])))


@pytest.mark.parametrize("gid", sorted(GRAPHS))
def test_predecessor_exact(gid):
    G = GRAPHS[gid]
    for s in list(G.nodes())[:3]:
        assert (nx.dijkstra_predecessor_and_distance(G, s)
                == nr.dijkstra_predecessor_and_distance(G, s))


@pytest.mark.parametrize("gid", sorted(GRAPHS))
def test_all_pairs_exact(gid):
    G = GRAPHS[gid]
    assert (dict(nx.all_pairs_dijkstra_path_length(G))
            == dict(nr.all_pairs_dijkstra_path_length(G)))
    assert dict(nx.all_pairs_dijkstra_path(G)) == dict(nr.all_pairs_dijkstra_path(G))
    assert dict(nx.all_pairs_dijkstra(G)) == dict(nr.all_pairs_dijkstra(G))


def test_cutoff_target_variants():
    G = GRAPHS["R1"]
    s, t = list(G.nodes())[0], list(G.nodes())[100]
    for cutoff in (None, 0, 5.5, 1000.0):
        assert_same_dist(
            nx.single_source_dijkstra_path_length(G, s, cutoff=cutoff),
            nr.single_source_dijkstra_path_length(G, s, cutoff=cutoff))
        assert (nx.single_source_dijkstra(G, s, cutoff=cutoff)
                == nr.single_source_dijkstra(G, s, cutoff=cutoff))
    assert (nx.single_source_dijkstra(G, s, target=t)
            == nr.single_source_dijkstra(G, s, target=t))


def test_source_eq_target():
    G = GRAPHS["R1"]
    assert nr.single_source_dijkstra(G, 3, target=3) == (0, [3])
    assert nr.dijkstra_path_length(G, 3, 3) == 0
    assert nr.multi_source_dijkstra(G, {1, 3}, target=3) == (0, [3])


def test_errors_identical():
    G = GRAPHS["R1"]
    H = nx.Graph()
    H.add_edge(0, 1, weight=1.0)
    H.add_edge(1, 2, weight=-5.0)  # negative weight tripwire
    for mod in (nx, nr):
        with pytest.raises(ValueError, match="sources must not be empty"):
            mod.multi_source_dijkstra_path_length(G, set())
        with pytest.raises(nx.NodeNotFound):
            mod.multi_source_dijkstra_path_length(G, {9999})
    try:
        nx.single_source_dijkstra_path_length(H, 0)
        nx_err = None
    except Exception as e:
        nx_err = (type(e).__name__, e.args)
    try:
        nr.single_source_dijkstra_path_length(H, 0)
        nr_err = None
    except Exception as e:
        nr_err = (type(e).__name__, e.args)
    assert nx_err == nr_err and nx_err is not None
    # unreachable target
    G2 = nx.Graph()
    G2.add_edge("a", "b", weight=1.0)
    G2.add_node("z")
    with pytest.raises(nx.NetworkXNoPath):
        nr.dijkstra_path(G2, "a", "z")
    with pytest.raises(nx.NetworkXNoPath):
        nx.dijkstra_path(G2, "a", "z")
    with pytest.raises(nx.NetworkXNoPath):
        nr.multi_source_dijkstra(G2, {"a"}, target="z")
    with pytest.raises(nx.NetworkXNoPath):
        nx.multi_source_dijkstra(G2, {"a"}, target="z")


def test_weight_variants():
    G = nx.Graph()
    G.add_edge("a", "b", w=3.0)
    G.add_edge("b", "c")  # missing attr -> 1
    G.add_edge("a", "c", w=10.0)
    G.add_edge("c", "d", w=None)  # hidden edge
    for weight in ("w", None, 7):
        assert_same_dist(nx.single_source_dijkstra_path_length(G, "a", weight=weight),
                         nr.single_source_dijkstra_path_length(G, "a", weight=weight))
    # callable weight -> slow path, still identical
    f = lambda u, v, d: d.get("w", 1)
    assert_same_dist(nx.single_source_dijkstra_path_length(G, "a", weight=f),
                     nr.single_source_dijkstra_path_length(G, "a", weight=f))
    assert (nx.single_source_dijkstra(G, "a", weight=f)
            == nr.single_source_dijkstra(G, "a", weight=f))
    # non-numeric weight value -> kernel refuses -> stock error identical
    G3 = nx.Graph()
    G3.add_edge(0, 1, weight="heavy")
    try:
        nx.single_source_dijkstra_path_length(G3, 0)
        nx_err = None
    except Exception as e:
        nx_err = (type(e).__name__, str(e))
    try:
        nr.single_source_dijkstra_path_length(G3, 0)
        nr_err = None
    except Exception as e:
        nr_err = (type(e).__name__, str(e))
    assert nx_err == nr_err and nx_err is not None


def test_multigraph_fallback_identical():
    MG = nx.MultiGraph()
    MG.add_edge(0, 1, weight=5.0)
    MG.add_edge(0, 1, weight=1.0)
    MG.add_edge(1, 2, weight=2.0)
    assert_same_dist(nx.single_source_dijkstra_path_length(MG, 0),
                     nr.single_source_dijkstra_path_length(MG, 0))
    assert (nx.single_source_dijkstra(MG, 0) == nr.single_source_dijkstra(MG, 0))
    assert (nx.dijkstra_predecessor_and_distance(MG, 0)
            == nr.dijkstra_predecessor_and_distance(MG, 0))


def test_digraph_and_labels():
    DG = nx.DiGraph()
    DG.add_edge("x", "y", weight=1.5)
    DG.add_edge("y", "z", weight=2.5)
    DG.add_edge("x", "z", weight=9.0)
    assert_same_dist(nx.single_source_dijkstra_path_length(DG, "x"),
                     nr.single_source_dijkstra_path_length(DG, "x"))
    assert nx.dijkstra_path(DG, "x", "z") == nr.dijkstra_path(DG, "x", "z")
    # directedness respected
    with pytest.raises(nx.NetworkXNoPath):
        nr.dijkstra_path(DG, "z", "x")


def test_zero_weights_tie_breaking():
    G = nx.Graph()
    G.add_edge(0, 1, weight=0.0)
    G.add_edge(0, 2, weight=0.0)
    G.add_edge(1, 3, weight=1.0)
    G.add_edge(2, 3, weight=1.0)
    assert nx.single_source_dijkstra(G, 0) == nr.single_source_dijkstra(G, 0)
    assert (nx.single_source_dijkstra_path_length(G, 0)
            == nr.single_source_dijkstra_path_length(G, 0))
    assert (nx.dijkstra_predecessor_and_distance(G, 0)
            == nr.dijkstra_predecessor_and_distance(G, 0))


def test_duplicate_and_generator_sources():
    G = GRAPHS["R1"]
    assert_same_dist(nx.multi_source_dijkstra_path_length(G, [5, 5, 9]),
                     nr.multi_source_dijkstra_path_length(G, [5, 5, 9]))
    assert (nx.multi_source_dijkstra(G, [5, 5, 9])
            == nr.multi_source_dijkstra(G, [5, 5, 9]))
    assert_same_dist(
        nx.multi_source_dijkstra_path_length(G, (n for n in [5, 9])),
        nr.multi_source_dijkstra_path_length(G, (n for n in [5, 9])))


def test_no_core_fallback(monkeypatch):
    monkeypatch.setattr(nrw, "_core", None)
    G = GRAPHS["R1"]
    assert_same_dist(nx.single_source_dijkstra_path_length(G, 0),
                     nr.single_source_dijkstra_path_length(G, 0))
    assert nx.dijkstra_path(G, 0, 50) == nr.dijkstra_path(G, 0, 50)


def test_fuzz_random_graphs():
    for trial in range(30):
        rng = random.Random(10_000 + trial)
        n = rng.randint(5, 60)
        G = nx.gnp_random_graph(n, 0.15, seed=rng)
        for u, v in G.edges():
            r = rng.random()
            if r < 0.05:
                G.edges[u, v]["weight"] = None  # hidden
            elif r < 0.10:
                pass  # missing -> 1
            else:
                G.edges[u, v]["weight"] = round(rng.uniform(0, 20), 3)
        if G.number_of_nodes() == 0:
            continue
        s = rng.choice(list(G.nodes()))
        assert_same_dist(nx.single_source_dijkstra_path_length(G, s),
                         nr.single_source_dijkstra_path_length(G, s))
        assert (nx.single_source_dijkstra(G, s) == nr.single_source_dijkstra(G, s))
        assert (nx.dijkstra_predecessor_and_distance(G, s)
                == nr.dijkstra_predecessor_and_distance(G, s))
        t = rng.choice(list(G.nodes()))
        try:
            nx_p = nx.single_source_dijkstra(G, s, target=t)
            nx_e = None
        except Exception as e:
            nx_p, nx_e = None, (type(e).__name__, str(e))
        try:
            nr_p = nr.single_source_dijkstra(G, s, target=t)
            nr_e = None
        except Exception as e:
            nr_p, nr_e = None, (type(e).__name__, str(e))
        assert (nx_p, nx_e) == (nr_p, nr_e)
