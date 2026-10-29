"""Two-arm bench: stock networkx (base) vs networkx_rust (port).

Tracks from WORKLOAD.md (frozen): rep/path/multi/target/x1.
Each (arm, track) runs in a fresh subprocess; wall = median, RSS = child
ru_maxrss medians both arms; checksums must agree across arms per track.
Writes bench/bench_evidence.json; prints the README table.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from statistics import median

HERE = os.path.dirname(os.path.abspath(__file__))
PORT_PY = os.path.join(os.path.dirname(HERE), "python")
VPY = sys.executable


def build_graphs():
    import networkx as nx
    import random
    graphs = {}

    def ring(n, seed, extra_random):
        rng = random.Random(seed)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            G.add_edge(i, (i + 1) % n, weight=1.0 + (i % 9))
            G.add_edge(i, (i * 37 + 11) % n, weight=1.0 + ((i * 13) % 17))
            G.add_edge(i, (i * 101 + 7) % n, weight=1.0 + ((i * 29) % 23))
            if extra_random and rng.random() < 0.3:
                G.add_edge(i, rng.randrange(n),
                           weight=1.0 + rng.random() * 9)
        return G

    graphs["R1"] = ring(1000, 20261028, True)
    graphs["R2"] = ring(2500, 20261029, True)
    graphs["R5"] = ring(5000, 20261030, False)
    rng = random.Random(20261031)
    GRID = nx.grid_2d_graph(50, 50)
    for u, v in GRID.edges():
        GRID.edges[u, v]["weight"] = 1.0 + rng.random() * 9
    graphs["GRID"] = GRID
    return graphs


def sources_for(G, k, seed):
    import random
    rng = random.Random(seed)
    nodes = list(G.nodes())
    return [nodes[rng.randrange(len(nodes))] for _ in range(k)]


def run_track(arm, track):
    if arm == "port":
        sys.path.insert(0, PORT_PY)
        import networkx_rust as nr
        assert nr.HAS_CORE, "extension not built"
    else:
        import networkx as nx_alias
    import networkx as nx
    graphs = build_graphs()

    def length(G, s, cutoff=None, target=None):
        if arm == "base":
            if target is not None:
                return {target: nx.dijkstra_path_length(G, s, target)}
            return nx.single_source_dijkstra_path_length(G, s, cutoff=cutoff)
        if target is not None:
            return {target: nr.dijkstra_path_length(G, s, target)}
        return nr.single_source_dijkstra_path_length(G, s, cutoff=cutoff)

    def path_pair(G, s, t):
        if arm == "base":
            return nx.single_source_dijkstra(G, s, target=t)
        return nr.single_source_dijkstra(G, s, target=t)

    def multi(G, srcs):
        if arm == "base":
            return nx.multi_source_dijkstra_path_length(G, set(srcs))
        return nr.multi_source_dijkstra_path_length(G, set(srcs))

    dist_sum = 0.0
    reached = 0
    plen_sum = 0
    t0 = time.perf_counter()
    if track == "rep":
        for i, (gid, G) in enumerate(graphs.items()):
            for s in sources_for(G, 6, 9000 + i):
                d = length(G, s)
                dist_sum += sum(d.values())
                reached += len(d)
    elif track == "path":
        for i, (gid, G) in enumerate(graphs.items()):
            srcs = sources_for(G, 8, 9100 + i)
            for s, t in zip(srcs[:4], srcs[4:]):
                L, p = path_pair(G, s, t)
                dist_sum += L if not isinstance(L, dict) else sum(L.values())
                plen_sum += len(p) if not isinstance(p, dict) else sum(map(len, p.values()))
                reached += 1
    elif track == "multi":
        for i, (gid, G) in enumerate(graphs.items()):
            d = multi(G, sources_for(G, 8, 9200 + i))
            dist_sum += sum(d.values())
            reached += len(d)
    elif track == "target":
        for i, (gid, G) in enumerate(graphs.items()):
            srcs = sources_for(G, 8, 9300 + i)
            for s, t in zip(srcs[:4], srcs[4:]):
                d = length(G, s, target=t)
                dist_sum += sum(d.values())
                reached += len(d)
    elif track == "x1":
        G = graphs["R1"]
        srcs = sources_for(G, 2, 9400)
        L, p = path_pair(G, srcs[0], srcs[1])
        dist_sum += L
        plen_sum += len(p)
        reached += 1
    wall = time.perf_counter() - t0
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(json.dumps({"arm": arm, "track": track, "wall": wall,
                      "rss_mib": rss,
                      "checksum": [round(dist_sum, 6), reached, plen_sum]}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="base")
    ap.add_argument("--track", default="rep")
    ap.add_argument("--driver", action="store_true")
    args = ap.parse_args()
    env = dict(os.environ)
    env["PYTHONPATH"] = PORT_PY + os.pathsep + env.get("PYTHONPATH", "")
    if not args.driver:
        run_track(args.arm, args.track)
        return
    TRACKS = ["rep", "path", "multi", "target", "x1"]
    REPS = {"rep": 5, "path": 5, "multi": 5, "target": 5, "x1": 9}
    rows = {}
    for track in TRACKS:
        for arm in ["base", "port"]:
            walls, rsss, cks = [], [], []
            for r in range(REPS[track]):
                out = subprocess.run(
                    [VPY, os.path.join(HERE, "bench.py"), "--arm", arm,
                     "--track", track], capture_output=True, text=True, env=env)
                if out.returncode != 0:
                    print("FAIL", arm, track, out.stderr[-3000:],
                          file=sys.stderr)
                    sys.exit(1)
                j = json.loads(out.stdout.strip().splitlines()[-1])
                walls.append(j["wall"])
                rsss.append(j["rss_mib"])
                cks.append(tuple(j["checksum"]))
            rows[(arm, track)] = (median(walls), median(rsss), cks[0],
                                  len(set(cks)) == 1)
            print(f"{arm:4s} {track:6s} wall={median(walls):.4f}s "
                  f"rss={median(rsss):.1f}MiB checksum={cks[0]} "
                  f"stable={len(set(cks)) == 1}", flush=True)
    ok = True
    for track in TRACKS:
        cks = {rows[(a, track)][2] for a in ["base", "port"]}
        print(f"AGREE {track}: {len(cks) == 1} {cks}")
        ok = ok and len(cks) == 1
    ev = {"tracks": {}}
    print("\n| track | base (ms) | port (ms) | speedup | RSS base → port |")
    print("|---|---|---|---|---|")
    for track in TRACKS:
        b, br, _ = rows[("base", track)][:3]
        p, pr, _ = rows[("port", track)][:3]
        brm, prm = rows[("base", track)][1], rows[("port", track)][1]
        print(f"| {track} | {b*1e3:.1f} | {p*1e3:.1f} | {b/p:.2f}x "
              f"| {brm:.1f} → {prm:.1f} MiB |")
        ev["tracks"][track] = {"base_wall_s": b, "port_wall_s": p,
                               "speedup": b / p, "base_rss_mib": brm,
                               "port_rss_mib": prm,
                               "checksum": list(rows[("base", track)][2])}
    ev["agreement"] = ok
    with open(os.path.join(HERE, "bench_evidence.json"), "w") as f:
        json.dump(ev, f, indent=2)
    if not ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
