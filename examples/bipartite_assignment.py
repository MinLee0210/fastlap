"""
Bipartite-graph visualization of a linear assignment.

Draws the two partitions (workers on the left, jobs on the right) as node
columns and every admissible (row, col) pair as an edge shaded by cost; the
optimal assignment edges are highlighted in bold red with the cost labelled.

Run:
    python examples/bipartite_assignment.py                # saves assignment_graph.png
    python examples/bipartite_assignment.py --size 8 --seed 3
    python examples/bipartite_assignment.py --no-show      # headless
"""

import argparse

import numpy as np
import fastlap


def draw(matrix, algorithm, save_path, show):
    cost, row_assign, col_assign = fastlap.solve_lap(matrix, algorithm)
    nrows, ncols = matrix.shape

    import matplotlib
    if show is None:
        import os
        import sys
        show = bool(os.environ.get("DISPLAY") or sys.platform == "darwin")
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    g = nx.Graph()
    row_nodes = [f"r{i}" for i in range(nrows)]
    col_nodes = [f"c{j}" for j in range(ncols)]
    for n in row_nodes + col_nodes:
        g.add_node(n)
    for i in range(nrows):
        for j in range(ncols):
            g.add_edge(f"r{i}", f"c{j}", weight=float(matrix[i, j]))

    pos = {}
    pos.update((n, (0.0, nrows - 1 - i)) for i, n in enumerate(row_nodes))
    pos.update((n, (1.0, ncols - 1 - j)) for j, n in enumerate(col_nodes))

    chosen = {(f"r{i}", f"c{j}") for i, j in enumerate(row_assign) if j is not None}
    lo = float(np.min(matrix))
    hi = float(np.max(matrix))
    span = (hi - lo) or 1.0

    fig, ax = plt.subplots(figsize=(max(ncols * 0.6 + 3, 7), max(nrows * 0.55, 4)))

    # Non-matching edges, shaded by normalized cost (blues: cheap -> expensive).
    for (u, v, d) in g.edges(data=True):
        if (u, v) in chosen or (v, u) in chosen:
            continue
        t = (d["weight"] - lo) / span
        nx.draw_networkx_edges(g, pos, edgelist=[(u, v)], ax=ax, width=1.2,
                               edge_color=(t, 0.15, 0.45))

    # Matching edges on top.
    edge_labels = {}
    for (u, v) in chosen:
        w = g[u][v]["weight"]
        nx.draw_networkx_edges(g, pos, edgelist=[(u, v)], ax=ax, width=3.0,
                               edge_color="red")
        edge_labels[(u, v)] = f"{w:.1f}"

    nx.draw_networkx_labels(g, pos, ax=ax, font_size=10)
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, ax=ax,
                                 font_color="red", font_size=8,
                                 bbox=dict(facecolor="white", edgecolor="none",
                                           alpha=0.7))

    ax.set_title(f"Optimal assignment ({algorithm}) — total cost {cost:.3f}")
    ax.set_xlim(-0.25, 1.35)
    ax.set_ylim(-0.8, max(nrows, ncols) - 0.2)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved {save_path} (total cost = {cost:.3f})")
    if show:
        plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Bipartite assignment visualization")
    p.add_argument("--algo", default="lapjv")
    p.add_argument("--size", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", default="assignment_graph.png")
    p.add_argument("--show", action="store_true", default=None,
                   help="open a window (default: auto-detect display)")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    m = rng.uniform(1, 100, (args.size, args.size))
    draw(m, args.algo, args.save, args.show)
