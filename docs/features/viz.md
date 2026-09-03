# Visualisation & Demos

The `examples/` directory ships ready-to-run scripts for seeing assignments with your own eyes — in the terminal, or as images.

## Terminal heatmap (`terminal_ui.py`)

No third-party dependencies. Renders the cost matrix as an ANSI **block heatmap** (blue = cheap → red = expensive) with the optimal assignment cells highlighted in bold and bracketed, then prints the row→column assignment list:

```bash
uv run python examples/terminal_ui.py heatmap
uv run python examples/terminal_ui.py heatmap --algo auction --size 10 --seed 7
uv run python examples/terminal_ui.py heatmap --maximize
```

```
         0       1       2       3
r0     51.7    95.1   [ 15.3]   94.9
r1     31.9    42.9    82.9   [ 41.5]
...
```

## Algorithm comparison (`terminal_ui.py compare`)

Races every algorithm on the same random matrix and prints a timing + cost table, marking the (shared) optimal cost. Uses the `rich` library for a polished table when it's installed, and falls back to plain columns otherwise:

```bash
uv run python examples/terminal_ui.py compare --size 30
```

## Bipartite graph (`bipartite_assignment.py`)

Draws the assignment as an actual **bipartite graph**: rows and columns as two node columns, every admissible edge shaded by cost, and the matching edges overlaid in bold red with cost labels. Requires `matplotlib` and `networkx`:

```bash
uv run python examples/bipartite_assignment.py            # saves assignment_graph.png
uv run python examples/bipartite_assignment.py --size 8 --seed 3 --no-show
```

## Matplotlib heatmap (`visualize_assignment.py`)

The original example: a `YlOrRd` cost heatmap with the optimal cells ringed and their costs labelled. Requires `matplotlib`:

```bash
uv run python examples/visualize_assignment.py            # saves assignment.png
```

!!! note "A quick visual check"
    The terminal heatmap and the bipartite graph make great debugging companions: after a solve, eyeballing which cells got chosen (and at what cost) often exposes a subtly-wrong cost matrix far faster than reading the returned arrays.
