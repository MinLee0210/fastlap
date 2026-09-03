"""
Terminal UI demo for fastlap — no third-party dependencies required.

Renders the cost matrix as an ANSI block heatmap with the optimal assignment
marked (bright * markers), then prints the assignment and total cost. A
``compare`` mode runs every algorithm and prints a timing/cost table, using the
`rich` library if it happens to be installed and falling back to plain text.

Run:
    python examples/terminal_ui.py                 # 6x6 random demo, lapjv
    python examples/terminal_ui.py --algo auction  # pick an algorithm
    python examples/terminal_ui.py --size 10 --seed 7
    python examples/terminal_ui.py compare --size 30
"""

import argparse
import time

import numpy as np

import fastlap

# Truecolor escape helpers -------------------------------------------------

def _rgb(r, g, b, bg=False):
    return f"\x1b[48;2;{r};{g};{b}m" if bg else f"\x1b[38;2;{r};{g};{b}m"


BOLD = "\x1b[1m"
RESET = "\x1b[0m"

# Blue (cheap) -> white -> red (expensive).
def _heat_color(t):
    """t in [0, 1]; returns an (r, g, b) triple on a blue-white-red ramp."""
    if t < 0.5:
        s = t * 2.0
        return (int(60 + 130 * s), int(60 + 120 * s), 255)
    s = (t - 0.5) * 2.0
    return (255, int(255 - 120 * s), int(255 - 195 * s))


def heatmap(matrix, chosen, maximize=False):
    """Render `matrix` as a terminal heatmap; `chosen` maps row -> col or None.

    Cells on the optimal assignment are shown in bold with the value bracketed
    like ``[12.3]`` so they jump out even in black-and-white terminals."""
    lo = float(np.min(matrix))
    hi = float(np.max(matrix))
    span = (hi - lo) or 1.0
    nrows, ncols = matrix.shape
    lines = ["   " + " ".join(f"{c:>7}" for c in range(ncols))]
    for i in range(nrows):
        row = [f"r{i:<2}"]
        for j in range(ncols):
            v = matrix[i, j]
            t = 1.0 - ((v - lo) / span) if maximize else (v - lo) / span
            r, g, b = _heat_color(max(0.0, min(1.0, t)))
            is_chosen = chosen is not None and chosen[i] == j
            cell = f"{v:7.1f}" if not is_chosen else f"[{v:5.1f}]"
            style = _rgb(r, g, b, bg=True) + (BOLD if is_chosen else "")
            row.append(style + cell + RESET)
        lines.append(" ".join(row))
    return "\n".join(lines)


def assignment_table(matrix, row_assign, col_assign):
    """Simple two-column listing of who matched whom."""
    rows = []
    for i, j in enumerate(row_assign):
        if j is None:
            rows.append(f"  row {i:<3} -> (unassigned)")
        else:
            rows.append(f"  row {i:<3} -> col {j:<3}  cost {matrix[i, j]:8.3f}")
    unassigned_cols = [j for j, v in enumerate(col_assign) if v is None]
    if unassigned_cols:
        rows.append(f"  cols {unassigned_cols} unmatched")
    return "\n".join(rows)


def render_plain(matrix, algo, maximize):
    cost, rows, cols = fastlap.solve_lap(matrix, algo, maximize=maximize)
    mode = "maximize" if maximize else "minimize"
    print(f"\n{matrix.shape[0]}x{matrix.shape[1]} cost matrix — optimal "
          f"assignment ({algo}, {mode}):\n")
    print(heatmap(matrix, rows, maximize=maximize))
    print(f"\nTotal cost: {BOLD}{cost:.6f}{RESET}\n")
    print(assignment_table(matrix, rows, cols))
    return cost


def render_rich(matrix, algo, maximize):
    from rich import box
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    console = Console()
    cost, rows, cols = fastlap.solve_lap(matrix, algo, maximize=maximize)
    table = Table(title=f"Optimal assignment — {algo}", box=box.MINIMAL)
    table.add_column("row", justify="right")
    table.add_column("assigned to col", justify="right")
    table.add_column("cost", justify="right")
    for i, j in enumerate(rows):
        if j is None:
            table.add_row(str(i), "—", "—")
        else:
            table.add_row(str(i), str(j), f"{matrix[i, j]:.3f}")
    console.print(table)
    console.print(f"Total cost: [bold]{cost:.6f}[/bold]")
    return cost


def cmd_heatmap(args):
    rng = np.random.default_rng(args.seed)
    m = rng.uniform(1, 100, (args.size, args.size))
    cost = render_plain(m, args.algo, args.maximize)
    if args.rich and not args.maximize:
        print("\n[rich rendering]")
        render_rich(m, args.algo, args.maximize)
    return cost


def cmd_compare(args):
    rng = np.random.default_rng(args.seed)
    m = rng.uniform(1, 100, (args.size, args.size))
    algos = [a for a in fastlap.get_supported_algorithms()]
    mode = "maximize" if args.maximize else "minimize"
    print(f"Comparing {len(algos)} algorithms on {args.size}x{args.size} "
          f"({mode})…")
    results = {}
    for a in algos:
        t0 = time.perf_counter()
        c, *_ = fastlap.solve_lap(m, a, maximize=args.maximize)
        results[a] = (time.perf_counter() - t0, c)
    best = min(results.values(), key=lambda x: x[1]) if not args.maximize \
        else max(results.values(), key=lambda x: x[1])

    rows = [(a, t * 1e3, c) for a, (t, c) in results.items()]
    rows.sort(key=lambda r: r[1])
    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()
        table = Table(title="fastlap algorithm comparison")
        for col in ("algorithm", "time (ms)", "cost"):
            table.add_column(col, justify="right" if col != "algorithm" else "left")
        for a, t, c in rows:
            mark = "*" if abs(c - best[1]) < 1e-9 else ""
            table.add_row(f"{a}{mark}", f"{t:.3f}", f"{c:.6f}")
        console.print(table)
        console.print("\n* = optimal cost")
    except ImportError:
        print(f"\n{'algorithm':<14}{'time (ms)':>12}{'cost':>16}")
        for a, t, c in rows:
            mark = "*" if abs(c - best[1]) < 1e-9 else ""
            print(f"{a + mark:<15}{t:>12.3f}{c:>16.6f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="fastlap terminal demos")
    sub = p.add_subparsers(dest="cmd")

    heat = sub.add_parser("heatmap", help="heatmap + assignment in the terminal")
    heat.add_argument("--algo", default="lapjv")
    heat.add_argument("--size", type=int, default=6)
    heat.add_argument("--seed", type=int, default=42)
    heat.add_argument("--maximize", action="store_true")
    heat.add_argument("--rich", action="store_true", help="also print a rich table")

    cmp = sub.add_parser("compare", help="benchmark every algorithm in the terminal")
    cmp.add_argument("--size", type=int, default=30)
    cmp.add_argument("--seed", type=int, default=42)
    cmp.add_argument("--maximize", action="store_true")

    args = p.parse_args()
    if args.cmd == "compare":
        cmd_compare(args)
    else:
        cmd_heatmap(args)
