"""
Visualization example: plot a cost matrix as a heatmap and overlay the optimal assignment.

Requirements:
    pip install matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import fastlap


def visualize_assignment(matrix, algorithm="lapjv"):
    """Plot cost matrix heatmap with optimal assignment overlay."""
    cost, row_assign, col_assign = fastlap.solve_lap(matrix, algorithm=algorithm)
    n = len(row_assign)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Cost")

    for i in range(n):
        j = row_assign[i]
        if j != fastlap.solve_lap(matrix, algorithm=algorithm)[2][i]:
            continue
        ax.plot(j, i, "ko", markersize=12, markerfacecolor="none", markeredgewidth=2)
        ax.plot(j, i, "gx", markersize=8, markeredgewidth=2)
        ax.annotate(
            f"{matrix[i][j]:.1f}",
            (j, i),
            textcoords="offset points",
            xytext=(0, -15),
            ha="center",
            fontsize=8,
            color="green",
            fontweight="bold",
        )

    ax.set_xlabel("Column (Item)")
    ax.set_ylabel("Row (Agent)")
    ax.set_title(f"Optimal Assignment ({algorithm}) — Total Cost: {cost:.1f}")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    plt.tight_layout()
    plt.savefig("assignment.png", dpi=150)
    print(f"Saved assignment.png (cost={cost:.1f})")
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    matrix = np.random.uniform(0, 10, (6, 6))
    visualize_assignment(matrix, algorithm="lapjv")
