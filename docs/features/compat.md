# Compatibility Layers

Already have code built against SciPy or `lap`/`lapx`? fastlap ships drop-in shims so you can swap in the Rust-backed solver with a one-line import change — same call signature, same return shape and dtypes.

## SciPy drop-in

`scipy.optimize.linear_sum_assignment` returns `(row_ind, col_ind)` as `int64` NumPy arrays. `fastlap.compat.linear_sum_assignment` matches this exactly:

```python
from fastlap.compat import linear_sum_assignment

# Returns (row_ind, col_ind) int64 ndarrays exactly like SciPy
row_ind, col_ind = linear_sum_assignment(cost_matrix)
```

```diff
- from scipy.optimize import linear_sum_assignment
+ from fastlap.compat import linear_sum_assignment
```

It's also available at the top level as `fastlap.linear_sum_assignment`. Internally it always solves with `"lapjv"`.

## `lap.lapjv` / `lapx.lapjv` drop-in {: #lapjv-drop-in }

ByteTrack, BoT-SORT, and similar YOLO-based MOT pipelines commonly call `lap.lapjv(cost, extend_cost=True, cost_limit=...)`, expecting `(opt_cost, x, y)` back with `int32` arrays and `-1` for unassigned entries. `fastlap.lap.lapjv` matches that contract:

```python
import fastlap.lap as lap

opt_cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=0.5)
```

```diff
- import lap
+ import fastlap.lap as lap
```

It's also available at the top level as `fastlap.lapjv`.

### Parameters

```python
fastlap.lapjv(
    cost,
    extend_cost=True,   # accepted for compatibility; handled automatically
    cost_limit=None,    # unassigned pairs return -1, not None
    return_cost=True,   # set False to get just (x, y)
)
```

- `x[i]` is the column assigned to row `i`, or `-1` if unassigned.
- `y[j]` is the row assigned to column `j`, or `-1` if unassigned.
- `extend_cost` is accepted for signature compatibility — fastlap pads rectangular matrices automatically regardless of its value.

```python
matrix = np.array([[0.1, 0.9], [0.9, 0.8]])
# With cost_limit=0.5, row 1 (cost 0.8 > 0.5) is unassigned (-1)
cost, x, y = fastlap.lapjv(matrix, cost_limit=0.5)
print(x)  # [0, -1]
```

## Why bother?

Both shims exist purely so you never have to translate return formats by hand. Everything else fastlap offers — the other nine algorithms, `cost_limit` gating, batch solving, K-best, LBAP — is still reachable through `solve_lap` and friends; the compat layer is an on-ramp, not a ceiling.
