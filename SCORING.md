# Scoring & correctness — ground truth from `neurogolf_utils.py`

Read directly from `Dataset/neurogolf_utils/neurogolf_utils.py` (the official scorer). This
supersedes the reconstructed assumptions in `PLAN.md`/`STRATEGY.md`. `solver/scorer.py` wraps
this module so we measure exactly what the leaderboard measures.

## Points
```
points = max(1.0, 25.0 - ln(memory + params))     # per task, summed over 400
zero-cost network (memory==0 and params==0)  ->  25.0
```
`score_network()` returns `(memory, params)` and **does not check correctness**. The leaderboard
awards points only if the network is *also* correct, so an incorrect placeholder that costs 0
scores 25 *in isolation* but **0 on the board**. Our runner counts unsolved tasks as 0.

## `params` — element COUNT (dtype-independent)
`calculate_params` sums `prod(dims)` over initializers + sparse initializers + `Constant` node
tensors (`value`/`value_floats`/`value_ints`/...). Scalars cost 1.
- ⇒ int64→int32 or fp32→fp16 narrowing of **initializers does NOT reduce params**. (It only
  helps the 1.44 MB filesize cap.) The `narrow_int32` surgery pass is a no-op for score — kept
  only for filesize headroom.
- ⇒ Minimize params by using **fewer / smaller weight tensors**: Gather index (10) < reversed
  index (30) < 1×1 Conv (100) < dense constant grid (up to 9000).

## `memory` — BYTES of intermediate tensors (the dominant term)
`calculate_memory` sums `num_elements * dtype.itemsize` over every tensor that is a **node
output or value_info**, taking the **max shape seen across runs** (static shape inference +
ORT profiler trace). **The graph `input` and `output` tensors are excluded.**
- ⇒ Each *intermediate* `[1,10,30,30]` float32 tensor ≈ **36,000 bytes** (9000 elems × 4).
  One intermediate alone dwarfs any reasonable param count.
- ⇒ **Node count is the primary score driver.** A transform that writes straight to `output`
  in one node has `memory = 0`. Every extra node adds its output tensor's bytes.
- ⇒ fp16 reduces the **memory** term (intermediate activations) only — never params.

### Measured (via `solver/scorer.py`, confirming the model)
| builder | nodes | memory | params | points |
|---|---|---|---|---|
| identity | 1 (passthrough) | 0 | 0 | **25.000** |
| rot180 (single `Slice`, steps=-1) | 1 | 0 | 8 | **22.921** |
| color_permute (`Gather` axis=1) | 1 | 0 | 10 | 22.697 |
| flip_h/flip_v (single `Slice`) | 1 | 0 | ~4 | ~23.6 |
| rot90 (`Transpose`+`Gather`) | 2 | 36000 | 30 | **14.508** |

The rot90 example is the lesson: its one unavoidable intermediate (the transpose output) costs
36 KB and halves the score vs a single-node transform. **Golf = collapse to the fewest nodes
that write directly to `output`** (e.g. flips/rot180 became single `Slice` ops with negative
steps; the old 2-`Gather` form carried a 36 KB intermediate).

## Correctness — threshold, NOT argmax
`run_network` returns `(output > 0.0)` as 0/1. `verify_subset` requires that binary tensor to
**exactly equal the one-hot expected grid** (`np.array_equal`), where:
- a real cell of color `c` ⇒ channel `c` must be `>0`, all other channels `≤0`;
- a padding cell ⇒ **all 10 channels `≤0`** ("no color"); the harness trims trailing no-color
  cells/rows to recover the output H×W.

⇒ **argmax-based checking is wrong** — argmax treats an empty cell as color 0 (black), so it
would pass networks that the official scorer rejects (extra black padding → wrong size). Fixed:
`solver/solvers.runs_correct` now thresholds at >0 and compares to the one-hot, matching the
scorer. This also means networks need only the right **sign pattern**, not exact one-hot floats.

## Hard constraints (the rules)
- Exactly one input named `input`, one output named `output`; shapes `[1,10,30,30]`.
- **All tensor dims must be static and positive** (no dynamic/symbolic dims) — enforced by
  strict shape inference. So variable output size is expressed by zeroing padding, not by a
  dynamic shape.
- Opset domain must be `""`/`ai.onnx`; no functions, no custom domains, **no subgraphs**
  (`If`/`Loop` attributes rejected).
- **Banned op types:** `Loop, Scan, NonZero, Unique, Script, Function, Compress`, plus any
  `Sequence*` op. ⇒ flood-fill / BFS / connected-components must be **unrolled** (fixed #steps);
  no `NonZero` dynamic gather.
- **Filesize ≤ 1.44 MB** per ONNX file.
- No name collision between I/O tensors and initializers; duplicate value_info names rejected.
- Harness reference graphs use **opset 10 / IR 10**. Our builders currently emit opset 11 / IR 8
  and pass `check_model` + ORT; if any task trips shape inference, drop those builders to
  opset 10 (note: `Pad` takes `pads` as an attribute at opset ≤10 vs an input at 11+).

## Implications for strategy
1. Coverage still first (0 → solved is the big jump), but **golf is worth more than "fractions
   of a point"** for any multi-node solution — collapsing nodes to remove a 36 KB intermediate
   is ~+8 points.
2. Drop dtype-narrowing as a scoring lever (params are element counts); reserve fp16 for the
   memory term on genuinely unavoidable intermediates.
3. Prefer single-node `Slice`/`Gather`/`Transpose` straight to `output`; treat every extra
   intermediate full-grid tensor as ~−2 to −8 points and design it out where possible.
