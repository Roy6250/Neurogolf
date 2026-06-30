# opset-10 ONNX cookbook for ARC net-golf

The network for a task is a fixed DAG of ops mapping `input[1,10,30,30] -> output[1,10,30,30]`
(channel = color). Output shape is **static**; correctness is `(output > 0)` compared exactly to
the expected one-hot. These are the composable primitives; a task solution is a composition.
(Distilled from souldrive's "Compile, Don't Train" notebook + our `neurogolf_utils` reading.)

## 1. Geometry = `Gather`
A geometric map is a fixed permutation of positions. `Gather(X, idx, axis=a)` does
`Y[…,i,…]=X[…,idx[i],…]`. Horizontal flip: `idx=[W-1..0]`, axis=3. **Separable** perms
(`h'=ρ(h)`, `w'=σ(w)`) are two 1-D gathers (`P = P_row ⊗ P_col`). Rotations mix axes, so they
need `Transpose` (axis swap) + gather — the extra node is an unavoidable intermediate.

## 2. Variable output via MatMul (no `NonZero`/`CumSum`)
Keep a data-dependent subset of rows, stacked top-left, with static shape. `keep ∈ {0,1}^H`.
- **Prefix sum = strict-lower-triangular MatMul:** `L[i,j]=1 iff j<i`; `dest = L·keep`, so
  `dest[i] = Σ_{j<i} keep[j]` = destination row index of row i.
- **Compaction = permutation-matrix MatMul:** `R[a,i] = 1[a=dest[i]]·keep[i]`, built as
  `R = (|a − destᵀ| < 0.5) ⊙ keepᵀ`; then `OUT = R·G` ([H×H]·[H×W]=[H×W]). Kept rows land
  top-left (dest is injective & consecutive on kept rows), rest zero. Shape stays `H×W` —
  variable count lives in the zeros. Swap `keep` for compress / dedupe / any row predicate.

## 3. Fixed-K peel (object enumeration, no `Loop`)
Budget K, unrolled: `mₖ=ReduceMax(Rₖ)`; `objₖ = 1[Rₖ>mₖ−0.5]·1[mₖ>0.5]`;
`Rₖ₊₁ = Rₖ⊙(1−objₖ)`. Greedily extracts the argmax level-set each round; empty-safe.

## 4. Connected components = max-propagation
`seed(h,w)=h·W+w+1`; `L₀ = M⊙seed` (M=foreground). Repeat `T=H+W` times:
`Lₜ₊₁ = (max over plus-neighbourhood of Lₜ) ⊙ M`. Iterating masked dilation to a fixed point;
each component fills with its max seed id (unique, stable). T = worst-case 4-connected diameter.

## 5. Canvas + keepmask
Static `[1,10,30,30]`, content top-left; zero the rest with `keepmask(h,w)=1[h<H]·1[w<W]`
(outer product of coordinate comparisons). `OUT = Raw ⊙ keepmask`. Variable size lives in the
values, never the shape. `crop` = two static `Gather`s + keepmask, never a dynamic `Slice`.

## 6. bbox memory golf
Memory ≈ Σ bytes of intermediate tensors. Run the trace on a `B×B` content bbox, `Pad` back:
`mem ∝ B²`, so `Δpoints = ln((30/B)²) = 2·ln(30/B)` (B=10 ⇒ ~2.2 pts per heavy net).

## 7. Higher-order object filter
`result = Σₖ [p(objₖ)]·objₖ` (disjoint objects ⇒ sum = union). Predicate `p` = size
(`ReduceSum`), symmetry (`1[Σ|o−flip(o)|<0.5]`), touches-border, unique-color, … "select objects
that…" = choosing `p`.

## 8. Lossless `size_optimize`
`onnxoptimizer` + `onnxsim`: prune/dedupe initializers, constant-fold, eliminate redundant nodes.
Behaviour-identical — re-verify after. fp16/int8 is **not** lossless (breaks the `0.5` thresholds
& integer ids) and doesn't help params anyway (params = element count). Keep float32.

## 9. Strict-equality footguns
(1) Color 0 ≡ channel-0 = 1: a content black cell needs channel-0 refilled, else it decodes as
"no color". (2) Fills that overshoot paint the zero-padding → clip to content extent (`⊙ keepmask`).

## 10. Partial-program verify
Reference solver `= fₙ∘…∘f₁`. For each k, build a target from intermediate `xₖ` and compare your
graph truncated to k statements; first divergence = the exact mis-compiled statement.

---

### What we've built so far (`onnx_ops.py` / `solvers.py`)
Single-/few-op programs: identity, transpose, flips, rot90/180/270, color-permute (`Gather`),
conv1x1 color-map, upscale, tile, symmetry-completion (`Where`+mirror), most/least-frequent-color
(histogram). These cover the size-agnostic / size-constant whole-grid + simple-object tasks
(~14/400). The primitives #2 and #4 above are the unbuilt levers for the object-reasoning tail.
