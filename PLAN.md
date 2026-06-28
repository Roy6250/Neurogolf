# NeuroGolf 2026 — End-to-End Plan to Build Our Own Baseline

**Status:** draft v1 · **Date:** 2026-06-28
**Goal:** Build, from scratch, the 400-network ONNX submission (`task001.onnx … task400.onnx`)
that the `hoangvux/neurogolf` notebook *assumes already exists*. Without it we have nothing to
submit or optimize — this document is the recipe for that missing artifact.

---

## 0. The one strategic insight that drives everything

This is **not** blind ARC prediction. It is **program synthesis with the answers in hand.**

- The 400 tasks come from the **ARC-AGI public training set (v1)** — we can see every
  input→output pair.
- Correctness is validated against the **original ARC benchmarks + ARC-GEN-100K
  (procedurally generated variants) + a small private suite**. So each network must implement
  the **general transformation rule**, not memorize the few public examples. ARC-GEN gives us
  *many* I/O pairs per task — excellent for both inferring the rule and stress-testing it.
- Scoring (reconstructed from the notebook; **confirm against the real
  `neurogolf_utils.py`**):

  ```
  per-task:  cost   = memory + params           # estimated from the ONNX graph + a runtime trace
             points = max(1, 25 - ln(cost))     # smaller graph -> more points
  total:     sum of points over 400 tasks       # leaders sit around ~7151
  ```

**Two consequences that set our priorities:**

1. **Correctness is a hard gate.** An incorrect network scores ~0 for that task. The points
   formula contains *no* accuracy term — accuracy is enforced separately by functional
   validation. Therefore:
   - Turning a task from **wrong → correct** is worth **~10–18 points** (even with a big net).
   - **Golfing** an already-correct task is worth **fractions of a point**.
   - ⇒ **Coverage first, golf second.** `hoangvux/neurogolf` is a *late-stage golf tool*. We are
     missing the *early-stage coverage engine*. Build that.

2. **Symbolic beats learned, by a lot.** A fixed-weight symbolic graph (rotate/recolor/tile)
   has ~0 params ⇒ `points ≈ 25 - ln(tiny) ≈ 24`. A trained CNN has many params ⇒
   `points ≈ 25 - ln(large) ≈ 12–16`. So hand-/synthesis-built graphs are worth ~1.5–2× a
   trained net **and** are exact. Use trained nets only as a last-resort fallback to convert a
   0-point task into a correct-but-larger ~12-point task.

> Reference point: `7151 / 400 ≈ 17.9` avg points/task. To reach the top you need *nearly all
> 400 correct* AND moderately golfed. Our path: get to "all 400 correct" first (even crudely),
> then climb via golf.

---

## 1. Definition of done & milestone metrics

| Milestone | Definition | Target metric |
|-----------|------------|---------------|
| **M0 — Never sitting ducks** | A valid `submission.zip` with 400 well-formed ONNX files (identity/constant fallbacks) | Submission accepted; score > 0 |
| **M1 — Geometry+color solver** | Auto-solve all tasks expressible as flips/rotations/transpose/translate/tile/crop/recolor | ~X% tasks correct at near-max points |
| **M2 — DSL synthesis** | Compositional search over a primitive library, compiled to ONNX | Coverage up materially |
| **M3 — Full coverage** | Remaining hard tasks solved or covered by trained-CNN fallback | All 400 correct |
| **M4 — Golf loop** | Apply optimization passes (incl. hoangvux's dilation compaction) | Climb toward / past ~7151 |

We track, per task, in a registry: `{status: wrong|correct, method, params, memory, points, onnx_path}`.

---

## 2. Infrastructure & data setup

### 2.1 Get the data (blocker — do this first, in a browser)
- The Kaggle **API can list but not download** competition files until the **rules are accepted**
  in the browser (we hit `403 Forbidden` on `DownloadDataFile`). Action:
  1. Open `https://www.kaggle.com/competitions/neurogolf-2026/rules` and click **"I Understand and Accept"**.
  2. Then: `kaggle competitions download -c neurogolf-2026 -p data/ && unzip data/*.zip -d data/`.
- Files we need locally:
  - `neurogolf_utils/neurogolf_utils.py` — **the real scorer**. Read it line-by-line and
    confirm the exact `score_network()` return tuple, the cost→points formula, any **op
    whitelist, opset constraint, or max-file-size limit**. Our reconstructed formula above is
    from the notebook and must be validated.
  - `task001.json … task400.json` — each has keys `train`, `test`, `arc-gen`, each a list of
    `{"input": grid, "output": grid}` where a grid is a 2-D list of ints 0–9.
  - The ARC-GEN pairs are our generalization safety net — use **all** of them in verification.

### 2.2 Environment
- Mirror the Kaggle image: `numpy`, `pandas`, `onnx`, `onnxruntime`, `onnx-tool==1.0.1`.
  Pin `onnxruntime` to the competition image's version to avoid op-support drift.
- CPU-only inference (the scorer runs `CPUExecutionProvider`). No GPU needed except possibly
  for the Tier-C trainer.

### 2.3 Repo layout
```
neurogolf/
  data/                      # task JSONs + arc-gen (after accepting rules)
  ng/
    codec.py                 # grid <-> (1,10,30,30) one-hot tensor; encode/decode/compare
    onnx_build.py            # graph builder helpers (make_node/initializer/graph/model)
    primitives.py            # numpy reference impl of every DSL primitive
    compile.py               # DSL program -> minimal ONNX graph (one builder per primitive)
    synth.py                 # program-synthesis search engine
    train.py                 # small-CNN fallback: fit + export to ONNX
    verify.py                # exact onnxruntime check + random-equiv + public/arc-gen validation
    scorer.py                # thin wrapper around neurogolf_utils.score_network + points calc
    golf.py                  # optimization passes (dead-node, const-fold, dilation compaction…)
    registry.py              # per-task best store; assemble versioned baseline
  scripts/
    bootstrap_fallback.py    # M0: build identity/constant valid 400-file submission
    solve_all.py             # run synth across all tasks, write registry
    build_submission.py      # assemble submission.zip from registry bests
  tests/
```

---

## 3. Core libraries to build (the scaffolding)

These are shared by every solving tier. Build and unit-test them before any solving.

### 3.1 `codec.py` — the tensor interface (locked by the notebook)
- **Encode:** grid `(H,W)` (H,W ∈ 1..30, values 0..9) → tensor `(1, 10, 30, 30)` float32,
  one-hot over the **channel** axis, padded to 30×30. (Exactly the notebook's `encode_input_grid`.)
- **Decode:** network output `(1,10,30,30)` → grid via `argmax` over channels, then crop to the
  active `H×W` region. ⚠️ **Open question:** how does the scorer know the output H×W? Either the
  graph emits a dynamically-sized tensor, or there's a convention (e.g. argmax + trim trailing
  empty rows/cols). Resolve from `neurogolf_utils.py` and the starter notebook before committing
  to a decode convention — this is the single most important format detail.
- **Compare:** exact equality on the encoded representation (the notebook compares one-hot
  tensors with `np.array_equal`).

### 3.2 `onnx_build.py` — graph construction helpers
- Thin wrappers over `onnx.helper.make_node / make_tensor / make_graph / make_model` and
  `numpy_helper.from_array`.
- A `compose()` that chains primitive sub-graphs (wire output of op N to input of op N+1).
- Always run `onnx.checker.check_model` + `shape_inference.infer_shapes` on output.
- Target the **lowest opset** the scorer accepts (smaller/safer); avoid exotic ops.

### 3.3 `verify.py` — correctness gates (copy the notebook's rigor)
Three independent gates, all must pass:
1. **Exact public validation:** run the ONNX in onnxruntime over the task's `train`+`test`+
   `arc-gen` pairs; output must equal the ground-truth output for **every** pair.
2. **Random-equivalence vs numpy reference:** for synthesized programs, also run the numpy
   reference impl on random + edge-case grids (1×1, all-0, all-9, ramp) and require exact match
   — catches ONNX/numpy divergence and shape bugs the examples miss.
3. **Self-consistency:** re-load the saved `.onnx` from disk and re-run (catches serialization
   issues).

### 3.4 `scorer.py` — score wrapper
- Wrap `neurogolf_utils.score_network(model, trace_path)` exactly as the notebook does:
  enable onnxruntime profiling, run one sample input, pass the trace to the scorer, read
  `(memory, params, …)`, compute `points`.
- Expose `score(onnx_path) -> {memory, params, cost, points}` so synthesis can rank candidates
  by real points, not our estimate.

---

## 4. Solving strategy — tiers (in priority order)

Run tasks through tiers cheapest-first; first tier that produces a *verified* solution wins,
and we keep golfing it later.

### Tier A — Geometry + color (fixed size relation) → **do this first**
Covers a large, high-value chunk at ~0 params (≈ max points). Auto-detected by brute force.

**Detection:** for each task, test a library of parameterized closed-form transforms against
*all* I/O pairs; accept if one matches every pair exactly.

**The ONNX vocabulary (one-hot `(1,10,30,30)` representation):**

| Transform | ONNX realization | Params |
|-----------|------------------|--------|
| Identity / copy | passthrough | 0 |
| Horizontal / vertical flip | `Slice` step −1 (or reversed `Gather`) on W / H | 0 |
| Rotate 90/180/270 | `Transpose` + flip | 0 |
| Diagonal transpose | `Transpose` swap H,W | 0 |
| Translate / shift | `Pad` then `Slice` | 0 |
| Tile grid n×m | `Tile` | 0 |
| Crop subgrid | `Slice` | 0 |
| Integer upscale (block expand) | `Resize` nearest, or `Tile`+`Reshape` | 0 |
| Integer downscale | strided `Slice` / pooling + threshold | 0 |
| Recolor / palette permute | channel-axis `Gather` (0 params) or 1×1 `Conv` 10→10 with a 10×10 permutation weight | ~0–100 |
| Conditional color map A→B | `Equal` + `Where` | 0 |

**Action:** implement each as (numpy reference in `primitives.py`) + (ONNX builder in
`compile.py`), plus a detector that fits parameters (rotation k, tile factors, color map) from
the examples. Expect this tier alone to clear a meaningful share of the 400.

### Tier B — Neighborhood / logical / symmetry (DSL synthesis)
For tasks needing local computation or composition. Build a DSL and search (Section 5–6).

Extra ONNX primitives:
| Operation | ONNX realization |
|-----------|------------------|
| Border / outline detect | fixed 3×3 `Conv` neighbor-count + `Greater`/`Where` |
| Morphology dilate / erode | `MaxPool` / negated `MaxPool` |
| Bounded flood fill (k steps) | **unrolled** `Conv`+`Clip` repeated k times |
| Symmetry repair | `Max`/`Where` of input with its flips/rotations |
| Most/least frequent color | `ReduceSum` over space + `ArgMax`/`ArgMin` → constant grid |
| Count / threshold | `ReduceSum` + comparisons |
| Object gravity (fall) | repeated shift (`Pad`/`Slice`) + `Where` |

### Tier C — Hard / object-level / conditional → fallback
For whatever survives Tier B:
- **C1 deeper synthesis:** extend the DSL with object extraction (connected components via
  unrolled propagation), per-object reasoning.
- **C2 trained-CNN fallback:** ARC-GEN gives plenty of pairs — train a small fully-convolutional
  net for **same-size pixel-to-pixel** tasks, export to ONNX, then golf (prune/quantize). Worth
  ~12 pts vs 0. Size-changing tasks are much harder for a CNN — prefer synthesis there.
- **C3 last resort:** keep the M0 fallback network (valid file, ~0/1 pt) so the submission stays
  complete. Never leave a task without a well-formed `.onnx`.

---

## 5. The DSL / primitive library

- **Reuse, don't reinvent:** seed the primitive set from an existing ARC DSL (e.g. Michael
  Hodel's `arc-dsl`) — but every primitive we adopt must have a **proven ONNX realization**
  (numpy ref + graph builder + equivalence test). A primitive with no compact ONNX form is
  useless here regardless of how well it fits.
- Each primitive = a triple: `(numpy_fn, onnx_builder, param_spec)`.
- Maintain a "primitive cost" (params/ops) so the synthesizer can prefer cheaper programs.

---

## 6. Program-synthesis engine (`synth.py`)

For each unsolved task:
1. **Gather all pairs** (`train`+`test`+`arc-gen`) — the more the better for rejecting
   overfit programs.
2. **Enumerate** DSL programs up to depth K with pruning (type-guided; deduplicate by behavior
   on a canonical probe set; beam by partial-match score).
3. **Test in numpy** against **all** pairs. Accept only on **100% exact** match (a program that
   fits train but fails one arc-gen pair is rejected — that's our generalization guard).
4. **Compile** the accepted program → ONNX (`compile.py`).
5. **Verify** via all three gates in `verify.py`.
6. **Score** with the real scorer; among all passing programs keep the **fewest-params** one.
7. **Register** the winner in `registry.py`.

Notes:
- Run tasks **in parallel** (CPU-bound, independent) — this is the obvious place to fan out.
- Cache canonicalized sub-results to avoid re-evaluating equivalent partial programs.

---

## 7. M0 — never be sitting ducks (do this on day 1)

Before any solving, guarantee a **complete, valid, submittable** artifact:

- `bootstrap_fallback.py` writes 400 well-formed `taskNNN.onnx` files, each a trivial valid graph
  (identity passthrough, or constant most-common-output-color). Some tasks (identity-rule ones)
  will already be *correct*; the rest are valid-but-wrong placeholders.
- Assemble `submission.zip` (exactly 400 files, `ZIP_DEFLATED`) and submit once to confirm the
  pipeline end-to-end against the live leaderboard.
- From here we only ever **replace** a task's file when a verified-better solution appears —
  exactly the conservative ratchet `hoangvux/neurogolf` uses (control zip + single-task swap +
  integrity asserts). We adopt that discipline wholesale.

---

## 8. Golf / optimization pass (`golf.py`)

Once a task is correct, shrink it without changing behavior. This is where the
`hoangvux/neurogolf` techniques plug in directly:
- **Sparse-Conv → dilated-Conv compaction** (their core trick): detect mostly-zero conv kernels
  with evenly-spaced nonzeros, rewrite as smaller kernel + `dilations`, preserving the effective
  receptive field. Lossless, params-only.
- **Dead-node elimination, constant folding, identity-op removal.**
- **dtype shrink** where the scorer counts bytes (e.g. float32→float16/int8 for fixed weights),
  if the scorer and runtime allow it.
- **Operator fusion / simplification** (`onnx-simplifier`-style), re-verified each time.
- After every pass: re-run all three verification gates + re-score; keep only if `points`
  strictly improves and correctness holds. (Mirror their "change one thing, prove it, keep
  control copy" loop.)

---

## 9. Verification & CI loop

- A `make verify TASK=NNN` that runs the three gates + scorer for one task.
- A nightly/`solve_all` job that: pulls registry, re-verifies every current best against
  *fresh* arc-gen samples (guards against silent regressions / overfit), rebuilds
  `submission.zip`, prints total estimated points and a per-task diff vs last version.
- Treat any drop in correctness as a release blocker.

---

## 10. Versioning & iteration management (`registry.py`)

- Store each task's current best as `(onnx_bytes, method, params, memory, points, verified_at)`.
- Version the assembled baseline like `hoangvux` does (`baselinev1/.../19`): every accepted
  improvement bumps a version; `build_submission.py` always assembles from registry bests.
- Keep a byte-exact **control** of the last-submitted zip so we can always fall back.

---

## 11. Suggested sequencing

1. **Day 0–1:** Accept rules → download data → read the real `neurogolf_utils.py` (lock the
   scoring formula, opset/op/size constraints, and the output-shape convention). Build
   `codec.py`, `onnx_build.py`, `scorer.py`, `verify.py`. Ship **M0** fallback submission.
2. **Week 1:** **Tier A** geometry+color auto-solver + detectors. Re-submit. Measure coverage.
3. **Week 2:** **Tier B** DSL + synthesis engine. Expand primitive library. Re-submit.
4. **Week 3:** **Tier C** for the long tail (deeper synthesis, then trained-CNN fallback) →
   push toward all-400-correct.
5. **Ongoing:** **M4 golf** — fold in the `hoangvux` dilation pass and friends; climb the board.

---

## 12. Risks & open questions (resolve early)

- **[CRITICAL] Output shape convention.** How does a network signal a variable output H×W to
  the scorer? Dynamic ONNX output vs. a trim convention. *Must* be answered from
  `neurogolf_utils.py` / starter notebook before building Tier A. Everything downstream depends
  on it.
- **Real scoring formula.** Confirm `score_network()`'s exact return and the cost→points map;
  our `25 - ln(memory+params)` is reconstructed, not authoritative.
- **Op / opset whitelist & file-size cap.** The scorer may only support a subset of ONNX ops or
  cap file size — pick the primitive vocabulary accordingly.
- **arc-gen generalization vs private suite.** Passing all public+arc-gen pairs is our best
  proxy; the private suite could still catch over-fit rules. Bias synthesis toward the
  simplest program that fits (Occam) to maximize generalization.
- **Runtime version drift.** onnxruntime op behavior can differ between our env and the scorer's
  image — pin versions; verify in the matching image.
- **Synthesis blow-up.** Deep DSL search is exponential — invest in pruning, behavioral dedup,
  and per-task time budgets; accept Tier-C fallback when search times out.

---

## 13. Immediate next actions

- [ ] Accept competition rules in browser; download data; **read `neurogolf_utils.py` in full.**
- [ ] Confirm the **output-shape convention** and the **exact scoring formula** from that file.
- [ ] Implement `codec.py` + `scorer.py` + `verify.py` and unit-test against a few task JSONs.
- [ ] Ship the **M0** 400-file fallback `submission.zip` and confirm a non-zero leaderboard score.
- [ ] Stand up the Tier-A geometry+color auto-solver and measure first real coverage.

---

### Appendix — what `hoangvux/neurogolf` is, for context
That notebook is **only the golf stage (M4)**: it loads an *already-correct* 400-network baseline
(`baselinev1` v19, scoring 7151.32), finds sparse convs it can re-express as dilated convs
(fewer params, identical output), verifies equivalence + public correctness + official score,
and swaps in **one** improved task per run. It never builds or solves a task — it assumes the
baseline this document describes already exists. This plan builds that prerequisite.
