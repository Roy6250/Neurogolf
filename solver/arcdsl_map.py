"""Map NeuroGolf tasks -> arc-dsl reference solvers, and analyse primitive frequency.

For each NeuroGolf task we run every arc-dsl solve_<hash> on the task's train inputs and
keep the solver(s) that reproduce ALL train outputs exactly. This (a) tells us which tasks
have a known reference program (the transpiler's coverage ceiling), and (b) counts which DSL
primitives appear most across matched solvers — i.e. what to implement in ONNX first.

Point ARC_DSL at a local clone of github.com/michaelhodel/arc-dsl.

  ./.venv/bin/python solver/arcdsl_map.py [ARC_DSL_DIR]
"""
import ast
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "Dataset"
ARC_DSL = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/private/tmp/claude-501/-Users-ananyaroy-sayantan/"
    "0aaa9c77-4c01-4eec-b934-8bb413b54cbc/scratchpad/arc-dsl")


def load_solvers():
    sys.path.insert(0, str(ARC_DSL))
    import solvers as S
    funcs = {n[len("solve_"):]: getattr(S, n) for n in dir(S) if n.startswith("solve_")}
    src = (ARC_DSL / "solvers.py").read_text()
    return funcs, src


def solver_sources(src):
    """hash -> list of primitive names called in solve_<hash> (via AST)."""
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("solve_"):
            calls = [n.func.id for n in ast.walk(node)
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
            out[node.name[len("solve_"):]] = calls
    return out


def as_tuple(grid):
    return tuple(tuple(int(v) for v in row) for row in grid)


def main():
    funcs, src = load_solvers()
    prim_calls = solver_sources(src)

    matched = {}          # taskNNN -> hash
    for i in range(1, 401):
        tf = DATA / f"task{i:03d}.json"
        if not tf.exists():
            continue
        train = json.loads(tf.read_text()).get("train", [])
        if not train:
            continue
        pairs = [(as_tuple(e["input"]), as_tuple(e["output"])) for e in train]
        for h, fn in funcs.items():
            try:
                if all(fn(inp) == out for inp, out in pairs):
                    matched[f"task{i:03d}"] = h
                    break
            except Exception:
                continue

    print(f"matched {len(matched)}/400 tasks to an arc-dsl solver\n")

    # primitive frequency over matched solvers
    prim_freq = Counter()
    prog_len = Counter()
    for tn, h in matched.items():
        calls = prim_calls.get(h, [])
        prog_len[len(calls)] += 1
        for c in set(calls):
            prim_freq[c] += 1

    print("most common DSL primitives across matched solvers (task count):")
    for p, c in prim_freq.most_common(30):
        print(f"  {p:18s} {c}")

    print("\nsolver length distribution (#primitive calls -> #tasks):")
    for L in sorted(prog_len):
        print(f"  len {L:2d}: {prog_len[L]}")

    # how many tasks are solvable with only the top-K primitives
    top = [p for p, _ in prim_freq.most_common()]
    for K in (5, 10, 15, 20, 30):
        allowed = set(top[:K])
        n = sum(1 for h in matched.values()
                if set(prim_calls.get(h, [])) <= allowed)
        print(f"tasks whose solver uses only the top-{K} primitives: {n}")

    (REPO / "arcdsl_matched.json").write_text(json.dumps(matched, indent=0))
    print(f"\nwrote arcdsl_matched.json ({len(matched)} tasks)")


if __name__ == "__main__":
    main()
