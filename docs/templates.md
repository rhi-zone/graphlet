# Template matching

For matching an **arbitrary** query graph (not a small named catalog motif),
`graphlet` provides a thin parallel arm delegating to petgraph's VF2
`subgraph_isomorphisms_iter`:

```rust
use graphlet::template::{induced_matches_unlabelled, count_induced_matches};

let matches = induced_matches_unlabelled(&pattern, &host); // Vec<Vec<usize>>
let n = count_induced_matches(&pattern, &host);
```

## Induced-native

petgraph's `subgraph_isomorphisms_iter` returns **node-induced** subgraph
isomorphisms natively (its docstring states "'subgraph' always means a 'node-induced
subgraph'", its `is_feasible` rejects extra host edges, and empirically a P3 pattern
finds 0 matches in a triangle host). So the induced arm is free — `graphlet` delegates
and does no filtering. Pass `node_match` / `edge_match` predicates via
`induced_matches` to match on node/edge attributes.

## Why no non-induced template arm

Non-induced (monomorphism) matching of an arbitrary template is **deliberately not
implemented**. It cannot be recovered by filtering petgraph's output (induced ⊂
monomorphism — a post-filter only shrinks the set), the k-bounded `s(P,C)` trick from
the [catalog](/motifs) does not apply to an unbounded template, and no consumer needs
it today (science domains are induced; small named software patterns are served by
the catalog arm). It is reserved as a future additive method, never an erroring
runtime toggle.
