# Census substrate

The organizing center is a single pipeline:

```
enumerate connected k-subsets → canonically label → fold
```

Connected induced k-subgraphs are enumerated exactly once each by ESU
(Wernicke 2006). The traversal owns an `O(V+E)` adjacency snapshot of the graph;
petgraph's only `O(1)` borrowing adjacency probe is its `O(V²)` adjacency matrix,
infeasible at scale, so the core snapshots once at construction and then runs on
plain indices.

## Two readouts of one pass

```rust
use graphlet::{count, enumerate, Selector};

let sel = Selector::connected_k_subsets(4);

// Lazy instance enumeration: each Instance carries host NodeIds + its ClassId.
for inst in enumerate(&g, &sel) {
    // inst.nodes, inst.class
}

// Streaming count: a class → count map, folded without materializing instances.
let census = count(&g, &sel);
```

`enumerate` is a lazy explicit-stack iterator whose frames are bounded by `k`, so
`count` (which drives an allocation-free recursive visitor) keeps peak memory
tracking graph size `O(V+E)` rather than the number of instances. Collecting the
iterator (`enumerate(&g, &sel).collect()`) is available when you actually want every
instance — at `O(instances)` memory by construction.

## Canonical labelling

A k-node induced subgraph is labelled by the minimum, over all `k!` vertex
permutations, of the packed upper-triangle adjacency bitmask. The minimum is
insertion-order invariant, so two subgraphs share a `ClassId` iff they are
isomorphic. Class counts are the classical 2 / 6 / 21 at k = 3 / 4 / 5.
