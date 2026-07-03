# Orbits (GDV / GDD)

Two nodes of one graphlet are in the same **orbit** when an automorphism of that
graphlet maps one to the other. There are **73 orbits** across connected graphlets of
order 2..=5 (undirected). The **graphlet-degree vector (GDV)** of a node counts, per
orbit, the instances in which that node occupies the orbit.

```rust
use graphlet::{graphlet_degree_vectors, Registry};

let reg = Registry::build();               // 73 orbits; reusable across graphs
let gdv = graphlet_degree_vectors(&g, &reg);

let row = gdv.row(0);                       // 73-entry vector for node index 0
let dist = gdv.degree_distribution(orbit);  // GDD: degree → number of nodes
```

For every subgraph instance, each participating node is attributed to its orbit via
the arg-permutation witnessing the instance→canonical isomorphism; the orbit is
looked up in a union-find automorphism registry. Because orbit ids are
automorphism-invariant, the attribution is independent of enumeration order.

The attribution is verified against an independent brute-force per-node oracle (zero
mismatches on paths, cycles, stars, cliques, and fuzzed random graphs) and satisfies
`Σ_v GDV[v][orbit] = census(class) · orbit_size`.

::: info Orbit numbering
The internal orbit ids are stable and deterministic but are **not** ORCA's published
orbit ordering — aligning to ORCA is a mechanical relabelling.
:::
