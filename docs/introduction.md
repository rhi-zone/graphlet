# Introduction

`graphlet` is a petgraph-native library for the **structural / network-science
mining** subfield: graphlet and orbit statistics, subgraph census, and network-motif
detection. It is the home for the pieces genuinely absent from the Rust graph
ecosystem — the naive rest of the domain (traversal, shortest-path, flow, MST,
centrality, isomorphism, planarity) is already served by petgraph, rustworkx-core,
and graphalgs, which `graphlet` builds on rather than rebuilding.

It depends only on `petgraph` and `rand`, owning every small, well-understood
algorithm rather than relabelling a transitive trust chain through a facade.

## The shape

- **[Census substrate](/census):** the organizing center — *enumerate connected
  k-subsets → canonically label → fold* — with lazy instance enumeration and
  streaming counting as two readouts of one pass.
- **[Orbits (GDV/GDD)](/orbits):** per-node attribution to all 73 automorphism
  orbits at k ≤ 5 (undirected).
- **[Named motifs](/motifs):** a catalog (seeded with the diamond) that threads
  induced vs. non-induced semantics honestly.
- **[Template matching](/templates):** a thin parallel arm over petgraph's VF2,
  induced-native.

## Scope today

Undirected graphlets at k ≤ 5 are implemented and verified. Null-model generators,
graph kernels, motif significance, neighborhood statistics, scalable k=5, and
directed k ≥ 4 are documented as future work (see the crate's `rim` module).

## Genericity

Every entry point is generic over `Graph` / `StableGraph`, directed / undirected,
and arbitrary node/edge weights through one trait-bound set. Directed graphs are
analyzed on their underlying undirected structure.
