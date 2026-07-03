# graphlet

Graphlet analysis for [petgraph](https://docs.rs/petgraph): connected subgraph
census, graphlet-degree vectors (GDV/GDD), per-node orbit attribution, and
network-motif detection.

`graphlet` is a petgraph-native home for the **structural / network-science mining**
subfield — the pieces genuinely absent from the Rust graph ecosystem. The naive rest
of the graph domain (traversal, shortest-path, flow, MST, centrality, isomorphism,
planarity) is already well served by petgraph, rustworkx-core, and graphalgs;
`graphlet` builds on those rather than rebuilding them. It depends on **only
`petgraph` and `rand`**, owning every small, well-understood algorithm rather than
relabelling a transitive trust chain through a facade.

Documentation: <https://docs.rhi.zone/graphlet/>

## The census substrate

The organizing center is a single pipeline — *enumerate connected k-subsets →
canonically label → fold* — with instance enumeration and counting as two readouts of
one pass:

```rust
use graphlet::{count, enumerate, Selector};
use petgraph::graph::UnGraph;

let g: UnGraph<(), ()> = UnGraph::from_edges([(0, 1), (1, 2), (2, 0), (2, 3)]);
let sel = Selector::connected_k_subsets(3);

// Streaming class census (memory tracks graph size, not #instances):
let census = count(&g, &sel);

// Lazy instance enumeration (each carries host NodeIds + its ClassId):
for inst in enumerate(&g, &sel) {
    let _ = (&inst.nodes, inst.class);
}
```

The traversal owns an `O(V+E)` adjacency snapshot and is generic over `Graph` /
`StableGraph`, directed / undirected, and arbitrary node/edge weights through one
trait-bound set. Directed graphs are analyzed on their underlying undirected
structure.

## Graphlet-degree vectors (GDV / GDD)

```rust
use graphlet::{graphlet_degree_vectors, Registry};

let reg = Registry::build();                // 73 orbits (k ≤ 5), reusable
let gdv = graphlet_degree_vectors(&g, &reg);
let row = gdv.row(0);                        // 73-entry vector for node index 0
```

Every node of every instance is attributed to its automorphism orbit; verified
node-for-node against an independent brute-force oracle.

## Named motifs

```rust
use graphlet::catalog::{count_diamonds, find_diamonds, Induced};

let induced     = count_diamonds(&g, Induced::Yes);
let non_induced = count_diamonds(&g, Induced::No);   // K4s contribute 6 diamonds each
let occurrences = find_diamonds(&g, Induced::Yes);   // spine + tips per occurrence
```

The catalog arm threads `Induced` honestly: induced counts come from the census;
non-induced (monomorphism) counts are derived from the induced census via a fixed
`s(P,C)` table (verified 1105/1105 against a brute-force oracle at k = 3,4,5), with no
separate monomorphism enumerator.

## Template matching

For an arbitrary query graph, `graphlet::template` is a thin wrapper over petgraph's
VF2 `subgraph_isomorphisms_iter`, which is **node-induced native**. Non-induced
matching of an arbitrary template is deliberately deferred (see the docs).

## Status

In development. Implemented and verified: the undirected census substrate at k ≤ 5,
canonical labelling and stable class ids, per-node orbits (GDV/GDD), the diamond
catalog with induced and non-induced semantics, and the induced VF2 template arm.
Documented as future work (the `rim` module): null-model generators, graph kernels,
motif significance / z-scores, neighborhood statistics (link-prediction,
assortativity, rich-club), scalable k=5 (ORCA / g-trie), and directed k ≥ 4.

## License

MIT.
