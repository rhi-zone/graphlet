# graphlet

`graphlet` is a Rust library for [graphlet](https://en.wikipedia.org/wiki/Graphlets) and
[network-motif](https://en.wikipedia.org/wiki/Network_motif) mining on top of
[petgraph](https://docs.rs/petgraph): it counts and enumerates small connected
subgraphs, attributes nodes to their graphlet orbits, finds and tests the
significance of named motifs, matches arbitrary template patterns, generates
null-model random graphs, and computes a few structural graph statistics
(clustering, assortativity, link prediction, graph kernels). It is for anyone
doing structural/network-science analysis on a petgraph graph who needs
motif-level detail that the rest of the Rust graph ecosystem does not provide.
It depends only on `petgraph` and `rand`, and works directly on `Graph`,
`StableGraph`, and `DiGraph`, directed or undirected, with any node/edge
weights.

## At a glance

```rust
use graphlet::catalog::{find_diamonds, Induced};
use graphlet::{count, graphlet_degree_vectors, Registry, Selector};
use petgraph::graph::UnGraph;

// A diamond (K4 minus one edge: two triangles sharing the 0-2 spine) with a
// pendant vertex 4 hanging off node 3.
let g: UnGraph<(), ()> =
    UnGraph::from_edges([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (3, 4)]);

// Class census of connected 3-node induced subgraphs (triangles and paths).
let census = count(&g, &Selector::connected_k_subsets(3));
assert_eq!(census.values().sum::<u64>(), 6);

// Per-node graphlet degree vectors across all 73 orbits (k <= 5).
// The registry is built once and reused across calls.
let reg = Registry::build();
let gdv = graphlet_degree_vectors(&g, &reg);
assert_eq!(gdv.orbit_count(), 73);

// Named-motif query: this graph contains exactly one induced diamond.
assert_eq!(find_diamonds(&g, Induced::Yes).len(), 1);
```

## What it does

### Subgraph census

The organizing center of the crate is a single pipeline: enumerate every
connected induced subgraph up to 5 nodes, label it by canonical isomorphism
class, and fold. `enumerate` yields each match lazily as an `Instance`
(the host node ids plus its class); `count` folds the same traversal into a
`Census` (class to count) whose memory tracks graph size, not the number of
matches. Both are generic over `Graph`, `StableGraph`, and `DiGraph`; the
underlying structure is always treated as simple and undirected (self-loops
stripped, parallel edges deduped, directed edges unioned).

### Graphlet degree vectors (GDV/GDD)

`graphlet_degree_vectors` attributes every node of every matched subgraph to
its automorphism orbit within that subgraph, producing the 73-entry
graphlet-degree vector per node, for graphlets of order 2 through 5. A
`Registry` computes the orbit structure once and is reused across calls.

The exact path canonicalizes every subgraph instance, which gets expensive at
k=5. `rim::scalable::fast_graphlet_degree_vectors` recovers the same 73-orbit
result an ORCA-style way, without ever canonically labelling a connected
5-node subset: it counts degrees, per-edge triangles, and per-vertex K4
memberships directly for orders 2 through 4, and decomposes order-5 orbits
through connected 4-node cores plus attachment tallies. It is verified exact,
node-for-node and count-for-count, against the exact census path across a
battery of standard graphs (paths, cycles, stars, complete graphs, wheels,
trees, complete bipartite graphs, the Petersen graph, the cube graph Q3) plus
fuzzed random graphs. Because the exact path pays a `k!` canonicalization per
5-subset while the fast path never labels one, the gap widens with graph size
and density rather than sitting at a fixed multiplier: on seeded G(n, p) random
graphs it grows from roughly 10x at n=40 to over 100x by n=150, and past there
the exact path leaves the seconds regime entirely while the fast path stays in
the millisecond-to-few-second range (at n=200, p=0.1 and n=100+, p=0.3 the exact
census is already impractical to time, yet the fast path finishes in a few
seconds). The shipped `examples/scalable_bench.rs` reproduces the whole sweep;
run it with `cargo run --release -p graphlet --example scalable_bench`.

### Named motifs

`catalog` provides named `Pattern` constructors for the standard small
motifs, path (`P_k`), cycle (`C_k`), star, complete graph (`K_k`), triangle,
claw, paw, diamond, plus a `MotifCatalog` to register and look up arbitrary
patterns by name. `count_pattern` / `count_motif` count any connected pattern
of order up to 5 against a host graph, in both induced and non-induced
semantics; `find_motif` returns the actual node-set occurrences (`find_diamonds`
is a thin wrapper over `find_motif` for the diamond pattern).

### Template matching

`template` matches an arbitrary petgraph query graph against a host graph, in
two distinct semantics, unbounded in pattern size and honoring node/edge
predicates and directedness:

- **Induced** (`induced_matches`): delegates to petgraph's own VF2
  (`subgraph_isomorphisms_iter`).
- **Non-induced / [monomorphism](https://en.wikipedia.org/wiki/Subgraph_isomorphism_problem)**
  (`monomorphisms`): this crate's own
  ordered-backtracking, edge-preserving enumerator (petgraph has no
  monomorphism search).

Both return raw embeddings (every automorphism of the pattern counted
separately); divide by `|Aut(pattern)|` for distinct occurrences.

### Null-model generators (`rim::null_model`)

Random-graph generators for significance testing, each seeded via `&mut impl
Rng`: the [configuration model](https://en.wikipedia.org/wiki/Configuration_model) (raw
and simple/loop-free variants), degree-preserving double-edge-swap rewiring,
[Watts-Strogatz](https://en.wikipedia.org/wiki/Watts%E2%80%93Strogatz_model) small-world
graphs, and an
[LFR benchmark](https://en.wikipedia.org/wiki/Lancichinetti%E2%80%93Fortunato%E2%80%93Radicchi_benchmark)
graph with planted community structure. LFR is a documented partial:
it approximates the target degree and mixing parameters rather than enforcing
them exactly, and has other known deviations from the reference algorithm
(see its doc comment).

### Significance testing (`rim::significance`)

`motif_significance` and `census_significance_profile` compare an observed
motif count, or the full graphlet census, against an ensemble of null graphs,
reporting a z-score and an empirical p-value per target. The significance
profile's normalized (unit-length) vector is `None` whenever any z-score in
the profile is non-finite (which happens when a null model's count has zero
variance but disagrees with the observed count) rather than silently leaking
a `NaN`.

### Neighborhood statistics (`rim::neighborhood`)

[Link-prediction](https://en.wikipedia.org/wiki/Link_prediction) indices (common
neighbors, Jaccard, Adamic-Adar, resource allocation, preferential attachment),
local/average/global
[clustering coefficients](https://en.wikipedia.org/wiki/Clustering_coefficient) with
triangle counting,
[degree assortativity](https://en.wikipedia.org/wiki/Assortativity) (Newman/Pearson),
and the [rich-club coefficient](https://en.wikipedia.org/wiki/Rich-club_coefficient).

### [Graph kernels](https://en.wikipedia.org/wiki/Graph_kernel) (`rim::kernels`)

A graphlet kernel (built directly from the census substrate's own class-count
vectors), a
[Weisfeiler-Lehman](https://en.wikipedia.org/wiki/Weisfeiler_Leman_graph_isomorphism_test)
subtree kernel, a [shortest-path](https://en.wikipedia.org/wiki/Shortest_path_problem)
kernel, and a generic Gram-matrix builder (raw or cosine-normalized) usable with any of
the three feature representations, or your own.

### Directed graphlets (`rim::directed`)

A directed generalization of the census/orbit substrate, respecting arc
direction throughout (two directed graphlets that fold to the same undirected
class, such as `a -> b -> c` and `a -> b <- c`, are distinct directed
classes): directed graphlet census, per-node directed orbits, for
weakly-connected subgraphs of order 2 through 5, plus the standard 16-type
Holland-Leinhardt directed triad census (order 3, every 3-subset, connected or
not). k=5 is exact but slow: canonicalizing a directed 5-node instance costs
5! permutations of a 20-bit mask, and building the order-5 registry sweeps all
2^20 labelled digraphs once.

## Examples

### Census and graphlet degree vectors

```rust
use graphlet::{count, graphlet_degree_vectors, Registry, Selector};
use petgraph::graph::UnGraph;

// A 5-cycle.
let g: UnGraph<(), ()> = UnGraph::from_edges([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]);

let census = count(&g, &Selector::connected_k_subsets(3));
// Every 3-subset of a 5-cycle is a path, never a triangle.
assert_eq!(census.len(), 1);

let reg = Registry::build();
let gdv = graphlet_degree_vectors(&g, &reg);
// Every node of a symmetric cycle sits in the same orbit for every orbit id.
let row0 = gdv.row(0).to_vec();
for v in 1..5 {
    assert_eq!(gdv.row(v), row0.as_slice());
}
```

### Motif significance against a null model

```rust
use graphlet::catalog::{Induced, Pattern};
use graphlet::rim::significance::{motif_significance, NullModel};
use petgraph::graph::UnGraph;
use rand::SeedableRng;
use rand::rngs::StdRng;

// Five disjoint triangles: over-represented relative to a degree-preserving
// null (which breaks triangles up into larger cycles).
let g: UnGraph<(), ()> = UnGraph::from_edges([
    (0u32, 1), (1, 2), (2, 0),
    (3, 4), (4, 5), (5, 3),
    (6, 7), (7, 8), (8, 6),
    (9, 10), (10, 11), (11, 9),
    (12, 13), (13, 14), (14, 12),
]);
let mut rng = StdRng::seed_from_u64(42);
let tri = Pattern::triangle();
let results = motif_significance(
    &g,
    &[("triangle", &tri, Induced::Yes)],
    40,
    NullModel::DegreePreserving { n_swaps_per_edge: 10 },
    &mut rng,
);
let (_name, entry) = &results[0];
assert!(entry.z_score > 0.0, "triangles should be over-represented");
```

### Graphlet kernel between two graphs

```rust
use graphlet::rim::kernels::{graphlet_features, graphlet_kernel_cosine};
use petgraph::graph::UnGraph;

let triangle: UnGraph<(), ()> = UnGraph::from_edges([(0, 1), (1, 2), (2, 0)]);
let path: UnGraph<(), ()> = UnGraph::from_edges([(0, 1), (1, 2)]);

let ft = graphlet_features(&triangle, 3);
let fp = graphlet_features(&path, 3);
// Different classes at k=3, so they share no graphlet-class counts.
assert_eq!(graphlet_kernel_cosine(&ft, &fp), 0.0);
```

## Scope, and what this is not

- The census, orbit, catalog, and kernel machinery treats every input as a
  *simple undirected* graph: self-loops are stripped, parallel edges (and
  directed reciprocal pairs) are deduped, and directed input is analyzed on
  its underlying undirected structure. Only `rim::directed` and the triad
  census respect arc direction.
- Graphlets and orbits are bounded to order 5 (k <= 5); this is where the
  "science-facing" surface (GDV/GDD, orbits, the named-motif catalog) is
  closed. The `template` module has no such bound and matches patterns of any
  size, at the cost of being a search rather than a closed-form count.
- Exact k=5 enumeration (both the undirected census/orbit path and the
  directed graphlet census) is slow on large or dense graphs: canonicalizing
  each instance costs up to `k!` permutations. The fast ORCA-style path
  (`rim::scalable`) covers only undirected orbit counting; there is no
  equivalent fast path yet for the directed side or for k > 5.
- This is a structural-mining / graphlet library, not a general graph-algorithms
  toolkit. It has no shortest paths, no centrality measures, no flow
  algorithms, and no general isomorphism beyond what the census/template
  machinery needs. For those, use `petgraph` directly (which also provides the
  VF2 isomorphism search this crate builds on) or `rustworkx-core` for a
  broader algorithm suite.

## License

MIT. See the `LICENSE` file.
