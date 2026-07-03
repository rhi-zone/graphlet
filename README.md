# graphlet

A petgraph-native Rust library for graphlet analysis of undirected simple graphs. It
computes:

- the connected induced-subgraph census (each connected induced subgraph up to 5 nodes,
  folded by canonical isomorphism class),
- per-node graphlet degree vectors and their distribution (GDV / GDD) across all 73
  orbits of the graphlets up to 5 nodes,
- named-motif detection (diamonds), reporting both induced and non-induced counts,
- induced matching of arbitrary template graphs via petgraph's VF2.

It runs directly on petgraph's own types, generic over `Graph` and `StableGraph`, so it
reads the graph you already have. It depends only on `petgraph` and `rand`.

Documentation lives at <https://docs.rhi.zone/graphlet/>.

## Example

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

- Enumerates and counts connected induced subgraphs (the subgraph census), assigning
  each a canonical isomorphism-class label so structurally identical subgraphs fold
  together regardless of node numbering. `enumerate` yields instances lazily; `count`
  streams the same traversal into per-class counts.
- Computes per-node graphlet degree vectors (GDV) and the graphlet degree distribution
  (GDD) across all 73 orbits.
- Detects named motifs (diamonds), giving induced counts (read off the census) and
  non-induced counts (derived from the induced census through a fixed conversion table,
  not a separate enumerator).
- Matches arbitrary user-supplied template graphs by induced subgraph isomorphism,
  delegating to petgraph's VF2 `subgraph_isomorphisms_iter`.

## Scope

- Undirected simple graphs. Self-loops are ignored and parallel edges are deduplicated.
  Directed input is analyzed on its underlying undirected structure.
- Subgraph size up to 5 nodes.
- Generic over petgraph's `Graph` and `StableGraph`, over directed and undirected graphs,
  and over arbitrary node and edge weights.

## What it does not do

- No directed motifs beyond triads (no directed graphlets at k >= 4).
- No statistical significance testing (no z-scores of observed counts).
- No null-model generators.
- No graph kernels.
- No neighborhood statistics (link prediction, assortativity, rich-club coefficients).
- The 5-node census uses naive canonicalization and is not tuned for very large graphs,
  so it is slow on big inputs.
- No non-induced matching of arbitrary templates (only induced template matching).

## What to use instead

- petgraph: core graph algorithms, and VF2 subgraph isomorphism itself.
- rustworkx-core: a broad algorithm suite (shortest paths, centrality, DAG operations,
  generators).
- graphalgs: spectral and distance-based metrics.
- triadic-census: directed triad census.
- heterogeneous_graphlets: graphlets on heterogeneous graphs.
- ORCA and FANMOD (not Rust): mature references for orbit counting at scale and for motif
  significance testing.

## Install

`graphlet` is not yet published to crates.io, so depend on it by git for now:

```toml
[dependencies]
graphlet = { git = "https://github.com/rhi-zone/graphlet" }
petgraph = "0.8"
```

## License

MIT. See the `LICENSE` file.
