# graphlet

`graphlet` is a petgraph-native library for graphlet analysis and small-subgraph
structural mining. It is built directly on petgraph's own graph types, so it reads the
graph you already have rather than asking you to convert into a bespoke representation.
It depends on only `petgraph` and `rand`, and it owns the small, well-understood
algorithms it needs instead of pulling in a large graph-analysis stack.

Documentation lives at <https://docs.rhi.zone/graphlet/>.

## What it does

The core of the library enumerates and counts the connected induced subgraphs of a
graph (the subgraph census) for subgraphs of up to five nodes, on undirected simple
graphs. Each enumerated subgraph is assigned a canonical isomorphism-class label, so
structurally identical subgraphs are folded together regardless of how their nodes
happen to be numbered in the host graph.

On top of that census it computes per-node graphlet degree vectors and their
distribution across all 73 orbits of the graphlets up to five nodes (the GDV and GDD).
It detects named motifs such as the diamond, reporting both induced counts (read
directly off the census) and non-induced counts (derived from the induced census
through a fixed conversion table rather than a separate enumerator). It also matches
arbitrary user-supplied template graphs by induced subgraph isomorphism, delegating to
petgraph's VF2 implementation.

All of this works generically over petgraph's `Graph` and `StableGraph`, over directed
and undirected graphs (directed input is analyzed on its underlying undirected
structure), and over arbitrary node and edge weights.

## When to use it

Reach for `graphlet` when you already have a petgraph graph and you want graphlet or
motif structural analysis of it. It lets you do that analysis without converting your
graph into another library's types and without taking on a large dependency to get a
handful of census-based measures.

## When not to use it

`graphlet` treats its input as a simple undirected graph. Self-loops are ignored and
parallel edges are deduplicated, so if those carry meaning in your data the results
will not reflect them. It does not do directed motifs beyond the triad level, and it has
no directed graphlets at four nodes or more.

It also does not yet do statistical significance testing, meaning z-scores of observed
motif counts against random null models. It has no null-model generators, no graph
kernels, and no neighborhood statistics such as link prediction, assortativity, or
rich-club coefficients. These are planned rather than present (see ADR-0290 in the rhi
ecosystem docs for the rationale and scope).

Two further limits are worth stating plainly. The five-node census uses naive
canonicalization and is not tuned for very large graphs, so on big inputs it will be
slow. And while it matches arbitrary templates by induced isomorphism, it does not do
non-induced matching of arbitrary templates.

## Where to go instead

If your need falls outside the above, other libraries serve it better. For core graph
algorithms, and for VF2 subgraph isomorphism itself, use petgraph directly. For a broad
suite of graph algorithms such as shortest paths, centrality, DAG operations, and graph
generators, use rustworkx-core. For spectral and distance-based metrics, use graphalgs.
For a directed triad census, use triadic-census. For graphlets on heterogeneous graphs,
use heterogeneous_graphlets.

For orbit counting at scale and for motif significance testing, the established
references are ORCA and FANMOD. They are not Rust libraries, but if you need fast orbit
counting on large graphs or rigorous motif significance today, they are the mature tools
to reach for.

## Usage

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

## Install

`graphlet` is not yet published to crates.io, so depend on it by git for now:

```toml
[dependencies]
graphlet = { git = "https://github.com/rhi-zone/graphlet" }
petgraph = "0.8"
```

## License

MIT. See the `LICENSE` file.
