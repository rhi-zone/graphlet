//! Graphlet analysis for [petgraph](https://docs.rs/petgraph): connected subgraph
//! census, graphlet-degree vectors (GDV/GDD), per-node orbit attribution, and
//! network-motif detection.
//!
//! `graphlet` is a petgraph-native home for the structural / network-science
//! *mining* subfield — the pieces that are genuinely absent from the Rust graph
//! ecosystem (traversal, shortest-path, flow, isomorphism, planarity already live
//! in petgraph / rustworkx-core / graphalgs). It depends only on `petgraph` and
//! `rand`, owning every small, well-understood algorithm rather than relabelling a
//! transitive trust chain through a facade. See ADR-0290 in the rhi ecosystem docs
//! for the full rationale.
//!
//! # The census substrate
//!
//! The organizing center is a single pipeline — *enumerate connected k-subsets →
//! canonically label → fold* — with instance-enumeration and counting as two
//! readouts of one pass:
//!
//! - [`enumerate`] yields each connected induced k-subgraph lazily as an
//!   [`Instance`] carrying the host `NodeId`s and its [`ClassId`].
//! - [`count`] folds the same traversal into a [`Census`] (class → count) as a
//!   stream: its memory tracks graph size `O(V+E)`, not the number of instances.
//!
//! The traversal owns an `O(V+E)` adjacency [`Snapshot`] and is generic over
//! `Graph`/`StableGraph` × directedness × arbitrary weights through one trait-bound
//! set ([`GraphAdapter`]); analysis is on the *undirected* structure (in/out
//! neighborhoods are unioned).
//!
//! # Orbits (GDV / GDD)
//!
//! [`graphlet_degree_vectors`] attributes every node of every instance to its
//! automorphism orbit within the graphlet, producing the 73-entry graphlet-degree
//! vector per node (k ≤ 5, undirected) via a union-find automorphism [`Registry`].
//!
//! # Named motifs
//!
//! [`catalog`] provides a real motif catalog: named [`Pattern`](catalog::Pattern)
//! constructors for the standard small motifs (path `P_k`, cycle `C_k`, star, complete
//! `K_k`, the six connected 4-node motifs, the diamond), a
//! [`MotifCatalog`](catalog::MotifCatalog) to register and query arbitrary patterns by
//! name, and the general motif queries [`find_motif`](catalog::find_motif) /
//! [`count_motif`](catalog::count_motif) (with [`find_diamonds`](catalog::find_diamonds)
//! now a thin wrapper). Arbitrary connected patterns are countable directly via
//! [`Pattern::new`](catalog::Pattern::new) + [`count_pattern`](catalog::count_pattern).
//!
//! The census / catalog arm threads [`Induced`](catalog::Induced): induced *counts* are
//! read off the census and non-induced *counts* use the verified `s(P,C)` derivation (no
//! enumerator needed) at `k <= 5`; *instances* ([`find_motif`](catalog::find_motif)) are
//! enumerated by the pattern-instance engine below and deduped to distinct occurrences.
//!
//! # Template matching (the pattern-instance engine)
//!
//! [`template`] matches an arbitrary petgraph query graph against a host in two
//! honestly-distinct semantics, unbounded in `k`, honouring node/edge predicates and
//! directedness: **induced** ([`induced_matches`](template::induced_matches)) delegates
//! to petgraph's node-induced-native VF2 `subgraph_isomorphisms_iter`, and **non-induced
//! / monomorphism** ([`monomorphisms`](template::monomorphisms)) is this crate's own
//! ordered-backtracking edge-preserving enumerator (petgraph provides no monomorphism
//! search; it cannot be recovered by filtering the induced output). Both return raw
//! embeddings; divide by `|Aut(P)|` for distinct occurrences.
//!
//! # Example
//!
//! Build a petgraph graph, then read the three primary outputs — a class [`count`],
//! per-node [`graphlet_degree_vectors`], and a named-motif
//! [`find_diamonds`](catalog::find_diamonds) query — off the one census substrate:
//!
//! ```
//! use graphlet::catalog::{find_diamonds, Induced};
//! use graphlet::{count, graphlet_degree_vectors, Registry, Selector};
//! use petgraph::graph::UnGraph;
//!
//! // A diamond (K4 minus one edge: two triangles sharing the 0–2 spine) plus a
//! // pendant vertex 4 hanging off 3.
//! let g: UnGraph<(), ()> =
//!     UnGraph::from_edges([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (3, 4)]);
//!
//! // Class census of connected 3-node induced subgraphs (triangles + paths).
//! let census = count(&g, &Selector::connected_k_subsets(3));
//! assert_eq!(census.values().sum::<u64>(), 6);
//!
//! // Per-node graphlet-degree vectors (73 orbits, k <= 5); the registry is reusable.
//! let reg = Registry::build();
//! let gdv = graphlet_degree_vectors(&g, &reg);
//! assert_eq!(gdv.orbit_count(), 73);
//!
//! // Named-motif query: exactly one induced diamond.
//! assert_eq!(find_diamonds(&g, Induced::Yes).len(), 1);
//! ```

#![warn(missing_docs)]

mod canonical;
mod census;
mod orbit;
mod snapshot;

pub mod catalog;
pub mod rim;
pub mod template;

pub use canonical::ClassId;
pub use census::{count, enumerate, Census, Instance, Instances, Selector};
pub use orbit::{graphlet_degree_vectors, GdvTable, Registry};
pub use snapshot::{GraphAdapter, Snapshot};

#[cfg(test)]
mod tests;
