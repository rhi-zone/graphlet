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
//! [`catalog`] provides named-motif queries (seeded with the diamond). The census /
//! catalog arm threads [`Induced`](catalog::Induced): induced counts are read
//! directly off the census; non-induced (monomorphism) counts are derived from the
//! induced census via a fixed `s(P,C)` table — no separate monomorphism enumerator.
//!
//! # Template matching
//!
//! [`template`] is a thin parallel arm delegating to petgraph's VF2
//! `subgraph_isomorphisms_iter`, which is **node-induced native**. Non-induced
//! matching of an arbitrary template is deliberately deferred (see [`template`] and
//! ADR-0290) — it has no grounding beneficiary today and cannot reuse the k-bounded
//! `s(P,C)` trick.
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
