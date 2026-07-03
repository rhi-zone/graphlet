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
