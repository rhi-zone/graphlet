//! Directed motif analysis: the directed triad census (k=3, all 16 standard types)
//! and the directed graphlet census / per-node orbits (weakly-connected, k in `2..=4`).
//!
//! The undirected census substrate ([`crate::census`], [`crate::orbit`]) analyzes every
//! graph — directed or not — on its *underlying undirected structure* (in/out
//! neighborhoods unioned). This module instead respects arc direction throughout:
//! canonical labelling packs an ordered-pair (`k(k-1)`-bit) arc mask instead of an
//! unordered upper-triangle mask, and the automorphism/orbit machinery only accepts
//! direction-preserving permutations. Two directed graphlets that fold to the same
//! undirected class (say, `a -> b -> c` and `a -> b <- c`, both a path undirected) are
//! *distinct* directed classes.
//!
//! - [`triad::triad_census`] — the standard 16-type directed triad census (Holland &
//!   Leinhardt), over **every** 3-subset (connected or not). This is the well-known
//!   checkpoint: totals sum to `C(n, 3)`, and the 16-way partition is exhaustively
//!   verified (all `2^6` possible directed triads fall into exactly 16 isomorphism
//!   classes).
//! - [`census::count_directed`] / [`census::enumerate_directed`] — the directed
//!   graphlet census proper, restricted to **weakly-connected** subsets (the graphlet
//!   convention), at orders `2..=4` ([`census::MAX_K`]).
//! - [`orbit::directed_graphlet_degree_vectors`] — per-node directed graphlet-degree
//!   vectors over the same `2..=4` order range, via a directed-automorphism
//!   [`orbit::DirectedRegistry`].
//!
//! **Boundary (not this phase): directed k = 5.** The k <= 4 substrate above is exact
//! and verified against an independent brute-force directed oracle (see the crate's
//! test suite). Extending to k = 5 needs the same generalization one order further —
//! `k(k-1) = 20`-bit masks (still fits `u64`) and a `2^20`-mask registry-build sweep
//! (still cheap) — but completing *and* independently re-verifying it was out of scope
//! this phase. Nothing here approximates k = 5; it is simply absent. Use the
//! undirected [`crate::graphlet_degree_vectors`] / [`crate::count`] at k = 5 (on the
//! underlying undirected structure) in the meantime.

mod canonical;
mod census;
mod orbit;
pub mod triad;

pub use canonical::DirectedClassId;
pub use census::{
    count_directed, enumerate_directed, DirectedCensus, DirectedInstance, DirectedInstances,
    DirectedSelector, MAX_K,
};
pub use orbit::{directed_graphlet_degree_vectors, DirectedGdvTable, DirectedRegistry};
pub use snapshot::{DirectedGraphAdapter, DirectedSnapshot};

mod snapshot;
