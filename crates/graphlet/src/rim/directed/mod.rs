//! Directed motif analysis: the directed triad census (k=3, all 16 standard types)
//! and the directed graphlet census / per-node orbits (weakly-connected, k in `2..=5`).
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
//!   convention), at orders `2..=5` ([`census::MAX_K`]).
//! - [`orbit::directed_graphlet_degree_vectors`] — per-node directed graphlet-degree
//!   vectors over the same `2..=5` order range, via a directed-automorphism
//!   [`orbit::DirectedRegistry`].
//!
//! **Performance caveat (k = 5).** The `k <= 4` substrate is cheap; `k = 5` is exact but
//! slow, on two axes:
//! - Per-instance canonicalization exhausts `5! = 120` permutations of a 20-bit
//!   ordered-pair mask (`crates/graphlet/src/rim/directed/canonical.rs`), instead of
//!   `4! = 24` — an ESU pass at `k = 5` over a graph with many weakly-connected 5-subsets
//!   costs noticeably more per instance than at `k = 4`.
//! - Building a [`orbit::DirectedRegistry`] at `k = 5` sweeps all `2^20` labelled
//!   digraphs once (internally, to enumerate the 9364 weakly-connected classes — OEIS
//!   A003085). That sweep is seconds, not microseconds — fine to pay once per registry
//!   build, not something to do in a hot loop.
//!
//! Nothing here approximates k = 5: exactness is unconditional, verified against an
//! independent brute-force directed oracle (see the crate's test suite) exactly as at
//! `k <= 4`. `k = 6` is out of scope (`k(k-1) = 30`-bit masks still fit `u64`, but the
//! `2^30`-mask registry-build sweep is no longer cheap enough to pay routinely).

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
