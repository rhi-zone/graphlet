//! The rim: capabilities in scope for this subfield but not yet built.
//!
//! These modules are intentionally empty. Each names a genuine gap identified in
//! ADR-0290 and its adversarial rim verification; they are documented here so the
//! surface is discoverable and so no one re-derives whether they belong. Nothing in
//! the rim is stubbed with an erroring API — a missing capability is absent, not a
//! runtime landmine.

/// Null-model / random-graph generators for significance testing.
///
/// Implemented (ADR-0290, TRUE algorithmic voids): configuration model (raw and
/// simple), double-edge-swap degree-preserving rewiring, Watts–Strogatz, LFR
/// benchmark. All generators accept `&mut impl Rng` for seeded reproducibility.
pub mod null_model;

/// Structure-aware graph kernels.
///
/// Implemented (ADR-0290, phase 5): a graphlet kernel that reduces to the census
/// substrate's own class-count vectors (no reimplementation of graphlet counting),
/// a Weisfeiler–Lehman subtree kernel (label refinement, shared cross-graph
/// alphabet), a shortest-path kernel (BFS distance histograms), and a generic
/// Gram-matrix builder with optional cosine normalization. See
/// [`kernels::graphlet_kernel`], [`kernels::wl_kernel`],
/// [`kernels::shortest_path_kernel`], and [`kernels::gram_matrix`].
pub mod kernels;

/// Motif significance: z-scores and empirical p-values against a null model.
///
/// Compares an observed census / motif count against an ensemble from
/// [`null_model`] to score over- and under-representation per target class
/// (ADR-0290). See [`significance::motif_significance`] and
/// [`significance::census_significance_profile`].
pub mod significance;

/// Neighborhood statistics — a sibling module *outside* the census substrate.
///
/// Implements link-prediction indices (common neighbors, Jaccard, Adamic-Adar,
/// resource allocation, preferential attachment), degree assortativity
/// (Newman/Pearson), rich-club coefficient φ(k), and local/average/global
/// clustering coefficients with triangle counting. (ADR-0290, cohesion re-homing;
/// live threat: if `franken_networkx` internals ship over petgraph types, the
/// cohesion case for re-homing these weakens.)
pub mod neighborhood;

/// Scalable graphlet counting beyond naive per-subset canonicalization.
///
/// Implemented (ADR-0290, phase 6, partial close): an ORCA-style (Hočevar & Demšar
/// 2014) fast per-node orbit counter for graphlets of order `2..=4` (orbits `0..=14`)
/// — count degree, per-edge triangle, and per-vertex K4 quantities once, then solve a
/// small per-vertex linear system, instead of enumerating every connected 4-subset.
/// Verified exact against the census substrate's own `graphlet_degree_vectors` /
/// `count`, node-for-node and count-for-count, across a large battery plus a
/// proptest. See [`scalable::fast_graphlet_degree_vectors`] and
/// [`scalable::fast_count`].
///
/// **Still open (not this phase):** the full order-5 system (58 more orbits) is a
/// much larger equation set; completing it faithfully and verifiably was out of scope
/// here, so k = 5 fast counting is not yet implemented — use the exact
/// `graphlet_degree_vectors` / `count` for k = 5 in the meantime. See the module docs
/// for the exact boundary.
pub mod scalable;

/// Directed motif analysis: triad census (k=3, all 16 types) and directed graphlet
/// census / orbits (weakly-connected, k in `2..=4`).
///
/// Implemented (ADR-0290, phase 7): the standard 16-type Holland–Leinhardt directed
/// triad census, and a directed generalization of the census/orbit substrate — ordered
/// (directed) canonical labelling, weak-connectivity-restricted enumeration, and a
/// directed-automorphism orbit registry — at orders `2..=4`. Verified exact against an
/// independent brute-force directed oracle (triad classification decorrelated from the
/// canonical-mask machinery; k=4 classes/orbits decorrelated from the ESU-driven
/// production path) on adversarial (cycles, DAGs, tournaments, bidirectional-edge
/// graphs) and fuzzed random digraphs, plus a proptest.
///
/// **Still open (not this phase):** directed k = 5 (orbits and classes) is not
/// implemented — see [`directed`]'s module docs for the exact boundary and the
/// undirected fallback.
pub mod directed;
