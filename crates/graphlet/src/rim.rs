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
/// TODO (ADR-0290, OPEN gate): naive canonicalization is untenable at biological
/// scale for k = 5. Needs ORCA orbit-count equations or a g-trie, plus
/// ORCA-permutation alignment of the internal orbit ids.
pub mod scalable {}

/// Directed motif analysis at k ≥ 4.
///
/// TODO (ADR-0290, deferred by design): the census core is undirected at k ≥ 4 (k=3
/// triads are settled). Directed graphlet classes/orbits at k ≥ 4 are future work.
pub mod directed {}
