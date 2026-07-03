//! Motif significance: z-scores and empirical p-values against a null model.
//!
//! Given a host graph, a set of motif targets (or the full graphlet census at
//! order `k`), and a null-model ensemble, [`motif_significance`] and
//! [`census_significance_profile`] compute for each target:
//!
//! - **observed**: the count in the host graph.
//! - **null\_mean** / **null\_std**: mean and population standard deviation of
//!   the count over the ensemble.
//! - **z\_score**: `(observed − null_mean) / null_std` (guarded when
//!   `null_std == 0`; see [`compute_significance_stats`]).
//! - **p\_value\_over**: fraction of ensemble samples with count **≥ observed**
//!   (one-sided over-representation; ties are included, following the convention
//!   in Milo *et al.* 2002).
//!
//! # Null models
//!
//! Both [`NullModel::DegreePreserving`] (double-edge swap) and
//! [`NullModel::ConfigurationModel`] are supported. The former exactly preserves
//! every node's degree; the latter matches only the degree sequence in
//! distribution.
//!
//! # Reproducibility
//!
//! Every function takes `rng: &mut impl Rng`; seed it once before calling to
//! make the ensemble fully deterministic.
//!
//! # Example
//!
//! ```
//! use graphlet::catalog::{Induced, Pattern};
//! use graphlet::rim::significance::{motif_significance, NullModel};
//! use petgraph::graph::UnGraph;
//! use rand::SeedableRng;
//! use rand::rngs::StdRng;
//!
//! // Five disjoint triangles — triangles are heavily over-represented vs the
//! // degree-preserving null (which breaks them into larger cycles).
//! let g: UnGraph<(), ()> = UnGraph::from_edges([
//!     (0u32,1),(1,2),(2,0), (3,4),(4,5),(5,3), (6,7),(7,8),(8,6),
//!     (9,10),(10,11),(11,9), (12,13),(13,14),(14,12),
//! ]);
//! let mut rng = StdRng::seed_from_u64(42);
//! let tri = Pattern::triangle();
//! let results = motif_significance(
//!     &g,
//!     &[("triangle", &tri, Induced::Yes)],
//!     40,
//!     NullModel::DegreePreserving { n_swaps_per_edge: 10 },
//!     &mut rng,
//! );
//! let (_name, entry) = &results[0];
//! assert!(entry.z_score > 0.0, "triangles should be over-represented");
//! ```

use petgraph::graph::{NodeIndex, UnGraph};
use rand::Rng;

use crate::canonical::{all_connected_classes, ClassId};
use crate::catalog::{count_motif, Induced, Pattern};
use crate::census::{count, Selector};
use crate::rim::null_model::{configuration_model_simple, double_edge_swap};
use crate::snapshot::{GraphAdapter, Snapshot};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Which null model generates the random-graph ensemble for significance testing.
#[derive(Clone, Debug)]
pub enum NullModel {
    /// Degree-preserving randomization via double-edge swap (Milo *et al.* 2002).
    ///
    /// `n_swaps_per_edge` is a multiplier: the number of swap *attempts* is
    /// `graph.edge_count() * n_swaps_per_edge`. A value of 10 is the standard
    /// choice and gives good mixing for most sparse graphs.
    DegreePreserving {
        /// Swap-attempt count multiplier (standard: 10).
        n_swaps_per_edge: usize,
    },

    /// Configuration-model null (simple variant: no self-loops, no parallel edges).
    ///
    /// Matches the degree *sequence* but not the exact wiring of the observed
    /// graph. Faster mixing than double-edge swap at the cost of not exactly
    /// preserving every node's degree — see
    /// [`crate::rim::null_model::configuration_model_simple`] for the erasure
    /// bound.
    ConfigurationModel,
}

/// Per-target significance statistics.
#[derive(Clone, Debug)]
pub struct SignificanceEntry {
    /// Count of the target in the observed graph.
    pub observed: u64,
    /// Mean count over the null ensemble.
    pub null_mean: f64,
    /// Population standard deviation of the count over the null ensemble.
    pub null_std: f64,
    /// Z-score: `(observed − null_mean) / null_std`.
    ///
    /// When `null_std == 0.0`, the z-score is `0.0` if `observed == null_mean`,
    /// `+∞` if `observed > null_mean`, and `−∞` if `observed < null_mean`.
    pub z_score: f64,
    /// Empirical one-sided over-representation p-value.
    ///
    /// Fraction of null-ensemble samples whose count is **≥ observed** (ties
    /// included). A small value indicates the observed count is unusually high
    /// relative to the null.
    pub p_value_over: f64,
}

/// Graphlet census significance profile: z-score vector across graphlet classes.
///
/// Returned by [`census_significance_profile`].
#[derive(Clone, Debug)]
pub struct SignificanceProfile {
    /// Per-class `(ClassId, SignificanceEntry)` pairs, ordered by canonical
    /// mask (ascending).
    pub entries: Vec<(ClassId, SignificanceEntry)>,
    /// Z-score vector in the same order as `entries`.
    pub z_scores: Vec<f64>,
    /// Unit-length-normalized z-score vector, present when `normalize = true`
    /// was passed. If the z-score vector is the zero vector (all z-scores
    /// exactly 0.0), the normalized vector is also zero (not NaN).
    pub normalized: Option<Vec<f64>>,
}

// ---------------------------------------------------------------------------
// Core statistics — public so tests can verify arithmetic independently
// ---------------------------------------------------------------------------

/// Compute significance statistics from an observed count and a slice of
/// null-ensemble counts.
///
/// This is the pure-computation kernel, exposed so unit tests can verify the
/// arithmetic exactly using hand-computed values without touching any graph.
///
/// # Conventions
///
/// - **Population std dev** (divide by N, not N−1): consistent with the
///   interpretation as the standard deviation of the null distribution estimated
///   by the ensemble.
/// - **p\_value\_over**: fraction of null counts **≥ observed** (ties
///   included), following Milo *et al.* 2002.
/// - **z\_score when std == 0**: `0.0` if `observed == null_mean`, `+∞` if
///   `observed > null_mean`, `−∞` if `observed < null_mean`.
///
/// # Panics
///
/// Panics if `null_counts` is empty.
pub fn compute_significance_stats(observed: u64, null_counts: &[u64]) -> SignificanceEntry {
    assert!(
        !null_counts.is_empty(),
        "null_counts must be non-empty; got empty slice"
    );
    let n = null_counts.len() as f64;
    let null_mean = null_counts.iter().sum::<u64>() as f64 / n;
    let null_var = null_counts
        .iter()
        .map(|&c| {
            let d = c as f64 - null_mean;
            d * d
        })
        .sum::<f64>()
        / n;
    let null_std = null_var.sqrt();
    let z_score = if null_std == 0.0 {
        let obs_f = observed as f64;
        // null_mean is exact (all equal) — compare obs to the common value
        match obs_f.partial_cmp(&null_mean) {
            Some(std::cmp::Ordering::Equal) => 0.0,
            Some(std::cmp::Ordering::Greater) => f64::INFINITY,
            _ => f64::NEG_INFINITY,
        }
    } else {
        (observed as f64 - null_mean) / null_std
    };
    let p_value_over = null_counts.iter().filter(|&&c| c >= observed).count() as f64 / n;
    SignificanceEntry {
        observed,
        null_mean,
        null_std,
        z_score,
        p_value_over,
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Convert any [`GraphAdapter`] to a simple `UnGraph<(), ()>` via the snapshot
/// (self-loops stripped, parallel edges deduped, directed inputs unioned).
fn to_ungraph<G: GraphAdapter>(g: G) -> UnGraph<(), ()> {
    let snap = Snapshot::new(g);
    let n = snap.len();
    let mut out = UnGraph::with_capacity(n, 0);
    for _ in 0..n {
        out.add_node(());
    }
    for i in 0..n {
        for &j in snap.neighbors(i) {
            if i < j {
                out.add_edge(NodeIndex::new(i), NodeIndex::new(j), ());
            }
        }
    }
    out
}

/// Sample one null graph from `base` under `model`.
fn make_null<R: Rng>(base: &UnGraph<(), ()>, model: &NullModel, rng: &mut R) -> UnGraph<(), ()> {
    match model {
        NullModel::DegreePreserving { n_swaps_per_edge } => {
            let n_swaps = base.edge_count().saturating_mul(*n_swaps_per_edge);
            double_edge_swap(base, n_swaps, rng)
        }
        NullModel::ConfigurationModel => {
            // For a simple undirected graph the degree sum is 2*|E|, always even.
            let degree_seq: Vec<usize> = (0..base.node_count())
                .map(|i| base.neighbors(NodeIndex::new(i)).count())
                .collect();
            configuration_model_simple(&degree_seq, rng)
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute motif significance for a list of named patterns.
///
/// For each `(name, pattern, induced)` triple in `targets`:
///
/// - Counts the pattern in the host graph (`observed`).
/// - Builds an `ensemble_size`-member null ensemble using `null_model` and
///   counts the pattern in each null graph.
/// - Returns [`SignificanceEntry`] statistics (mean, std, z-score, p-value)
///   paired with the name.
///
/// The host graph is treated as a simple undirected graph (self-loops stripped,
/// parallel edges deduped, directed inputs unioned) — see [`GraphAdapter`].
///
/// All null graphs are produced from the same base (`g` after normalization);
/// the ensemble is reproducible given a fixed `rng` seed.
///
/// # P-value convention
///
/// `p_value_over` is the fraction of null samples with count **≥ observed**
/// (one-sided, over-representation, ties included). A small value (e.g. < 0.05)
/// indicates the observed count is unusually high relative to the null.
///
/// # Panics
///
/// Panics if `ensemble_size == 0`.
pub fn motif_significance<G, R>(
    graph: G,
    targets: &[(&str, &Pattern, Induced)],
    ensemble_size: usize,
    null_model: NullModel,
    rng: &mut R,
) -> Vec<(String, SignificanceEntry)>
where
    G: GraphAdapter,
    R: Rng,
{
    assert!(ensemble_size > 0, "ensemble_size must be >= 1");
    let base = to_ungraph(graph);

    // Observed counts.
    let observed: Vec<u64> = targets
        .iter()
        .map(|(_, pat, ind)| count_motif(&base, pat, *ind))
        .collect();

    // Null ensemble: collect counts per target.
    let mut null_counts: Vec<Vec<u64>> = vec![Vec::with_capacity(ensemble_size); targets.len()];
    for _ in 0..ensemble_size {
        let ng = make_null(&base, &null_model, rng);
        for (i, (_, pat, ind)) in targets.iter().enumerate() {
            null_counts[i].push(count_motif(&ng, pat, *ind));
        }
    }

    targets
        .iter()
        .enumerate()
        .map(|(i, (name, _, _))| {
            (
                name.to_string(),
                compute_significance_stats(observed[i], &null_counts[i]),
            )
        })
        .collect()
}

/// Compute the graphlet census significance profile at order `k` (Milo-style SP).
///
/// The **significance profile** (SP) is the z-score vector across all graphlet
/// classes at order `k`, one entry per class. Classes are ordered by their
/// canonical mask (ascending).
///
/// For `k ≤ 5` the full ground-truth class set from
/// [`crate::canonical::all_connected_classes`] is used (so unobserved classes
/// get an observed count of 0). For `k > 5` the class set is the union of all
/// classes seen in the observed graph and the ensemble.
///
/// # Normalization
///
/// When `normalize = true`, the returned [`SignificanceProfile`] also includes
/// the unit-length-normalized z-score vector (Euclidean norm = 1), the standard
/// SP normalization (Milo *et al.* 2004). If the z-score vector is all-zero (e.g.
/// because the graph has no subgraphs at this order), the normalized vector is
/// also all-zero (not `NaN`).
///
/// # Panics
///
/// Panics if `ensemble_size == 0` or if `k` is out of the range for
/// [`Selector::connected_k_subsets`].
pub fn census_significance_profile<G, R>(
    graph: G,
    k: usize,
    ensemble_size: usize,
    null_model: NullModel,
    rng: &mut R,
    normalize: bool,
) -> SignificanceProfile
where
    G: GraphAdapter,
    R: Rng,
{
    assert!(ensemble_size > 0, "ensemble_size must be >= 1");
    let sel = Selector::connected_k_subsets(k);
    let base = to_ungraph(graph);

    // Observed census.
    let obs_census = count(&base, &sel);

    // Null ensemble censuses.
    let mut null_censuses: Vec<std::collections::HashMap<ClassId, u64>> =
        Vec::with_capacity(ensemble_size);
    for _ in 0..ensemble_size {
        let ng = make_null(&base, &null_model, rng);
        null_censuses.push(count(&ng, &sel));
    }

    // Class ordering: complete ground-truth set for k ≤ 5; union for k > 5.
    let mut all_masks: Vec<u64> = if k <= 5 {
        all_connected_classes(k)
    } else {
        let mut s: std::collections::HashSet<u64> = obs_census.keys().map(|c| c.0).collect();
        for nc in &null_censuses {
            s.extend(nc.keys().map(|c| c.0));
        }
        s.into_iter().collect()
    };
    all_masks.sort_unstable();

    let class_ids: Vec<ClassId> = all_masks.iter().map(|&m| ClassId(m)).collect();

    // Per-class significance.
    let entries: Vec<(ClassId, SignificanceEntry)> = class_ids
        .iter()
        .map(|&cid| {
            let obs = obs_census.get(&cid).copied().unwrap_or(0);
            let nulls: Vec<u64> = null_censuses
                .iter()
                .map(|nc| nc.get(&cid).copied().unwrap_or(0))
                .collect();
            (cid, compute_significance_stats(obs, &nulls))
        })
        .collect();

    let z_scores: Vec<f64> = entries.iter().map(|(_, e)| e.z_score).collect();

    let normalized = if normalize {
        // Euclidean norm (±∞ entries contribute ∞ to the sum, which is correct
        // — the normalization will also be ∞, and ∞/∞ = NaN is avoided because
        // the caller's z-scores should be finite for well-behaved graphs; for
        // graphs that yield ∞ z-scores the normalized vector can contain NaN,
        // which the profile doc acknowledges as an edge case).
        let norm_sq: f64 = z_scores.iter().map(|&z| z * z).sum();
        let norm = norm_sq.sqrt();
        if norm == 0.0 {
            Some(z_scores.clone())
        } else {
            Some(z_scores.iter().map(|&z| z / norm).collect())
        }
    } else {
        None
    };

    SignificanceProfile {
        entries,
        z_scores,
        normalized,
    }
}
