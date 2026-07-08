//! Structure-aware graph kernels: graphlet, Weisfeiler–Lehman subtree, and
//! shortest-path, plus a generic Gram-matrix builder.
//!
//! Every kernel here follows the same shape: an explicit **feature extractor**
//! (`*_features` / `*_histogram` / `wl_refine`) turns a graph into a concrete finite
//! vector (a `HashMap` from some discrete key to a `u64` count), and the **kernel**
//! is the inner product of two such vectors. Because every kernel is literally
//! `⟨φ(G), φ(H)⟩` for an explicit finite `φ`, the Gram matrix over any set of
//! graphs is guaranteed positive semi-definite by elementary linear algebra
//! (`Φᵀ Φ` for a real matrix `Φ`) — the test suite checks this by Cholesky
//! decomposition as the defining sanity check on the wiring, not a formality.
//!
//! The graphlet kernel's feature vector is the census substrate's own
//! [`Census`](crate::Census) output — this module does not reimplement graphlet
//! counting, it only reads it off [`count`](crate::count).
//!
//! # Example
//!
//! ```
//! use graphlet::rim::kernels::{graphlet_features, graphlet_kernel, graphlet_kernel_cosine};
//! use petgraph::graph::UnGraph;
//!
//! // A triangle and a path of 3 nodes: both order-3 connected graphlets, but of
//! // different classes, so they share no graphlet-class counts at k=3.
//! let triangle: UnGraph<(), ()> = UnGraph::from_edges([(0, 1), (1, 2), (2, 0)]);
//! let path: UnGraph<(), ()> = UnGraph::from_edges([(0, 1), (1, 2)]);
//!
//! let ft = graphlet_features(&triangle, 3);
//! let fp = graphlet_features(&path, 3);
//! assert_eq!(graphlet_kernel(&ft, &fp), 0);
//! assert_eq!(graphlet_kernel_cosine(&ft, &fp), 0.0);
//!
//! // A graph is maximally self-similar under the cosine-normalized kernel.
//! assert!((graphlet_kernel_cosine(&ft, &ft) - 1.0).abs() < 1e-12);
//!
//! // Comparing features built at different orders `k` is a programmer error —
//! // `graphlet_kernel` panics rather than silently returning a bogus number
//! // (a triangle at k=3 and a 5-star at k=4 share the same bare `ClassId` mask).
//! let fp_k2 = graphlet_features(&path, 2);
//! let result = std::panic::catch_unwind(|| graphlet_kernel(&ft, &fp_k2));
//! assert!(result.is_err());
//! ```

use std::collections::{HashMap, VecDeque};

use crate::census::{count, Census, Selector};
use crate::snapshot::{GraphAdapter, Snapshot};

// ---------------------------------------------------------------------------
// Shared inner-product helper
// ---------------------------------------------------------------------------

/// Inner product of two sparse count vectors, keyed by any hashable discriminator.
/// Missing keys contribute `0` (both maps are treated as vectors over the union of
/// their keys, implicitly zero-padded).
fn inner_product<K: Eq + std::hash::Hash>(a: &HashMap<K, u64>, b: &HashMap<K, u64>) -> u64 {
    a.iter()
        .map(|(k, &v)| v * b.get(k).copied().unwrap_or(0))
        .sum()
}

// ---------------------------------------------------------------------------
// Graphlet kernel
// ---------------------------------------------------------------------------

/// The graphlet-class count vector of `g` at order `k`, tagged with the order `k`
/// it was built at.
///
/// [`ClassId`](crate::canonical::ClassId) is a bare adjacency bitmask with no order
/// tag of its own — a triangle at `k=3` and a 5-star at `k=4` encode the same mask
/// — so a [`Census`] alone cannot detect a cross-order comparison. Carrying `k`
/// alongside the census lets [`graphlet_kernel`] catch that case instead of
/// silently returning a bogus inner product.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphletFeatures {
    /// The order `k` these features were computed at.
    pub k: usize,
    /// The graphlet-class count vector itself, read directly off the crate's
    /// census substrate.
    pub census: Census,
}

/// Compute [`GraphletFeatures`] for `g` at order `k` — the feature representation
/// for [`graphlet_kernel`].
///
/// The `census` field is exactly `count(g, &Selector::connected_k_subsets(k))`:
/// this does not reimplement graphlet counting, it reuses the verified census
/// fold, tagged with `k` for [`graphlet_kernel`]'s cross-order guard.
///
/// # Panics
///
/// Panics unless `2 <= k <= MAX_K` (see [`Selector::connected_k_subsets`]).
#[must_use]
pub fn graphlet_features<G: GraphAdapter>(g: G, k: usize) -> GraphletFeatures {
    GraphletFeatures {
        k,
        census: count(g, &Selector::connected_k_subsets(k)),
    }
}

/// Graphlet kernel: the inner product of two graphlet-class count vectors, aligned
/// by class id.
///
/// `a` and `b` must be [`graphlet_features`] outputs at the *same* order `k` —
/// counts at mismatched orders are not comparable (the same [`ClassId`](crate::canonical::ClassId)
/// mask can denote different classes at different orders), and this function
/// panics rather than silently returning a meaningless value.
///
/// # Panics
///
/// Panics if `a.k != b.k`.
#[must_use]
pub fn graphlet_kernel(a: &GraphletFeatures, b: &GraphletFeatures) -> u64 {
    assert_eq!(
        a.k, b.k,
        "graphlet_kernel: mismatched orders (k={} vs k={}) — features must come from \
         graphlet_features calls at the same k, since ClassId does not carry an order \
         tag and a cross-k comparison would silently produce a bogus value",
        a.k, b.k
    );
    inner_product(&a.census, &b.census)
}

/// Cosine-normalized graphlet kernel: `k(G,H) / sqrt(k(G,G) * k(H,H))`, in `[0, 1]`.
///
/// Returns `0.0` if either graph's self-kernel is `0` (i.e. it has no connected
/// k-subgraphs at all, so its feature vector is the zero vector).
///
/// # Panics
///
/// Panics if `a.k != b.k` (see [`graphlet_kernel`]).
#[must_use]
pub fn graphlet_kernel_cosine(a: &GraphletFeatures, b: &GraphletFeatures) -> f64 {
    let kab = graphlet_kernel(a, b) as f64;
    let kaa = graphlet_kernel(a, a) as f64;
    let kbb = graphlet_kernel(b, b) as f64;
    let denom = (kaa * kbb).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        kab / denom
    }
}

// ---------------------------------------------------------------------------
// Weisfeiler-Lehman subtree kernel
// ---------------------------------------------------------------------------

/// A per-node label assignment, in a graph's own [`Snapshot`] compact-index order
/// (`labeling[i]` is compact node `i`'s label).
pub type Labeling = Vec<u64>;

/// Default initial WL labeling: each node's degree.
#[must_use]
pub fn degree_labeling<G: GraphAdapter>(g: G) -> Labeling {
    let snap = Snapshot::new(g);
    (0..snap.len())
        .map(|i| snap.neighbors(i).len() as u64)
        .collect()
}

/// Run `iterations` rounds of Weisfeiler–Lehman label refinement across a *batch*
/// of graphs sharing one compact signature alphabet, so their per-iteration labels
/// line up (the standard WL subtree kernel construction — Shervashidze et al.
/// 2011).
///
/// `initial[i]` is graph `i`'s starting labeling (e.g. [`degree_labeling`]); the
/// two must have matching lengths and each `initial[i]` must have one entry per
/// node of `graphs[i]` in `Snapshot` compact-index order.
///
/// Returns, for each graph, `iterations + 1` labelings: index `0` is `initial[i]`
/// unchanged, index `t` (`1 <= t <= iterations`) is the refined labeling after `t`
/// rounds. At each round, every node's new label is a compact id for the signature
/// `(old_label(v), sorted_multiset(old_label(u) for u in neighbors(v)))`; the id
/// dictionary is shared across all graphs in the batch (and reset per round), so
/// two nodes — in the same or different graphs — receive the same new label iff
/// they have the same signature.
///
/// # Panics
///
/// Panics if `graphs.len() != initial.len()`, or if some `initial[i].len()` does
/// not match `graphs[i]`'s node count.
#[must_use]
pub fn wl_refine<G: GraphAdapter>(
    graphs: &[G],
    initial: &[Labeling],
    iterations: usize,
) -> Vec<Vec<Labeling>> {
    assert_eq!(
        graphs.len(),
        initial.len(),
        "wl_refine: graphs/initial length mismatch"
    );
    let snaps: Vec<Snapshot<G::NodeId>> = graphs.iter().map(|&g| Snapshot::new(g)).collect();
    for (i, snap) in snaps.iter().enumerate() {
        assert_eq!(
            snap.len(),
            initial[i].len(),
            "wl_refine: initial labeling {i} has {} entries, graph has {} nodes",
            initial[i].len(),
            snap.len()
        );
    }

    let mut history: Vec<Vec<Labeling>> = initial.iter().map(|l| vec![l.clone()]).collect();
    let mut current: Vec<Labeling> = initial.to_vec();

    for _ in 0..iterations {
        // Signature -> compact new-label id, shared across the whole batch and
        // rebuilt fresh each round (cross-round id collisions are harmless: every
        // downstream use compares histograms within one round only).
        let mut sigs: HashMap<(u64, Vec<u64>), u64> = HashMap::new();
        let mut next_id = 0u64;
        let mut next: Vec<Labeling> = Vec::with_capacity(snaps.len());
        for (gi, snap) in snaps.iter().enumerate() {
            let mut labels = Vec::with_capacity(snap.len());
            for v in 0..snap.len() {
                let mut nbr_labels: Vec<u64> =
                    snap.neighbors(v).iter().map(|&u| current[gi][u]).collect();
                nbr_labels.sort_unstable();
                let sig = (current[gi][v], nbr_labels);
                let id = *sigs.entry(sig).or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                });
                labels.push(id);
            }
            next.push(labels);
        }
        current = next;
        for (gi, labels) in current.iter().enumerate() {
            history[gi].push(labels.clone());
        }
    }
    history
}

/// Histogram a labeling: label id -> node count.
#[must_use]
pub fn label_histogram(labels: &Labeling) -> HashMap<u64, u64> {
    let mut h = HashMap::new();
    for &l in labels {
        *h.entry(l).or_insert(0) += 1;
    }
    h
}

/// Weisfeiler–Lehman subtree kernel: the sum, over every iteration in `a`/`b`
/// (indices must line up — same `iterations` count, produced by one [`wl_refine`]
/// batch call so ids share an alphabet), of the inner product of that iteration's
/// label histogram.
///
/// # Panics
///
/// Panics if `a.len() != b.len()` (mismatched iteration counts).
#[must_use]
pub fn wl_kernel(a: &[Labeling], b: &[Labeling]) -> u64 {
    assert_eq!(
        a.len(),
        b.len(),
        "wl_kernel: mismatched iteration counts ({} vs {})",
        a.len(),
        b.len()
    );
    a.iter()
        .zip(b.iter())
        .map(|(la, lb)| inner_product(&label_histogram(la), &label_histogram(lb)))
        .sum()
}

/// Convenience: the WL subtree kernel between exactly two graphs, with
/// degree-based initial labels and `iterations` refinement rounds.
#[must_use]
pub fn wl_kernel_pair<G: GraphAdapter>(g: G, other: G, iterations: usize) -> u64 {
    let initial = vec![degree_labeling(g), degree_labeling(other)];
    let history = wl_refine(&[g, other], &initial, iterations);
    wl_kernel(&history[0], &history[1])
}

// ---------------------------------------------------------------------------
// Shortest-path kernel
// ---------------------------------------------------------------------------

/// Unweighted BFS distances from `s` to every node, in compact-index order;
/// `None` for unreachable nodes.
fn bfs_distances<N: Copy>(snap: &Snapshot<N>, s: usize) -> Vec<Option<usize>> {
    let mut dist = vec![None; snap.len()];
    dist[s] = Some(0);
    let mut queue = VecDeque::new();
    queue.push_back(s);
    while let Some(u) = queue.pop_front() {
        let du = dist[u].expect("queued node always has a distance");
        for &v in snap.neighbors(u) {
            if dist[v].is_none() {
                dist[v] = Some(du + 1);
                queue.push_back(v);
            }
        }
    }
    dist
}

/// Histogram of pairwise shortest-path distances (unweighted, via BFS), keyed by
/// distance: the feature representation for [`shortest_path_kernel`].
///
/// Each unordered pair of distinct reachable nodes contributes one observation
/// (each pair counted once, not twice). Pairs in different connected components
/// are omitted — the standard shortest-path-kernel convention (Borgwardt & Kriegel
/// 2005) — so a disconnected graph's histogram simply undercounts relative to
/// `C(n, 2)`.
#[must_use]
pub fn shortest_path_histogram<G: GraphAdapter>(g: G) -> HashMap<usize, u64> {
    let snap = Snapshot::new(g);
    let n = snap.len();
    let mut hist = HashMap::new();
    for s in 0..n {
        let dist = bfs_distances(&snap, s);
        for d in dist.iter().skip(s + 1).flatten() {
            *hist.entry(*d).or_insert(0) += 1;
        }
    }
    hist
}

/// Shortest-path kernel: the inner product of two shortest-path distance
/// histograms.
#[must_use]
pub fn shortest_path_kernel(a: &HashMap<usize, u64>, b: &HashMap<usize, u64>) -> u64 {
    inner_product(a, b)
}

// ---------------------------------------------------------------------------
// Gram matrix
// ---------------------------------------------------------------------------

/// Normalization mode for [`gram_matrix`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GramNormalization {
    /// Raw kernel values, unnormalized.
    Raw,
    /// Cosine-normalize: entry `(i,j)` divided by `sqrt(k(i,i) * k(j,j))`, giving a
    /// unit diagonal. An entry is `0.0` if either diagonal value is `0.0`.
    Cosine,
}

/// Compute the full `n x n` Gram (kernel) matrix over `items`, given any symmetric
/// kernel function `kernel`.
///
/// Generic over the feature representation `T` (a [`GraphletFeatures`], a WL
/// iteration history `Vec<Labeling>`, a shortest-path histogram, or anything else); only
/// `n(n+1)/2` kernel evaluations are performed (the matrix is symmetric by
/// construction, mirrored rather than recomputed).
///
/// With [`GramNormalization::Cosine`], the returned matrix has a unit diagonal
/// (except where a self-kernel is `0.0`, which stays `0.0` throughout its row and
/// column).
///
/// # Example
///
/// ```
/// use graphlet::rim::kernels::{gram_matrix, graphlet_features, graphlet_kernel, GramNormalization};
/// use petgraph::graph::UnGraph;
///
/// let triangle: UnGraph<(), ()> = UnGraph::from_edges([(0, 1), (1, 2), (2, 0)]);
/// let path: UnGraph<(), ()> = UnGraph::from_edges([(0, 1), (1, 2)]);
/// let feats = vec![graphlet_features(&triangle, 3), graphlet_features(&path, 3)];
///
/// let m = gram_matrix(&feats, |a, b| graphlet_kernel(a, b) as f64, GramNormalization::Cosine);
/// assert!((m[0][0] - 1.0).abs() < 1e-12);
/// assert!((m[1][1] - 1.0).abs() < 1e-12);
/// assert_eq!(m[0][1], 0.0); // triangle and path share no k=3 graphlet class
/// ```
#[must_use]
pub fn gram_matrix<T>(
    items: &[T],
    kernel: impl Fn(&T, &T) -> f64,
    normalization: GramNormalization,
) -> Vec<Vec<f64>> {
    let n = items.len();
    let mut raw = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in i..n {
            let k = kernel(&items[i], &items[j]);
            raw[i][j] = k;
            raw[j][i] = k;
        }
    }
    match normalization {
        GramNormalization::Raw => raw,
        GramNormalization::Cosine => {
            let diag: Vec<f64> = (0..n).map(|i| raw[i][i]).collect();
            let mut out = vec![vec![0.0; n]; n];
            for i in 0..n {
                for j in 0..n {
                    let denom = (diag[i] * diag[j]).sqrt();
                    out[i][j] = if denom == 0.0 { 0.0 } else { raw[i][j] / denom };
                }
            }
            out
        }
    }
}

/// Convenience: the graphlet-kernel Gram matrix over a batch of graphs at order
/// `k`.
#[must_use]
pub fn graphlet_gram_matrix<G: GraphAdapter>(
    graphs: &[G],
    k: usize,
    normalization: GramNormalization,
) -> Vec<Vec<f64>> {
    let feats: Vec<GraphletFeatures> = graphs.iter().map(|&g| graphlet_features(g, k)).collect();
    gram_matrix(&feats, |a, b| graphlet_kernel(a, b) as f64, normalization)
}

/// Convenience: the WL subtree-kernel Gram matrix over a batch of graphs, with
/// degree-based initial labels and `iterations` refinement rounds.
#[must_use]
pub fn wl_gram_matrix<G: GraphAdapter>(
    graphs: &[G],
    iterations: usize,
    normalization: GramNormalization,
) -> Vec<Vec<f64>> {
    let initial: Vec<Labeling> = graphs.iter().map(|&g| degree_labeling(g)).collect();
    let history = wl_refine(graphs, &initial, iterations);
    gram_matrix(&history, |a, b| wl_kernel(a, b) as f64, normalization)
}

/// Convenience: the shortest-path-kernel Gram matrix over a batch of graphs.
#[must_use]
pub fn shortest_path_gram_matrix<G: GraphAdapter>(
    graphs: &[G],
    normalization: GramNormalization,
) -> Vec<Vec<f64>> {
    let feats: Vec<HashMap<usize, u64>> =
        graphs.iter().map(|&g| shortest_path_histogram(g)).collect();
    gram_matrix(
        &feats,
        |a, b| shortest_path_kernel(a, b) as f64,
        normalization,
    )
}
