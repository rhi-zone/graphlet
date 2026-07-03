//! Neighborhood statistics: link prediction, clustering, degree assortativity,
//! and rich-club coefficient.
//!
//! All functions treat the input as a *simple undirected* graph — self-loops are
//! stripped and parallel edges deduped — consistent with the rest of the crate.
//!
//! # Example
//!
//! ```
//! use graphlet::rim::neighborhood::{
//!     common_neighbors, jaccard, local_clustering, global_clustering,
//!     total_triangles, degree_assortativity, rich_club,
//! };
//! use petgraph::graph::{NodeIndex, UnGraph};
//!
//! // A triangle (K₃).
//! let g: UnGraph<(), ()> = UnGraph::from_edges([(0u32, 1), (1, 2), (2, 0)]);
//! let n0 = NodeIndex::new(0);
//! let n1 = NodeIndex::new(1);
//! assert_eq!(common_neighbors(&g, n0, n1), 1);
//! assert!((local_clustering(&g, n0) - 1.0).abs() < 1e-10);
//! assert_eq!(total_triangles(&g), 1);
//! assert!((global_clustering(&g) - 1.0).abs() < 1e-10);
//! ```

use std::collections::HashSet;

use crate::catalog::Pattern;
use crate::census::{count, Selector};
use crate::snapshot::{GraphAdapter, Snapshot};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Build a raw-slot → compact-index reverse mapping.
///
/// The Snapshot constructor compacts node IDs to `0..n`; this re-derives the
/// same mapping so we can translate host `NodeId`s back to compact indices
/// without exposing private Snapshot internals.
/// Holes (removed `StableGraph` nodes) map to `usize::MAX`.
fn compact_map<G: GraphAdapter>(g: G) -> Vec<usize> {
    let bound = g.node_bound();
    let mut compact = vec![usize::MAX; bound];
    for (idx, v) in g.node_identifiers().enumerate() {
        compact[g.to_index(v)] = idx;
    }
    compact
}

// ---------------------------------------------------------------------------
// Link-prediction index types
// ---------------------------------------------------------------------------

/// All five link-prediction scores for a node pair.
#[derive(Clone, Debug, PartialEq)]
pub struct LinkPredictionScores {
    /// |N(u) ∩ N(v)|
    pub common_neighbors: usize,
    /// |N(u) ∩ N(v)| / |N(u) ∪ N(v)|; `0.0` if the union is empty
    pub jaccard: f64,
    /// Σ 1/ln(deg(w)) over common neighbors w with deg(w) ≥ 2
    pub adamic_adar: f64,
    /// Σ 1/deg(w) over common neighbors w with deg(w) ≥ 1
    pub resource_allocation: f64,
    /// deg(u) × deg(v)
    pub preferential_attachment: usize,
}

// ---------------------------------------------------------------------------
// Link prediction: per-pair functions
// ---------------------------------------------------------------------------

/// Count the common neighbors of `u` and `v`.
pub fn common_neighbors<G: GraphAdapter>(g: G, u: G::NodeId, v: G::NodeId) -> usize {
    let compact = compact_map(g);
    let snap = Snapshot::new(g);
    let ui = compact[g.to_index(u)];
    let vi = compact[g.to_index(v)];
    let nu: HashSet<usize> = snap.neighbors(ui).iter().copied().collect();
    snap.neighbors(vi)
        .iter()
        .filter(|&&w| nu.contains(&w))
        .count()
}

/// Jaccard similarity: |N(u) ∩ N(v)| / |N(u) ∪ N(v)|.
///
/// Returns `0.0` if both neighborhoods are empty.
pub fn jaccard<G: GraphAdapter>(g: G, u: G::NodeId, v: G::NodeId) -> f64 {
    let compact = compact_map(g);
    let snap = Snapshot::new(g);
    let ui = compact[g.to_index(u)];
    let vi = compact[g.to_index(v)];
    let nu: HashSet<usize> = snap.neighbors(ui).iter().copied().collect();
    let nv: HashSet<usize> = snap.neighbors(vi).iter().copied().collect();
    let inter = nu.intersection(&nv).count();
    let union = nu.union(&nv).count();
    if union == 0 {
        0.0
    } else {
        inter as f64 / union as f64
    }
}

/// Adamic-Adar index: Σ 1/ln(deg(w)) over common neighbors w.
///
/// Common neighbors w with deg(w) ≤ 1 are skipped (ln ≤ 0).
pub fn adamic_adar<G: GraphAdapter>(g: G, u: G::NodeId, v: G::NodeId) -> f64 {
    let compact = compact_map(g);
    let snap = Snapshot::new(g);
    let ui = compact[g.to_index(u)];
    let vi = compact[g.to_index(v)];
    let nu: HashSet<usize> = snap.neighbors(ui).iter().copied().collect();
    snap.neighbors(vi)
        .iter()
        .filter(|&&w| nu.contains(&w))
        .map(|&w| {
            let deg = snap.neighbors(w).len();
            if deg >= 2 {
                1.0 / (deg as f64).ln()
            } else {
                0.0
            }
        })
        .sum()
}

/// Resource allocation index: Σ 1/deg(w) over common neighbors w.
///
/// Common neighbors w with deg(w) = 0 are skipped (isolated nodes cannot
/// propagate resources — impossible in a simple connected component but
/// guarded for correctness).
pub fn resource_allocation<G: GraphAdapter>(g: G, u: G::NodeId, v: G::NodeId) -> f64 {
    let compact = compact_map(g);
    let snap = Snapshot::new(g);
    let ui = compact[g.to_index(u)];
    let vi = compact[g.to_index(v)];
    let nu: HashSet<usize> = snap.neighbors(ui).iter().copied().collect();
    snap.neighbors(vi)
        .iter()
        .filter(|&&w| nu.contains(&w))
        .map(|&w| {
            let deg = snap.neighbors(w).len();
            if deg >= 1 {
                1.0 / deg as f64
            } else {
                0.0
            }
        })
        .sum()
}

/// Preferential attachment score: deg(u) × deg(v).
pub fn preferential_attachment<G: GraphAdapter>(g: G, u: G::NodeId, v: G::NodeId) -> usize {
    let compact = compact_map(g);
    let snap = Snapshot::new(g);
    let ui = compact[g.to_index(u)];
    let vi = compact[g.to_index(v)];
    snap.neighbors(ui).len() * snap.neighbors(vi).len()
}

/// Score every non-edge `(u, v)` (u < v in compact index order) with all five
/// link-prediction indices.
///
/// Pairs where an edge already exists are skipped. This is `O(V² × max_deg)`
/// in the worst case — fine at the scale this crate targets.
pub fn score_non_edges<G: GraphAdapter>(g: G) -> Vec<(G::NodeId, G::NodeId, LinkPredictionScores)> {
    let snap = Snapshot::new(g);
    let n = snap.len();
    let mut out = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            if snap.adjacent(i, j) {
                continue;
            }
            let ni: HashSet<usize> = snap.neighbors(i).iter().copied().collect();
            let nj: HashSet<usize> = snap.neighbors(j).iter().copied().collect();
            let common_vec: Vec<usize> = ni.intersection(&nj).copied().collect();
            let common = common_vec.len();
            let union = ni.union(&nj).count();
            let jacc = if union == 0 {
                0.0
            } else {
                common as f64 / union as f64
            };
            let aa: f64 = common_vec
                .iter()
                .map(|&w| {
                    let deg = snap.neighbors(w).len();
                    if deg >= 2 {
                        1.0 / (deg as f64).ln()
                    } else {
                        0.0
                    }
                })
                .sum();
            let ra: f64 = common_vec
                .iter()
                .map(|&w| {
                    let deg = snap.neighbors(w).len();
                    if deg >= 1 {
                        1.0 / deg as f64
                    } else {
                        0.0
                    }
                })
                .sum();
            let pa = snap.neighbors(i).len() * snap.neighbors(j).len();
            out.push((
                snap.id(i),
                snap.id(j),
                LinkPredictionScores {
                    common_neighbors: common,
                    jaccard: jacc,
                    adamic_adar: aa,
                    resource_allocation: ra,
                    preferential_attachment: pa,
                },
            ));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Clustering coefficients and triangle counting
// ---------------------------------------------------------------------------

/// Count the triangles that pass through node `u`.
///
/// A triangle through `u` is a pair of distinct neighbors `(v, w)` with `v–w`
/// also an edge. Each such triangle is counted exactly once.
pub fn node_triangles<G: GraphAdapter>(g: G, u: G::NodeId) -> usize {
    let compact = compact_map(g);
    let snap = Snapshot::new(g);
    let ui = compact[g.to_index(u)];
    let nbrs = snap.neighbors(ui);
    let mut t = 0;
    for a in 0..nbrs.len() {
        for b in (a + 1)..nbrs.len() {
            if snap.adjacent(nbrs[a], nbrs[b]) {
                t += 1;
            }
        }
    }
    t
}

/// Total number of distinct triangles in `g`.
///
/// Equivalent to `Σ_u node_triangles(u) / 3` but computed in a single pass.
/// Cross-verifiable against the k=3 census triangle class — see the
/// `total_triangles_matches_census` test.
pub fn total_triangles<G: GraphAdapter>(g: G) -> usize {
    let snap = Snapshot::new(g);
    let n = snap.len();
    let mut sum = 0usize;
    for u in 0..n {
        let nbrs = snap.neighbors(u);
        for a in 0..nbrs.len() {
            for b in (a + 1)..nbrs.len() {
                if snap.adjacent(nbrs[a], nbrs[b]) {
                    sum += 1;
                }
            }
        }
    }
    // Each triangle is counted once per vertex, i.e. three times.
    sum / 3
}

/// Local clustering coefficient of `u`: fraction of neighbor pairs that are
/// connected to each other.
///
/// Defined as `2t / (k(k−1))` where `k = deg(u)` and `t` is the number of
/// edges among the neighbors of `u`.  Returns `0.0` if `deg(u) < 2`.
pub fn local_clustering<G: GraphAdapter>(g: G, u: G::NodeId) -> f64 {
    let compact = compact_map(g);
    let snap = Snapshot::new(g);
    let ui = compact[g.to_index(u)];
    let k = snap.neighbors(ui).len();
    if k < 2 {
        return 0.0;
    }
    let nbrs = snap.neighbors(ui);
    let mut t = 0usize;
    for a in 0..nbrs.len() {
        for b in (a + 1)..nbrs.len() {
            if snap.adjacent(nbrs[a], nbrs[b]) {
                t += 1;
            }
        }
    }
    2.0 * t as f64 / (k * (k - 1)) as f64
}

/// Average local clustering coefficient over all nodes.
///
/// Returns `0.0` for an empty graph.
pub fn average_clustering<G: GraphAdapter>(g: G) -> f64 {
    let snap = Snapshot::new(g);
    let n = snap.len();
    if n == 0 {
        return 0.0;
    }
    let sum: f64 = (0..n)
        .map(|u| {
            let k = snap.neighbors(u).len();
            if k < 2 {
                return 0.0;
            }
            let nbrs = snap.neighbors(u);
            let mut t = 0usize;
            for a in 0..nbrs.len() {
                for b in (a + 1)..nbrs.len() {
                    if snap.adjacent(nbrs[a], nbrs[b]) {
                        t += 1;
                    }
                }
            }
            2.0 * t as f64 / (k * (k - 1)) as f64
        })
        .sum();
    sum / n as f64
}

/// Global clustering coefficient (transitivity):
/// `3 × total_triangles / Σ_u C(deg_u, 2)`.
///
/// Returns `0.0` if no node has degree ≥ 2 (no closed triplets exist).
///
/// Derivation: the numerator `Σ_u node_triangles(u)` equals `3 × total_triangles`
/// (each triangle contributes 1 per vertex); the denominator counts closed
/// triplets (paths of length 2 centered at each vertex).  The ratio simplifies
/// to `(Σ_u node_triangles(u)) / (Σ_u C(deg_u, 2))`.
pub fn global_clustering<G: GraphAdapter>(g: G) -> f64 {
    let snap = Snapshot::new(g);
    let n = snap.len();
    let mut tri_sum = 0usize; // Σ_u node_triangles(u) = 3 × total_triangles
    let mut triplets = 0usize; // Σ_u C(deg_u, 2)
    for u in 0..n {
        let k = snap.neighbors(u).len();
        triplets += k.saturating_sub(1) * k / 2;
        let nbrs = snap.neighbors(u);
        for a in 0..nbrs.len() {
            for b in (a + 1)..nbrs.len() {
                if snap.adjacent(nbrs[a], nbrs[b]) {
                    tri_sum += 1;
                }
            }
        }
    }
    if triplets == 0 {
        return 0.0;
    }
    // tri_sum = 3 × total_triangles, triplets = Σ C(k,2)
    // transitivity = 3 × total_triangles / triplets = tri_sum / triplets
    tri_sum as f64 / triplets as f64
}

// ---------------------------------------------------------------------------
// Degree assortativity (Newman/Pearson)
// ---------------------------------------------------------------------------

/// Degree assortativity coefficient (Newman 2002).
///
/// Treats each undirected edge as one observation `(j_e, k_e)` (the degrees at
/// its two endpoints) and computes the Newman formula:
///
/// ```text
/// r = (M⁻¹ Σ jₑkₑ  −  [M⁻¹ Σ (jₑ+kₑ)/2]²)
///   / (M⁻¹ Σ (jₑ²+kₑ²)/2  −  [M⁻¹ Σ (jₑ+kₑ)/2]²)
/// ```
///
/// Returns `f64::NAN` if the denominator is zero (e.g., regular graphs or
/// graphs with no edges).
pub fn degree_assortativity<G: GraphAdapter>(g: G) -> f64 {
    let snap = Snapshot::new(g);
    let n = snap.len();
    // Collect edges as (j, k) degree pairs — one entry per undirected edge.
    let mut edges: Vec<(f64, f64)> = Vec::new();
    for u in 0..n {
        let ju = snap.neighbors(u).len() as f64;
        for &v in snap.neighbors(u) {
            if v > u {
                let kv = snap.neighbors(v).len() as f64;
                edges.push((ju, kv));
            }
        }
    }
    let m = edges.len() as f64;
    if m == 0.0 {
        return f64::NAN;
    }
    // S1 = M⁻¹ Σ (jₑ + kₑ)/2  =  mean degree over edge endpoints
    let s1: f64 = edges.iter().map(|&(j, k)| (j + k) / 2.0).sum::<f64>() / m;
    // S2 = M⁻¹ Σ (jₑ² + kₑ²)/2
    let s2: f64 = edges
        .iter()
        .map(|&(j, k)| (j * j + k * k) / 2.0)
        .sum::<f64>()
        / m;
    // S3 = M⁻¹ Σ jₑ kₑ
    let s3: f64 = edges.iter().map(|&(j, k)| j * k).sum::<f64>() / m;
    let denom = s2 - s1 * s1;
    if denom == 0.0 {
        return f64::NAN;
    }
    (s3 - s1 * s1) / denom
}

// ---------------------------------------------------------------------------
// Rich-club coefficient
// ---------------------------------------------------------------------------

/// Rich-club coefficient φ(k): fraction of edges present among the N_{>k} nodes
/// with degree strictly greater than k.
///
/// `φ(k) = 2 E_{>k} / (N_{>k} (N_{>k} − 1))`
///
/// Returns `0.0` if fewer than two nodes have degree > k.
pub fn rich_club<G: GraphAdapter>(g: G, k: usize) -> f64 {
    let snap = Snapshot::new(g);
    let n = snap.len();
    let rich: Vec<usize> = (0..n).filter(|&u| snap.neighbors(u).len() > k).collect();
    let nr = rich.len();
    if nr < 2 {
        return 0.0;
    }
    let mut e_rich = 0usize;
    for a in 0..rich.len() {
        for b in (a + 1)..rich.len() {
            if snap.adjacent(rich[a], rich[b]) {
                e_rich += 1;
            }
        }
    }
    2.0 * e_rich as f64 / (nr * (nr - 1)) as f64
}

/// Rich-club curve: φ(k) for k in `0..=max_degree`.
///
/// Returns an empty vector for an empty graph. The vector is ordered by k
/// ascending; each entry is `(k, phi(k))`.
pub fn rich_club_curve<G: GraphAdapter>(g: G) -> Vec<(usize, f64)> {
    let snap = Snapshot::new(g);
    let n = snap.len();
    if n == 0 {
        return Vec::new();
    }
    let max_deg = (0..n).map(|u| snap.neighbors(u).len()).max().unwrap_or(0);
    (0..=max_deg)
        .map(|k| {
            let rich: Vec<usize> = (0..n).filter(|&u| snap.neighbors(u).len() > k).collect();
            let nr = rich.len();
            if nr < 2 {
                return (k, 0.0);
            }
            let mut e_rich = 0usize;
            for a in 0..rich.len() {
                for b in (a + 1)..rich.len() {
                    if snap.adjacent(rich[a], rich[b]) {
                        e_rich += 1;
                    }
                }
            }
            (k, 2.0 * e_rich as f64 / (nr * (nr - 1)) as f64)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Triangle-census cross-check helper (used in tests)
// ---------------------------------------------------------------------------

/// Count triangles via the k=3 census — the independent cross-check oracle for
/// [`total_triangles`].
///
/// The k=3 census enumerates every connected 3-node induced subgraph and labels
/// it by its canonical class. The triangle class (K₃) is `Pattern::triangle()`'s
/// class; its count equals `total_triangles`.
pub fn census_triangle_count<G: GraphAdapter>(g: G) -> usize {
    let sel = Selector::connected_k_subsets(3);
    let census = count(g, &sel);
    let tri_class = Pattern::triangle().class_id();
    census.get(&tri_class).copied().unwrap_or(0) as usize
}
