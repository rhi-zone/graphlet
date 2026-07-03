//! Named-motif catalog over the census substrate.
//!
//! This arm threads [`Induced`] as a real parameter — *both* values are honoured:
//!
//! - **Induced** counts are read straight off the census (the count of node-sets
//!   whose induced subgraph is the motif).
//! - **Non-induced** (monomorphism) counts are derived from the induced census by a
//!   fixed per-`(P,C)` table `s(P,C)` — the number of edge-preserving bijections of
//!   the pattern `P` into graphlet class `C`. The identity
//!   `mono_labelled(P in G) = Σ_C indCount(C)·s(P,C)` was verified against an
//!   independent brute-force monomorphism oracle (1105/1105, k = 3,4,5). There is
//!   **no separate monomorphism enumerator**; the non-induced readout is a bounded
//!   post-pass over the same induced census.
//!
//! Both sides here report *distinct occurrences* (node-sets / structural
//! embeddings), obtained from the labelled identity by dividing by `|Aut(P)| =
//! s(P,P)`. This is the census/catalog arm; arbitrary-template matching lives in
//! [`crate::template`] and is induced-only.

use std::collections::HashMap;

use crate::canonical::{all_connected_classes, canonical_by, class_to_adj, pairs, perms};
use crate::census::{count, for_each_subset, Selector};
use crate::snapshot::{GraphAdapter, Snapshot};

/// Whether a motif query counts induced subgraphs or monomorphisms (non-induced,
/// extra host edges allowed).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Induced {
    /// Induced: the matched vertices induce exactly the motif (no extra edges).
    Yes,
    /// Non-induced (monomorphism): every motif edge is present; extra host edges
    /// among the matched vertices are allowed.
    No,
}

/// A connected motif pattern of order `k`, prepared for census-based counting.
#[derive(Clone, Debug)]
pub struct Pattern {
    k: usize,
    /// Local adjacency of the pattern's `k` vertices.
    adj: Vec<Vec<usize>>,
    /// Canonical class mask of the pattern.
    class: u64,
    /// `|Aut(P)| = s(P,P)`, the labelled→distinct divisor.
    aut: u64,
}

impl Pattern {
    /// Build a pattern of order `k` from an undirected edge list over vertices `0..k`.
    ///
    /// # Panics
    ///
    /// Panics unless `2 <= k <= 5`; on an edge that is out of range or a self-loop
    /// (`a == b`); or if the resulting pattern is disconnected (the census enumerates
    /// only connected subsets, so a disconnected pattern has no census readout).
    #[must_use]
    pub fn new(k: usize, edges: &[(usize, usize)]) -> Self {
        assert!(
            (2..=5).contains(&k),
            "catalog patterns are supported for 2 <= k <= 5"
        );
        let mut adj = vec![Vec::new(); k];
        for &(a, b) in edges {
            assert!(a < k && b < k && a != b, "edge out of range / self-loop");
            if !adj[a].contains(&b) {
                adj[a].push(b);
                adj[b].push(a);
            }
        }
        assert!(
            crate::canonical::connected(&adj),
            "catalog patterns must be connected"
        );
        let ps = perms(k);
        let class = canonical_by(k, &ps, |i, j| adj[i].contains(&j));
        let aut = s_pc(&adj, &adj, &ps);
        Pattern { k, adj, class, aut }
    }

    /// The 4-node diamond (`K4` minus one edge): two triangles sharing an edge.
    #[must_use]
    pub fn diamond() -> Self {
        Pattern::new(4, &[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
    }

    /// The pattern's order (number of vertices).
    #[inline]
    #[must_use]
    pub fn order(&self) -> usize {
        self.k
    }

    /// The pattern's canonical graphlet class.
    #[inline]
    #[must_use]
    pub fn class_id(&self) -> crate::canonical::ClassId {
        crate::canonical::ClassId(self.class)
    }
}

/// `s(P,C)`: the number of edge-preserving bijections `V(P) → V(C)` (spanning
/// monomorphisms — every pattern edge maps to a class edge; extra class edges are
/// allowed). Depends only on `(P,C)`, never on the host graph.
fn s_pc(padj: &[Vec<usize>], cadj: &[Vec<usize>], ps: &[Vec<usize>]) -> u64 {
    let mut count = 0u64;
    for perm in ps {
        let mut ok = true;
        'outer: for (i, nbrs) in padj.iter().enumerate() {
            for &j in nbrs {
                if j > i && !cadj[perm[i]].contains(&perm[j]) {
                    ok = false;
                    break 'outer;
                }
            }
        }
        if ok {
            count += 1;
        }
    }
    count
}

/// Count occurrences of `pattern` in `g` under the chosen [`Induced`] semantics,
/// as *distinct occurrences* (node-sets / structural embeddings).
///
/// Induced: the census count of the pattern's class. Non-induced: the verified
/// `Σ_C indCount(C)·s(P,C)` labelled sum divided by `|Aut(P)|`.
///
/// `g` is treated as a *simple undirected* graph (self-loops stripped, parallel edges
/// deduped, directed inputs unioned) — see [`GraphAdapter`]. Note the asymmetry with
/// [`Pattern::new`], which *rejects* self-loop pattern edges rather than stripping them.
#[must_use]
pub fn count_pattern<G>(g: G, pattern: &Pattern, induced: Induced) -> u64
where
    G: GraphAdapter,
{
    let k = pattern.k;
    let census = count(g, &Selector::connected_k_subsets(k));
    // Re-key the census by raw class mask for `s(P,C)` lookups.
    let by_mask: HashMap<u64, u64> = census.into_iter().map(|(c, n)| (c.0, n)).collect();
    match induced {
        Induced::Yes => by_mask.get(&pattern.class).copied().unwrap_or(0),
        Induced::No => {
            let ps = perms(k);
            let labelled: u64 = all_connected_classes(k)
                .into_iter()
                .map(|mask| {
                    let cnt = by_mask.get(&mask).copied().unwrap_or(0);
                    if cnt == 0 {
                        0
                    } else {
                        cnt * s_pc(&pattern.adj, &class_to_adj(mask, k), &ps)
                    }
                })
                .sum();
            labelled / pattern.aut
        }
    }
}

/// One diamond occurrence: the two shared-edge (degree-3) `spine` vertices and the
/// two `tip` (degree-2) vertices.
#[derive(Clone, Copy, Debug)]
pub struct Diamond<N> {
    /// The shared edge of the two triangles.
    pub spine: [N; 2],
    /// The two apex vertices, each adjacent to both spine vertices.
    pub tips: [N; 2],
}

/// Count diamonds in `g` (see [`count_pattern`]).
#[must_use]
pub fn count_diamonds<G>(g: G, induced: Induced) -> u64
where
    G: GraphAdapter,
{
    count_pattern(g, &Pattern::diamond(), induced)
}

/// Enumerate diamond occurrences in `g`.
///
/// Induced diamonds come from 4-subsets whose induced class is the diamond. For
/// non-induced, each `K4` additionally yields its 6 spanning diamonds (its `C(4,2)`
/// tip-pairs) — the diamond specialization of the `s(P,C)` expansion.
///
/// `g` is treated as a *simple undirected* graph (self-loops stripped, parallel edges
/// deduped, directed inputs unioned) — see [`GraphAdapter`].
#[must_use]
pub fn find_diamonds<G>(g: G, induced: Induced) -> Vec<Diamond<G::NodeId>>
where
    G: GraphAdapter,
{
    let snapshot = Snapshot::new(g);
    let ps = perms(4);
    let diamond_class = Pattern::diamond().class;
    let k4_class = canonical_by(4, &ps, |i, j| i != j); // complete graph on 4 vertices
    let mut out = Vec::new();
    for_each_subset(&snapshot, 4, |sub| {
        let class = canonical_by(4, &ps, |i, j| snapshot.adjacent(sub[i], sub[j]));
        if class == diamond_class {
            // Spine = the two degree-3 vertices, tips = the two degree-2 vertices.
            let mut spine = Vec::new();
            let mut tips = Vec::new();
            for (local, &node) in sub.iter().enumerate() {
                let deg = (0..4)
                    .filter(|&o| o != local && snapshot.adjacent(sub[local], sub[o]))
                    .count();
                if deg == 3 {
                    spine.push(snapshot.id(node));
                } else {
                    tips.push(snapshot.id(node));
                }
            }
            out.push(Diamond {
                spine: [spine[0], spine[1]],
                tips: [tips[0], tips[1]],
            });
        } else if induced == Induced::No && class == k4_class {
            // Each unordered tip-pair (spine = its complement) is a distinct diamond.
            for &(a, b) in &pairs(4) {
                let tips = [snapshot.id(sub[a]), snapshot.id(sub[b])];
                let rest: Vec<usize> = (0..4).filter(|&x| x != a && x != b).collect();
                let spine = [snapshot.id(sub[rest[0]]), snapshot.id(sub[rest[1]])];
                out.push(Diamond { spine, tips });
            }
        }
    });
    out
}
