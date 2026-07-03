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
//! s(P,P)`.
//!
//! [`count_pattern`] / [`count_motif`] use this verified census fast path for *counts*
//! (`k <= 5`); *instances* ([`find_motif`], [`find_diamonds`]) are enumerated by the
//! [`crate::template`] pattern-instance engine (VF2 induced / this crate's monomorphism
//! enumerator) and deduped to distinct occurrences. Arbitrary-size templates outside the
//! census domain are served by [`crate::template`] directly.

use std::collections::{HashMap, HashSet};

use petgraph::graph::UnGraph;

use crate::canonical::{all_connected_classes, canonical_by, class_to_adj, perms};
use crate::census::{count, Selector};
use crate::snapshot::{GraphAdapter, Snapshot};
use crate::template::{induced_matches_unlabelled, monomorphisms_unlabelled};

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

    /// The path `P_k` on `k` vertices (`0–1–…–(k-1)`). `2 <= k <= 5`.
    #[must_use]
    pub fn path(k: usize) -> Self {
        let edges: Vec<(usize, usize)> = (0..k.saturating_sub(1)).map(|i| (i, i + 1)).collect();
        Pattern::new(k, &edges)
    }

    /// The cycle `C_k` on `k` vertices. `3 <= k <= 5`.
    #[must_use]
    pub fn cycle(k: usize) -> Self {
        assert!((3..=5).contains(&k), "cycle C_k needs 3 <= k <= 5");
        let mut edges: Vec<(usize, usize)> = (0..k - 1).map(|i| (i, i + 1)).collect();
        edges.push((k - 1, 0));
        Pattern::new(k, &edges)
    }

    /// The star `K_{1,k-1}` on `k` vertices: a center `0` adjacent to `1..k`. `2 <= k <= 5`.
    #[must_use]
    pub fn star(k: usize) -> Self {
        let edges: Vec<(usize, usize)> = (1..k).map(|i| (0, i)).collect();
        Pattern::new(k, &edges)
    }

    /// The complete graph `K_k` on `k` vertices. `2 <= k <= 5`.
    #[must_use]
    pub fn complete(k: usize) -> Self {
        let edges: Vec<(usize, usize)> = (0..k)
            .flat_map(|i| ((i + 1)..k).map(move |j| (i, j)))
            .collect();
        Pattern::new(k, &edges)
    }

    /// The triangle `K_3` (the 3-node closed motif).
    #[must_use]
    pub fn triangle() -> Self {
        Pattern::complete(3)
    }

    /// The claw `K_{1,3}` — the 4-node star (one center, three leaves).
    #[must_use]
    pub fn claw() -> Self {
        Pattern::star(4)
    }

    /// The paw — a triangle (`0,1,2`) with a pendant vertex `3` on vertex `0`. One of
    /// the six connected 4-node motifs.
    #[must_use]
    pub fn paw() -> Self {
        Pattern::new(4, &[(0, 1), (1, 2), (2, 0), (0, 3)])
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

/// A named collection of motif [`Pattern`]s, for registering and querying arbitrary
/// user patterns by name alongside the standard ones.
///
/// ```
/// use graphlet::catalog::{count_motif, Induced, MotifCatalog, Pattern};
/// use petgraph::graph::UnGraph;
///
/// let mut cat = MotifCatalog::standard();
/// cat.register("my_square", Pattern::cycle(4)); // register an arbitrary pattern
///
/// let g: UnGraph<(), ()> = UnGraph::from_edges([(0, 1), (1, 2), (2, 3), (3, 0)]);
/// let square = cat.get("my_square").unwrap();
/// assert_eq!(count_motif(&g, square, Induced::Yes), 1);
/// ```
#[derive(Clone, Debug, Default)]
pub struct MotifCatalog {
    patterns: HashMap<String, Pattern>,
}

impl MotifCatalog {
    /// An empty catalog.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A catalog pre-loaded with the standard small motifs: `p3`, `triangle`, `p4`,
    /// `c4`, `claw`, `paw`, `diamond`, `k4`, `p5`, `c5`, `star5`, `k5`.
    #[must_use]
    pub fn standard() -> Self {
        let mut c = Self::new();
        c.register("p3", Pattern::path(3));
        c.register("triangle", Pattern::triangle());
        c.register("p4", Pattern::path(4));
        c.register("c4", Pattern::cycle(4));
        c.register("claw", Pattern::claw());
        c.register("paw", Pattern::paw());
        c.register("diamond", Pattern::diamond());
        c.register("k4", Pattern::complete(4));
        c.register("p5", Pattern::path(5));
        c.register("c5", Pattern::cycle(5));
        c.register("star5", Pattern::star(5));
        c.register("k5", Pattern::complete(5));
        c
    }

    /// Register `pattern` under `name`, returning any pattern the name previously held.
    pub fn register(&mut self, name: impl Into<String>, pattern: Pattern) -> Option<Pattern> {
        self.patterns.insert(name.into(), pattern)
    }

    /// The pattern registered under `name`, if any.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Pattern> {
        self.patterns.get(name)
    }

    /// The registered names, sorted.
    #[must_use]
    pub fn names(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.patterns.keys().map(String::as_str).collect();
        v.sort_unstable();
        v
    }

    /// The number of registered patterns.
    #[must_use]
    pub fn len(&self) -> usize {
        self.patterns.len()
    }

    /// Whether the catalog is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }
}

/// The pattern's automorphism permutations: the `perm`s of `0..k` that preserve its
/// adjacency. Their count is `|Aut(P)| = s(P,P)`.
fn pattern_automorphisms(pattern: &Pattern) -> Vec<Vec<usize>> {
    let k = pattern.k;
    perms(k)
        .into_iter()
        .filter(|p| {
            (0..k).all(|i| {
                pattern.adj[i]
                    .iter()
                    .all(|&j| pattern.adj[p[i]].contains(&p[j]))
            })
        })
        .collect()
}

/// Count occurrences of `pattern` in `g` under the chosen [`Induced`] semantics, as
/// *distinct occurrences* — the general motif count generalizing [`count_diamonds`].
///
/// This is the verified census fast path (identical to [`count_pattern`]): induced
/// counts are read off the census, and non-induced counts use the `s(P,C)` derivation,
/// valid for connected patterns at `k <= 5`. For arbitrary-size templates outside the
/// census domain, use the [`crate::template`] monomorphism engine directly.
#[must_use]
pub fn count_motif<G>(g: G, pattern: &Pattern, induced: Induced) -> u64
where
    G: GraphAdapter,
{
    count_pattern(g, pattern, induced)
}

/// Enumerate distinct occurrences of `pattern` in `g` as node mappings — the general
/// motif instance query generalizing [`find_diamonds`] to any [`Pattern`].
///
/// Each returned `Vec<G::NodeId>` is indexed by pattern vertex: entry `v` is the host
/// node that pattern vertex `v` maps to. Instances are enumerated by the
/// [`crate::template`] pattern-instance engine — petgraph VF2 for [`Induced::Yes`]
/// (node-induced), this crate's monomorphism enumerator for [`Induced::No`]
/// (edge-preserving) — then deduped to *distinct occurrences* by folding out the
/// `|Aut(P)|` embeddings of each occurrence (the same normalization [`count_motif`] /
/// [`count_pattern`] apply to counts). Hence `find_motif(..).len() == count_motif(..)`.
///
/// `g` is treated as a *simple undirected* graph (self-loops stripped, parallel edges
/// deduped, directed inputs unioned) — see [`GraphAdapter`].
#[must_use]
pub fn find_motif<G>(g: G, pattern: &Pattern, induced: Induced) -> Vec<Vec<G::NodeId>>
where
    G: GraphAdapter,
{
    let snapshot = Snapshot::new(g);
    let n = snapshot.len();
    let k = pattern.k;

    // Rebuild the host and pattern as plain `UnGraph`s in the snapshot's dense index
    // space (host index i == snapshot index i), so the engine's returned host indices
    // map straight back through `snapshot.id`.
    let mut host: UnGraph<(), ()> = UnGraph::default();
    let hidx: Vec<_> = (0..n).map(|_| host.add_node(())).collect();
    for i in 0..n {
        for &j in snapshot.neighbors(i) {
            if i < j {
                host.add_edge(hidx[i], hidx[j], ());
            }
        }
    }
    let mut pat: UnGraph<(), ()> = UnGraph::default();
    let pidx: Vec<_> = (0..k).map(|_| pat.add_node(())).collect();
    for (i, nbrs) in pattern.adj.iter().enumerate() {
        for &j in nbrs {
            if i < j {
                pat.add_edge(pidx[i], pidx[j], ());
            }
        }
    }

    let raw = match induced {
        Induced::Yes => induced_matches_unlabelled(&pat, &host),
        Induced::No => monomorphisms_unlabelled(&pat, &host),
    };

    // Dedup raw embeddings to distinct occurrences: every occurrence appears as exactly
    // `|Aut(P)|` embeddings (the automorphism action is free on injective maps), so
    // keying by the lexicographically minimal automorphism image collapses each orbit to
    // one representative — yielding raw/|Aut(P)| = the census-derived count.
    let auts = pattern_automorphisms(pattern);
    let mut seen: HashSet<Vec<usize>> = HashSet::new();
    let mut reps: Vec<Vec<usize>> = Vec::new();
    for e in raw {
        let key = auts
            .iter()
            .map(|p| (0..k).map(|c| e[p[c]]).collect::<Vec<usize>>())
            .min()
            .expect("a pattern has at least the identity automorphism");
        if seen.insert(key) {
            reps.push(e);
        }
    }
    reps.into_iter()
        .map(|e| e.into_iter().map(|hi| snapshot.id(hi)).collect())
        .collect()
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

/// Enumerate diamond occurrences in `g` — a thin wrapper over [`find_motif`] with the
/// diamond [`Pattern`].
///
/// Induced diamonds come from 4-subsets whose induced class is the diamond. For
/// non-induced, each `K4` additionally yields its 6 spanning diamonds. In the diamond
/// [`Pattern`] the degree-3 vertices are `0` and `2` (the shared `spine`) and the
/// degree-2 vertices are `1` and `3` (the `tips`), so each occurrence's node mapping is
/// projected onto those roles.
///
/// `g` is treated as a *simple undirected* graph (self-loops stripped, parallel edges
/// deduped, directed inputs unioned) — see [`GraphAdapter`].
#[must_use]
pub fn find_diamonds<G>(g: G, induced: Induced) -> Vec<Diamond<G::NodeId>>
where
    G: GraphAdapter,
{
    find_motif(g, &Pattern::diamond(), induced)
        .into_iter()
        .map(|m| Diamond {
            spine: [m[0], m[2]],
            tips: [m[1], m[3]],
        })
        .collect()
}
