//! The directed census spine: enumerate weakly-connected k-subsets, label by directed
//! adjacency, fold.
//!
//! Structurally this mirrors the crate's internal undirected census module exactly — ESU (Wernicke 2006) drives
//! the traversal — but two things change: connectivity is checked against the
//! *undirected union* of arcs (a subset only needs to be weakly connected), while class
//! labelling uses the *directed* arc relation (so `a -> b` and `b -> a` are distinct).
//! Capped at `k <= 5` ([`MAX_K`](crate::rim::directed::MAX_K)): `k = 3` for the (connected-only) directed-triad
//! graphlet, `k in 4..=5` for the directed graphlet census; the full 16-type triad
//! census (including disconnected types) lives separately in
//! [`triad`](crate::rim::directed::triad).

use std::collections::HashMap;

use super::canonical::{canonical_by, DirectedClassId};
use super::snapshot::{DirectedGraphAdapter, DirectedSnapshot};
use crate::canonical::perms;

/// The largest supported directed-graphlet order (see module docs for why this is
/// smaller than the undirected crate's own internal `MAX_K` (currently `11`):
/// canonicalization still
/// exhausts `k!` permutations, but the mask now needs `k(k-1)` bits instead of
/// `k(k-1)/2`, and — more binding — the automorphism/orbit registry enumerates all
/// `2^(k(k-1))` labelled digraphs once at build time. That sweep is `2^20` masks at
/// `k = 5` (still exact, but slow — seconds, not microseconds — to build the
/// [`DirectedRegistry`](crate::rim::directed::DirectedRegistry)); `k = 6` would need `2^30` and is
/// out of scope).
pub const MAX_K: usize = 5;

/// One connected (weakly) induced k-node directed subgraph: its host `NodeId`s and its
/// directed-graphlet class.
#[derive(Clone, Debug)]
pub struct DirectedInstance<N> {
    /// The matched vertices, in discovery order.
    pub nodes: Vec<N>,
    /// The canonical directed-graphlet class.
    pub class: DirectedClassId,
}

/// What to enumerate: weakly-connected induced directed subgraphs of order `k`.
#[derive(Clone, Copy, Debug)]
pub struct DirectedSelector {
    k: usize,
}

impl DirectedSelector {
    /// Select weakly-connected induced directed k-subsets.
    ///
    /// # Panics
    ///
    /// Panics unless `2 <= k <= MAX_K` (currently 5).
    #[must_use]
    pub fn weakly_connected_k_subsets(k: usize) -> Self {
        assert!(
            (2..=MAX_K).contains(&k),
            "directed graphlet order k must be in 2..={MAX_K}, got {k}"
        );
        DirectedSelector { k }
    }

    /// The subgraph order this selector enumerates.
    #[inline]
    #[must_use]
    pub fn k(&self) -> usize {
        self.k
    }
}

/// A class -> count map: the readout of a [`count_directed`] fold.
pub type DirectedCensus = HashMap<DirectedClassId, u64>;

/// Recursively visit each weakly-connected induced k-subset once, as index-space
/// slices. ESU driven off the undirected neighbor union — identical shape to
/// [`crate::census::for_each_subset`], duplicated here rather than made generic so the
/// directed substrate stays a self-contained seam (its adjacency probe differs: arc,
/// not undirected edge).
pub(crate) fn for_each_subset<N: Copy>(
    snapshot: &DirectedSnapshot<N>,
    k: usize,
    mut f: impl FnMut(&[usize]),
) {
    fn ext_of<N: Copy>(
        s: &DirectedSnapshot<N>,
        sub: &[usize],
        w: usize,
        v: usize,
        ext: &[usize],
    ) -> Vec<usize> {
        let mut e = ext.to_vec();
        for &u in s.undirected_neighbors(w) {
            if u <= v || sub.contains(&u) || e.contains(&u) {
                continue;
            }
            if !sub.iter().any(|&x| s.undirected_adjacent(x, u)) {
                e.push(u);
            }
        }
        e
    }
    fn rec<N: Copy>(
        s: &DirectedSnapshot<N>,
        sub: &mut Vec<usize>,
        ext: &mut Vec<usize>,
        v: usize,
        k: usize,
        f: &mut impl FnMut(&[usize]),
    ) {
        if sub.len() == k {
            f(sub);
            return;
        }
        while let Some(w) = ext.pop() {
            let mut child = ext_of(s, sub, w, v, ext);
            sub.push(w);
            rec(s, sub, &mut child, v, k, f);
            sub.pop();
        }
    }
    for v in 0..snapshot.len() {
        let mut sub = vec![v];
        let mut ext: Vec<usize> = snapshot
            .undirected_neighbors(v)
            .iter()
            .copied()
            .filter(|&u| u > v)
            .collect();
        rec(snapshot, &mut sub, &mut ext, v, k, &mut f);
    }
}

/// Eagerly-built iterator over weakly-connected induced k-node directed subgraphs.
///
/// Unlike the undirected [`crate::Instances`], this collects instances up front rather
/// than driving a lazy explicit-stack traversal: directed support is capped at
/// `k <= 5`, where instance counts stay bounded, so the undirected core's O(V+E)-memory
/// streaming discipline is not load-bearing here.
pub struct DirectedInstances<N> {
    items: std::vec::IntoIter<DirectedInstance<N>>,
}

impl<N> Iterator for DirectedInstances<N> {
    type Item = DirectedInstance<N>;

    fn next(&mut self) -> Option<DirectedInstance<N>> {
        self.items.next()
    }
}

/// Enumerate every weakly-connected induced directed subgraph of order `sel.k`.
///
/// `g` is treated as a *simple directed* graph: self-loop arcs are stripped, parallel
/// same-direction arcs deduped. See [`DirectedGraphAdapter`].
#[must_use]
pub fn enumerate_directed<G>(g: G, sel: &DirectedSelector) -> DirectedInstances<G::NodeId>
where
    G: DirectedGraphAdapter,
{
    let snapshot = DirectedSnapshot::new(g);
    let ps = perms(sel.k);
    let mut items = Vec::new();
    for_each_subset(&snapshot, sel.k, |sub| {
        let class = DirectedClassId(canonical_by(sel.k, &ps, |i, j| {
            snapshot.has_arc(sub[i], sub[j])
        }));
        let nodes = sub.iter().map(|&i| snapshot.id(i)).collect();
        items.push(DirectedInstance { nodes, class });
    });
    DirectedInstances {
        items: items.into_iter(),
    }
}

/// Fold the directed census: class -> count over every weakly-connected induced
/// k-node directed subgraph.
///
/// `g` is treated as a *simple directed* graph (self-loop arcs stripped, parallel
/// same-direction arcs deduped) — see [`DirectedGraphAdapter`].
#[must_use]
pub fn count_directed<G>(g: G, sel: &DirectedSelector) -> DirectedCensus
where
    G: DirectedGraphAdapter,
{
    let snapshot = DirectedSnapshot::new(g);
    let ps = perms(sel.k);
    let mut census: DirectedCensus = HashMap::new();
    for_each_subset(&snapshot, sel.k, |sub| {
        let class = DirectedClassId(canonical_by(sel.k, &ps, |i, j| {
            snapshot.has_arc(sub[i], sub[j])
        }));
        *census.entry(class).or_insert(0) += 1;
    });
    census
}
