//! The census spine: enumerate connected k-subsets, label, fold.
//!
//! Connected induced k-subsets are enumerated exactly once each by ESU (Wernicke
//! 2006). [`enumerate`] exposes this as a lazy explicit-stack iterator whose frames
//! are bounded by `k`; [`count`] folds the same traversal into a [`Census`] without
//! ever materializing an [`Instance`], so its memory tracks `O(V+E)` rather than the
//! instance count.

use std::collections::HashMap;

use crate::canonical::{canonical_by, perms, ClassId};
use crate::snapshot::{GraphAdapter, Snapshot};

/// One connected induced k-subgraph: its host `NodeId`s and its graphlet class.
#[derive(Clone, Debug)]
pub struct Instance<N> {
    /// The matched vertices, in ESU discovery order.
    pub nodes: Vec<N>,
    /// The canonical graphlet class.
    pub class: ClassId,
}

/// The largest supported graphlet order.
///
/// A class is identified by a `u64` canonical mask packing `k(k-1)/2` upper-triangle
/// bits; `k = 11` needs 55 bits (fits) while `k = 12` needs 66 (overflows `u64` and
/// would *silently* corrupt the labelling). The census core is correct for `2..=11`,
/// but note that canonicalization enumerates `k!` permutations, so orders past `k ≈ 8`
/// are wildly expensive and the science-facing surface (GDV/orbits, catalog) is closed
/// at `k ≤ 5`.
pub const MAX_K: usize = 11;

/// What to enumerate: connected induced subgraphs of order `k`.
///
/// `k` is validated at construction ([`Selector::connected_k_subsets`]); it is private
/// so a `Selector` can never carry an out-of-range order.
#[derive(Clone, Copy, Debug)]
pub struct Selector {
    /// Subgraph order, guaranteed in `2..=MAX_K`.
    k: usize,
}

impl Selector {
    /// Select connected induced k-subsets.
    ///
    /// # Panics
    ///
    /// Panics unless `2 <= k <= MAX_K` (currently 11). `k < 2` is degenerate (a single
    /// vertex is not a graphlet); `k > 11` overflows the `u64` canonical mask. Out-of-range
    /// `k` is a programming error, so it panics rather than returning a `Result`.
    #[must_use]
    pub fn connected_k_subsets(k: usize) -> Self {
        assert!(
            (2..=MAX_K).contains(&k),
            "graphlet order k must be in 2..={MAX_K}, got {k}"
        );
        Selector { k }
    }

    /// The subgraph order this selector enumerates.
    #[inline]
    #[must_use]
    pub fn k(&self) -> usize {
        self.k
    }
}

/// A class → count map: the readout of a [`count`] fold.
pub type Census = HashMap<ClassId, u64>;

/// A stack frame of the ESU traversal: the extension set still to be drained at this
/// depth and the root `v` that anchors the current tree. The subset itself is *not*
/// stored per frame — it lives once in [`Instances::sub`], shared across the whole
/// root-to-leaf path (see `next`).
struct Frame {
    ext: Vec<usize>,
    v: usize,
}

/// Lazy iterator over connected induced k-subgraphs, owning an adjacency snapshot.
///
/// Constructed by [`enumerate`]. Explicit-stack ESU: `next` advances one leaf at a
/// time and never holds more than the current root-to-leaf path (`O(k · depth)`).
pub struct Instances<N> {
    snapshot: Snapshot<N>,
    k: usize,
    ps: Vec<Vec<usize>>,
    next_root: usize,
    frames: Vec<Frame>,
    /// The current subset, shared across the frame stack: its length equals the frame
    /// depth, so `sub[d]` is the vertex introduced by frame `d`. Pushed/popped in lock
    /// step with `frames`, so no per-step subset clone is needed.
    sub: Vec<usize>,
}

impl<N: Copy> Instances<N> {
    fn new(snapshot: Snapshot<N>, k: usize) -> Self {
        Instances {
            k,
            ps: perms(k),
            next_root: 0,
            frames: Vec::new(),
            sub: Vec::new(),
            snapshot,
        }
    }

    /// Exclusive extension of `w` w.r.t. `sub` (Wernicke ESU), appended to `ext`.
    fn extend(&self, sub: &[usize], w: usize, v: usize, mut ext: Vec<usize>) -> Vec<usize> {
        for &u in self.snapshot.neighbors(w) {
            if u <= v || sub.contains(&u) || ext.contains(&u) {
                continue;
            }
            // Exclusive: u must not be adjacent to any vertex already in sub.
            if !sub.iter().any(|&s| self.snapshot.adjacent(s, u)) {
                ext.push(u);
            }
        }
        ext
    }

    fn class_of(&self, sub: &[usize]) -> ClassId {
        ClassId(canonical_by(self.k, &self.ps, |i, j| {
            self.snapshot.adjacent(sub[i], sub[j])
        }))
    }
}

impl<N: Copy> Iterator for Instances<N> {
    type Item = Instance<N>;

    fn next(&mut self) -> Option<Instance<N>> {
        loop {
            if self.frames.is_empty() {
                if self.next_root >= self.snapshot.len() {
                    return None;
                }
                let v = self.next_root;
                self.next_root += 1;
                let ext: Vec<usize> = self
                    .snapshot
                    .neighbors(v)
                    .iter()
                    .copied()
                    .filter(|&u| u > v)
                    .collect();
                self.sub.clear();
                self.sub.push(v);
                self.frames.push(Frame { ext, v });
                continue;
            }
            let last = self.frames.len() - 1;
            let Some(w) = self.frames[last].ext.pop() else {
                // Frame drained: retreat, popping the vertex it introduced.
                self.frames.pop();
                self.sub.pop();
                continue;
            };
            let v = self.frames[last].v;
            if self.sub.len() + 1 == self.k {
                // Leaf: complete the subset, emit, and restore it in place.
                self.sub.push(w);
                let class = self.class_of(&self.sub);
                let nodes = self.sub.iter().map(|&i| self.snapshot.id(i)).collect();
                self.sub.pop();
                return Some(Instance { nodes, class });
            }
            // Exclusive neighborhood is computed against the subset BEFORE w is added;
            // the child's extension seeds from this frame's remaining `ext`.
            let remaining = self.frames[last].ext.clone();
            let child_ext = self.extend(&self.sub, w, v, remaining);
            self.sub.push(w);
            self.frames.push(Frame { ext: child_ext, v });
        }
    }
}

/// Enumerate every connected induced subgraph of order `sel.k`, lazily.
///
/// The returned iterator owns an `O(V+E)` snapshot of `g`; it does not borrow `g`
/// past construction. `enumerate(g, sel).collect()` materializes all instances
/// (`O(instances)` memory) — use [`count`] to avoid that.
///
/// `g` is treated as a *simple undirected* graph: self-loops are stripped, parallel
/// edges deduped, and directed inputs unioned. See [`GraphAdapter`] for the full
/// precondition.
#[must_use]
pub fn enumerate<G>(g: G, sel: &Selector) -> Instances<G::NodeId>
where
    G: GraphAdapter,
{
    Instances::new(Snapshot::new(g), sel.k)
}

/// Recursively visit each connected induced k-subset once, as index-space slices.
/// The recursive form is the permanent independent oracle for the explicit-stack
/// iterator, and the allocation-free driver for [`count`].
pub(crate) fn for_each_subset<N: Copy>(
    snapshot: &Snapshot<N>,
    k: usize,
    mut f: impl FnMut(&[usize]),
) {
    fn ext_of<N: Copy>(
        s: &Snapshot<N>,
        sub: &[usize],
        w: usize,
        v: usize,
        ext: &[usize],
    ) -> Vec<usize> {
        let mut e = ext.to_vec();
        for &u in s.neighbors(w) {
            if u <= v || sub.contains(&u) || e.contains(&u) {
                continue;
            }
            if !sub.iter().any(|&x| s.adjacent(x, u)) {
                e.push(u);
            }
        }
        e
    }
    fn rec<N: Copy>(
        s: &Snapshot<N>,
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
            .neighbors(v)
            .iter()
            .copied()
            .filter(|&u| u > v)
            .collect();
        rec(snapshot, &mut sub, &mut ext, v, k, &mut f);
    }
}

/// Fold the census as a stream: no [`Instance`] is allocated, so peak memory tracks
/// graph size (`O(V+E)` for the snapshot plus `≤ #classes` for the map), not the
/// number of subgraph instances.
///
/// `g` is treated as a *simple undirected* graph (self-loops stripped, parallel edges
/// deduped, directed inputs unioned) — see [`GraphAdapter`].
#[must_use]
pub fn count<G>(g: G, sel: &Selector) -> Census
where
    G: GraphAdapter,
{
    let snapshot = Snapshot::new(g);
    let ps = perms(sel.k);
    let mut census: Census = HashMap::new();
    for_each_subset(&snapshot, sel.k, |sub| {
        let class = ClassId(canonical_by(sel.k, &ps, |i, j| {
            snapshot.adjacent(sub[i], sub[j])
        }));
        *census.entry(class).or_insert(0) += 1;
    });
    census
}
