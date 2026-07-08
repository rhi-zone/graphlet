//! Per-node directed orbit attribution: directed graphlet-degree vectors.
//!
//! Generalizes [`crate::orbit`] from undirected automorphisms to **directed**
//! automorphisms: two nodes of one directed graphlet are in the same orbit when a
//! direction-preserving automorphism (a permutation whose relabelled arc mask equals
//! the class mask) maps one to the other. Covers weakly-connected directed graphlets
//! of order `2..=4` (the [`crate::rim::directed::census::MAX_K`] boundary — directed
//! k = 5 orbits are not implemented; use the exact undirected
//! [`crate::graphlet_degree_vectors`] / [`crate::count`] for k = 5 in the meantime, on
//! the graph's underlying undirected structure).

use std::collections::{BTreeMap, HashMap};

use super::canonical::{all_weakly_connected_classes, canonical_arg_by, class_to_arcs, mask_of_by};
use super::census::for_each_subset;
use super::snapshot::{DirectedGraphAdapter, DirectedSnapshot};
use crate::canonical::perms;

/// The maximum directed graphlet order attributed (weakly connected, closed at
/// `k <= 4`).
const MAX_K: usize = 4;

/// The directed automorphism-orbit registry: assigns a dense global orbit id to every
/// (class, local-orbit) across orders `2..=MAX_K`, and records each orbit's size.
#[derive(Clone, Debug)]
pub struct DirectedRegistry {
    /// `(k, class-mask) -> [global orbit id per canonical slot]`.
    slot_global: HashMap<(usize, u64), Vec<usize>>,
    /// `global id -> (k, class-mask, orbit size)`.
    orbit_info: Vec<(usize, u64, usize)>,
}

impl DirectedRegistry {
    /// Build the directed orbit registry for orders `2..=4`.
    #[must_use]
    pub fn build() -> Self {
        let mut slot_global = HashMap::new();
        let mut orbit_info: Vec<(usize, u64, usize)> = Vec::new();
        let mut next_id = 0usize;
        for k in 2..=MAX_K {
            for class in all_weakly_connected_classes(k) {
                let part = orbit_partition(class, k);
                let n_local = part.iter().copied().max().unwrap() + 1;
                let mut sizes = vec![0usize; n_local];
                for &o in &part {
                    sizes[o] += 1;
                }
                let base = next_id;
                let slotmap: Vec<usize> = part.iter().map(|&lo| base + lo).collect();
                for &sz in &sizes {
                    orbit_info.push((k, class, sz));
                }
                next_id += n_local;
                slot_global.insert((k, class), slotmap);
            }
        }
        DirectedRegistry {
            slot_global,
            orbit_info,
        }
    }

    /// Total number of directed orbits across orders `2..=4`.
    #[inline]
    #[must_use]
    pub fn orbit_count(&self) -> usize {
        self.orbit_info.len()
    }

    /// `(order, class-mask, orbit size)` for a global orbit id.
    #[inline]
    #[must_use]
    pub fn orbit_meta(&self, id: usize) -> (usize, u64, usize) {
        self.orbit_info[id]
    }

    /// Number of distinct weakly-connected directed-graphlet classes at order `k`
    /// (`k` in `2..=4`), or 0 if `k` is out of that range.
    #[must_use]
    pub fn class_count(&self, k: usize) -> usize {
        self.slot_global.keys().filter(|&&(kk, _)| kk == k).count()
    }

    /// Global orbit id per canonical slot for a `(order, class-mask)`.
    #[cfg(test)]
    pub(crate) fn slot_map(&self, k: usize, class: u64) -> &[usize] {
        &self.slot_global[&(k, class)]
    }
}

/// Partition the canonical representative of `class` into **directed** automorphism
/// orbits: `slot_orbit[c]` is the local orbit index of canonical slot `c`.
fn orbit_partition(class: u64, k: usize) -> Vec<usize> {
    let out = class_to_arcs(class, k);
    let has_arc = |a: usize, b: usize| out[a].contains(&b);
    let ps = perms(k);
    // Directed automorphisms: permutations whose relabelled arc mask equals the class
    // mask (arc direction must be preserved, not just adjacency).
    let autos: Vec<&Vec<usize>> = ps
        .iter()
        .filter(|p| mask_of_by(k, p, has_arc) == class)
        .collect();

    let mut parent: Vec<usize> = (0..k).collect();
    fn find(parent: &mut [usize], x: usize) -> usize {
        if parent[x] != x {
            let r = find(parent, parent[x]);
            parent[x] = r;
        }
        parent[x]
    }
    for p in &autos {
        for (i, &pi) in p.iter().enumerate() {
            let a = find(&mut parent, i);
            let b = find(&mut parent, pi);
            if a != b {
                parent[a] = b;
            }
        }
    }
    let mut root_to_orbit: BTreeMap<usize, usize> = BTreeMap::new();
    let mut slot_orbit = vec![0usize; k];
    for (i, slot) in slot_orbit.iter_mut().enumerate() {
        let r = find(&mut parent, i);
        let n = root_to_orbit.len();
        *slot = *root_to_orbit.entry(r).or_insert(n);
    }
    slot_orbit
}

/// Per-node directed graphlet-degree vectors over a digraph.
///
/// Row `i` is the directed GDV of the node with [`id`](DirectedGdvTable::id) `i`; entry
/// `o` counts the weakly-connected instances in which that node occupies directed
/// orbit `o`.
#[derive(Clone, Debug)]
pub struct DirectedGdvTable<N> {
    orbit_count: usize,
    ids: Vec<N>,
    rows: Vec<Vec<u64>>,
}

impl<N: Copy> DirectedGdvTable<N> {
    /// Number of nodes (rows).
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether the table is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Number of directed orbits (columns).
    #[inline]
    #[must_use]
    pub fn orbit_count(&self) -> usize {
        self.orbit_count
    }

    /// Host `NodeId` for row `i`.
    #[inline]
    #[must_use]
    pub fn id(&self, i: usize) -> N {
        self.ids[i]
    }

    /// The directed graphlet-degree vector of row `i`.
    #[inline]
    #[must_use]
    pub fn row(&self, i: usize) -> &[u64] {
        &self.rows[i]
    }

    /// Iterate `(NodeId, directed GDV)` rows.
    pub fn iter(&self) -> impl Iterator<Item = (N, &[u64])> {
        self.ids
            .iter()
            .copied()
            .zip(self.rows.iter().map(Vec::as_slice))
    }
}

/// Compute the directed graphlet-degree vector of every node (weakly-connected orders
/// `2..=4`).
///
/// One weakly-connected-ESU pass per order attributes each node of each instance to
/// its directed orbit. Pass a [`DirectedRegistry`] so it can be reused across graphs.
///
/// `g` is treated as a *simple directed* graph (self-loop arcs stripped, parallel
/// same-direction arcs deduped) — see [`DirectedGraphAdapter`].
#[must_use]
pub fn directed_graphlet_degree_vectors<G>(
    g: G,
    reg: &DirectedRegistry,
) -> DirectedGdvTable<G::NodeId>
where
    G: DirectedGraphAdapter,
{
    let snapshot = DirectedSnapshot::new(g);
    let n = snapshot.len();
    let mut rows = vec![vec![0u64; reg.orbit_count()]; n];
    for k in 2..=MAX_K {
        let ps = perms(k);
        for_each_subset(&snapshot, k, |sub| {
            let (class, arg) = canonical_arg_by(k, &ps, |i, j| snapshot.has_arc(sub[i], sub[j]));
            let slotmap = &reg.slot_global[&(k, class)];
            for (c, &slot) in slotmap.iter().enumerate() {
                rows[sub[arg[c]]][slot] += 1;
            }
        });
    }
    let ids = (0..n).map(|i| snapshot.id(i)).collect();
    DirectedGdvTable {
        orbit_count: reg.orbit_count(),
        ids,
        rows,
    }
}
