//! Per-node orbit attribution: graphlet-degree vectors (GDV) and distributions (GDD).
//!
//! Two nodes of one graphlet are in the same *orbit* when an automorphism of that
//! graphlet maps one to the other. There are 73 orbits across connected graphlets of
//! order `2..=5` (undirected). For every subgraph instance, each participating node
//! is attributed to its orbit via the arg-permutation witnessing the
//! instance→canonical isomorphism; the orbit is looked up in a union-find
//! automorphism [`Registry`]. Because orbit ids are automorphism-invariant, the
//! attribution is independent of enumeration order.

use std::collections::{BTreeMap, HashMap};

use crate::canonical::{all_connected_classes, canonical_arg_by, class_to_adj, mask_of_by, perms};
use crate::census::for_each_subset;
use crate::snapshot::{GraphAdapter, Snapshot};

/// The maximum graphlet order attributed (undirected, closed at k ≤ 5).
const MAX_K: usize = 5;

/// The automorphism-orbit registry: assigns a dense global orbit id to every
/// (class, local-orbit) across orders `2..=MAX_K`, and records each orbit's size.
///
/// Global ids are assigned deterministically (order ascending, then class mask
/// ascending, then local orbit ascending). They are stable but are *not* ORCA's
/// published orbit numbering — aligning to ORCA is a mechanical relabelling.
#[derive(Clone, Debug)]
pub struct Registry {
    /// `(k, class-mask) → [global orbit id per canonical slot]`.
    slot_global: HashMap<(usize, u64), Vec<usize>>,
    /// `global id → (k, class-mask, orbit size)`.
    orbit_info: Vec<(usize, u64, usize)>,
}

impl Registry {
    /// Build the orbit registry for orders `2..=5` (73 orbits).
    pub fn build() -> Self {
        let mut slot_global = HashMap::new();
        let mut orbit_info: Vec<(usize, u64, usize)> = Vec::new();
        let mut next_id = 0usize;
        for k in 2..=MAX_K {
            for class in all_connected_classes(k) {
                let part = orbit_partition(class, k);
                let n_local = part.iter().copied().max().unwrap() + 1;
                let mut sizes = vec![0usize; n_local];
                for &o in &part {
                    sizes[o] += 1;
                }
                let base = next_id;
                let slotmap: Vec<usize> = part.iter().map(|&lo| base + lo).collect();
                for &sz in sizes.iter() {
                    orbit_info.push((k, class, sz));
                }
                next_id += n_local;
                slot_global.insert((k, class), slotmap);
            }
        }
        Registry {
            slot_global,
            orbit_info,
        }
    }

    /// Total number of orbits (73 for `2..=5`).
    #[inline]
    pub fn orbit_count(&self) -> usize {
        self.orbit_info.len()
    }

    /// `(order, class-mask, orbit size)` for a global orbit id.
    #[inline]
    pub fn orbit_meta(&self, id: usize) -> (usize, u64, usize) {
        self.orbit_info[id]
    }

    /// Global orbit id per canonical slot for a `(order, class-mask)`.
    #[cfg(test)]
    pub(crate) fn slot_map(&self, k: usize, class: u64) -> &[usize] {
        &self.slot_global[&(k, class)]
    }
}

/// Partition the canonical representative of `class` into automorphism orbits:
/// `slot_orbit[c]` is the local orbit index of canonical slot `c`.
fn orbit_partition(class: u64, k: usize) -> Vec<usize> {
    let adj = class_to_adj(class, k);
    let is_adj = |a: usize, b: usize| adj[a].contains(&b);
    let ps = perms(k);
    // Automorphisms: permutations whose relabelled mask equals the class mask.
    let autos: Vec<&Vec<usize>> = ps
        .iter()
        .filter(|p| mask_of_by(k, p, is_adj) == class)
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
    // Relabel roots to dense `0..` by first appearance.
    let mut root_to_orbit: BTreeMap<usize, usize> = BTreeMap::new();
    let mut slot_orbit = vec![0usize; k];
    for (i, slot) in slot_orbit.iter_mut().enumerate() {
        let r = find(&mut parent, i);
        let n = root_to_orbit.len();
        *slot = *root_to_orbit.entry(r).or_insert(n);
    }
    slot_orbit
}

/// Per-node graphlet-degree vectors over a graph.
///
/// Row `i` is the 73-entry GDV of the node with [`id`](GdvTable::id) `i`; entry `o`
/// counts the instances in which that node occupies orbit `o`.
#[derive(Clone, Debug)]
pub struct GdvTable<N> {
    orbit_count: usize,
    ids: Vec<N>,
    rows: Vec<Vec<u64>>,
}

impl<N: Copy> GdvTable<N> {
    /// Number of nodes (rows).
    #[inline]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether the table is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Number of orbits (columns); 73 for the `2..=5` registry.
    #[inline]
    pub fn orbit_count(&self) -> usize {
        self.orbit_count
    }

    /// Host `NodeId` for row `i`.
    #[inline]
    pub fn id(&self, i: usize) -> N {
        self.ids[i]
    }

    /// The graphlet-degree vector of row `i`.
    #[inline]
    pub fn row(&self, i: usize) -> &[u64] {
        &self.rows[i]
    }

    /// Iterate `(NodeId, GDV)` rows.
    pub fn iter(&self) -> impl Iterator<Item = (N, &[u64])> {
        self.ids
            .iter()
            .copied()
            .zip(self.rows.iter().map(Vec::as_slice))
    }

    /// The graphlet-degree distribution for one orbit: `degree → number of nodes`.
    pub fn degree_distribution(&self, orbit: usize) -> BTreeMap<u64, u64> {
        let mut dist = BTreeMap::new();
        for row in &self.rows {
            *dist.entry(row[orbit]).or_insert(0) += 1;
        }
        dist
    }
}

/// Compute the graphlet-degree vector of every node (orders `2..=5`, undirected).
///
/// One ESU pass per order attributes each node of each instance to its orbit; cost
/// is a small constant factor over the class-only census. Pass a [`Registry`] so it
/// can be reused across graphs.
///
/// `g` is treated as a *simple undirected* graph (self-loops stripped, parallel edges
/// deduped, directed inputs unioned) — see [`GraphAdapter`].
pub fn graphlet_degree_vectors<G>(g: G, reg: &Registry) -> GdvTable<G::NodeId>
where
    G: GraphAdapter,
{
    let snapshot = Snapshot::new(g);
    let n = snapshot.len();
    let mut rows = vec![vec![0u64; reg.orbit_count()]; n];
    for k in 2..=MAX_K {
        let ps = perms(k);
        for_each_subset(&snapshot, k, |sub| {
            let (class, arg) = canonical_arg_by(k, &ps, |i, j| snapshot.adjacent(sub[i], sub[j]));
            let slotmap = &reg.slot_global[&(k, class)];
            for (c, &slot) in slotmap.iter().enumerate() {
                rows[sub[arg[c]]][slot] += 1;
            }
        });
    }
    let ids = (0..n).map(|i| snapshot.id(i)).collect();
    GdvTable {
        orbit_count: reg.orbit_count(),
        ids,
        rows,
    }
}
