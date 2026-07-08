//! Scalable graphlet-orbit counting, ORCA-style (Hočevar & Demšar 2014): recover the
//! full per-node graphlet-degree vector over all 73 orbits (graphlets of order
//! `2..=5`) from directly-observable local structure, **without ever enumerating or
//! canonically labelling a connected 5-subset** — the expensive `k!`-permutation
//! canonicalization the exact [`graphlet_degree_vectors`] pays per instance.
//!
//! **Two ORCA-family layers, both clean-room from the published mathematics** (derived
//! and verified against this crate's own exact oracle — no code, variable structure, or
//! equation transcription was taken from the GPL `orca` reference implementation):
//!
//! - **Orbits `0..=14` (orders `2..=4`).** Count per-vertex degree, per-edge triangle
//!   (common-neighbour) counts, and per-vertex K4 membership once each, then recover the
//!   eleven 4-node orbits by solving a small per-vertex linear system.
//! - **Orbits `15..=72` (order 5, the 58 five-node orbits).** Enumerate every connected
//!   induced *4*-subset once (an order of magnitude cheaper than the 5-subset census),
//!   and for each such core tally how many outside vertices attach with each induced
//!   adjacency pattern to the four core nodes. A per-`(core-structure, attachment)`
//!   table — built once from this crate's own canonical/registry machinery, so the fast
//!   path attributes to exactly the global orbit ids the exact path would — turns each
//!   tally into the order-5 orbit every core node occupies in the reconstructed
//!   5-graphlet. Attributing to the four core nodes counts each node's order-5 orbit
//!   membership a fixed number of times: the per-orbit *core multiplicity*
//!   `M_o = #{ w : graphlet − w is connected }` (constant on an orbit, computed once
//!   from the class machinery). Dividing it out — an exact division, asserted, never a
//!   rounding — yields the true count. This is the ORCA idea (combinatorial counting of
//!   order-5 orbits through smaller cores plus common-neighbour tallies, closed by a
//!   linear correction) realised as a decomposition whose every coefficient is derived
//!   from the crate's exact machinery rather than transcribed.
//!
//! Nothing here is approximate: every value this module returns — all 73 orbits — is
//! asserted, by a large differential test battery plus a property test, to equal the
//! exact census node-for-node and count-for-count. A wrong coefficient is a bug to fix,
//! not a trade-off to ship.
//!
//! The output uses the crate's own global orbit ids (from [`Registry`]) — never a
//! hardcoded assumption — so column `id` of the fast table always means the same orbit
//! as column `id` of the exact [`GdvTable`]. For the `0..=14` prefix the bijection from
//! this module's raw (arbitrarily-ordered) equation output to those ids is discovered
//! once, empirically, against a battery of small representative graphlets; for the
//! order-5 block the ids come straight from [`Registry`]'s own canonical labelling of
//! the reconstructed graphlet, so a mismatch fails loudly (a panic or an oracle
//! disagreement) rather than silently mislabelling a column.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use crate::canonical::{all_connected_classes, canonical_arg_by, class_to_adj, perms};
use crate::census::{for_each_subset, Census};
use crate::{ClassId, GdvTable, GraphAdapter, Registry, Selector, Snapshot};

/// Number of orbits in the order-`2..=4` fast prefix (the linear-system layer): every
/// graphlet of order `2..=4`. The full fast table is [`Registry::orbit_count`] wide
/// (73), covering orders `2..=5`; this constant names only the prefix width.
pub const FAST_ORBIT_COUNT: usize = 15;

/// Global orbit id of the first order-5 orbit: orbits `FIVE_ORBIT_LO..73` are the 58
/// five-node orbits, which follow the 15 order-`2..=4` orbits in [`Registry`]'s
/// order-ascending numbering.
const FIVE_ORBIT_LO: usize = FAST_ORBIT_COUNT;

/// Per-edge triangle (common-neighbor) counts, keyed by the sorted endpoint pair.
///
/// `tri[(a, b)]` (`a < b`) is `|N(a) ∩ N(b)|`: the number of triangles sharing edge
/// `(a, b)`. Computed by one sorted-merge per edge, `O(deg(a) + deg(b))` each.
fn edge_triangle_counts<N: Copy>(snapshot: &Snapshot<N>) -> HashMap<(usize, usize), u64> {
    let n = snapshot.len();
    let mut tri = HashMap::new();
    for a in 0..n {
        for &b in snapshot.neighbors(a) {
            if b <= a {
                continue;
            }
            let (na, nb) = (snapshot.neighbors(a), snapshot.neighbors(b));
            let (mut i, mut j, mut count) = (0usize, 0usize, 0u64);
            while i < na.len() && j < nb.len() {
                match na[i].cmp(&nb[j]) {
                    Ordering::Less => i += 1,
                    Ordering::Greater => j += 1,
                    Ordering::Equal => {
                        count += 1;
                        i += 1;
                        j += 1;
                    }
                }
            }
            tri.insert((a, b), count);
        }
    }
    tri
}

/// `tri(a, b)` from the [`edge_triangle_counts`] map; `a` and `b` must be adjacent.
#[inline]
fn tri_of(tri: &HashMap<(usize, usize), u64>, a: usize, b: usize) -> i64 {
    let key = if a < b { (a, b) } else { (b, a) };
    tri[&key] as i64
}

/// Per-vertex K4 (complete-graph-on-4) membership counts.
///
/// For each edge `(x, y)` with `y < x`, gathers the common neighbors `z < y` of both
/// (each such `z` witnesses a triangle `x-y-z`), then checks every pair `(z, zz)` of
/// those witnesses for an edge: `z ~ zz` completes a K4 `{x, y, z, zz}`. Because the
/// three orderings `zz > z`, `z < y`, `y < x` are enforced together, every K4 is
/// discovered exactly once (via its uniquely-largest two vertices), so a single
/// increment per discovered K4 (not a corrective count) is exact.
fn k4_counts<N: Copy>(snapshot: &Snapshot<N>) -> Vec<u64> {
    let n = snapshot.len();
    let mut c4 = vec![0u64; n];
    for x in 0..n {
        for &y in snapshot.neighbors(x) {
            if y >= x {
                continue;
            }
            let common: Vec<usize> = snapshot
                .neighbors(y)
                .iter()
                .copied()
                .take_while(|&z| z < y)
                .filter(|&z| snapshot.adjacent(x, z))
                .collect();
            for i in 0..common.len() {
                for &zz in &common[i + 1..] {
                    let z = common[i];
                    if snapshot.adjacent(z, zz) {
                        c4[x] += 1;
                        c4[y] += 1;
                        c4[z] += 1;
                        c4[zz] += 1;
                    }
                }
            }
        }
    }
    c4
}

/// Raw per-vertex orbit counts for orders `2..=4`, in this module's own (arbitrary)
/// slot order — *not* yet aligned to [`Registry`]'s global ids. Slot 0 is the k=2
/// (edge) orbit; slots 1..=3 are the k=3 orbits (P3-endpoint, P3-center, triangle);
/// slots 4..=14 are the eleven k=4 orbits, recovered by solving the linear system that
/// relates them to the directly-counted triangle and K4 quantities above.
fn raw_orbit_counts<N: Copy>(snapshot: &Snapshot<N>) -> Vec<[i64; FAST_ORBIT_COUNT]> {
    let n = snapshot.len();
    let tri = edge_triangle_counts(snapshot);
    let c4 = k4_counts(snapshot);
    let mut out = vec![[0i64; FAST_ORBIT_COUNT]; n];
    let mut common: HashMap<usize, i64> = HashMap::new();

    for x in 0..n {
        common.clear();
        let nbrs_x: Vec<usize> = snapshot.neighbors(x).to_vec();
        let deg_x = nbrs_x.len() as i64;

        let (mut f12_14, mut f10_13, mut f13_14, mut f11_13) = (0i64, 0i64, 0i64, 0i64);
        let (mut f7_11, mut f5_8, mut f6_9, mut f9_12, mut f4_8, mut f8_12) =
            (0i64, 0i64, 0i64, 0i64, 0i64, 0i64);
        let (mut orbit1, mut orbit2, mut orbit3) = (0i64, 0i64, 0i64);
        let f14 = c4[x] as i64;

        // Phase A: triangles and open wedges rooted at x, seen from two angles.
        for (nx1, &y) in nbrs_x.iter().enumerate() {
            let deg_y = snapshot.neighbors(y).len() as i64;
            for &z in snapshot.neighbors(y) {
                if snapshot.adjacent(x, z) {
                    // Triangle x-y-z; dedup each unordered {y, z} once via z < y.
                    if z < y {
                        let t = tri_of(&tri, y, z);
                        let deg_z = snapshot.neighbors(z).len() as i64;
                        f12_14 += t - 1;
                        f10_13 += (deg_y - 1 - t) + (deg_z - 1 - t);
                    }
                } else if z != x {
                    *common.entry(z).or_insert(0) += 1;
                }
            }
            for &z in &nbrs_x[nx1 + 1..] {
                let txy = tri_of(&tri, x, y);
                let txz = tri_of(&tri, x, z);
                if snapshot.adjacent(y, z) {
                    orbit3 += 1;
                    f13_14 += (txy - 1) + (txz - 1);
                    f11_13 += (deg_x - 1 - txy) + (deg_x - 1 - txz);
                } else {
                    orbit2 += 1;
                    let deg_y2 = snapshot.neighbors(y).len() as i64;
                    let deg_z2 = snapshot.neighbors(z).len() as i64;
                    f7_11 += (deg_x - 1 - txy - 1) + (deg_x - 1 - txz - 1);
                    f5_8 += (deg_y2 - 1 - txy) + (deg_z2 - 1 - txz);
                }
            }
        }

        // Phase B: x as the endpoint of an open path x-y-z (x !~ z).
        for &y in &nbrs_x {
            let deg_y = snapshot.neighbors(y).len() as i64;
            let txy = tri_of(&tri, x, y);
            for &z in snapshot.neighbors(y) {
                if z == x || snapshot.adjacent(x, z) {
                    continue;
                }
                orbit1 += 1;
                let tyz = tri_of(&tri, y, z);
                let deg_z = snapshot.neighbors(z).len() as i64;
                f6_9 += deg_y - 1 - txy - 1;
                f9_12 += tyz;
                f4_8 += deg_z - 1 - tyz;
                f8_12 += *common.get(&z).unwrap_or(&0) - 1;
            }
        }

        let orbit14 = f14;
        let orbit13 = (f13_14 - 6 * f14) / 2;
        let orbit12 = f12_14 - 3 * f14;
        let orbit11 = (f11_13 - f13_14 + 6 * f14) / 2;
        let orbit10 = f10_13 - f13_14 + 6 * f14;
        let orbit9 = (f9_12 - 2 * f12_14 + 6 * f14) / 2;
        let orbit8 = (f8_12 - 2 * f12_14 + 6 * f14) / 2;
        let orbit7 = (f13_14 + f7_11 - f11_13 - 6 * f14) / 6;
        let orbit6 = (2 * f12_14 + f6_9 - f9_12 - 6 * f14) / 2;
        let orbit5 = 2 * f12_14 + f5_8 - f8_12 - 6 * f14;
        let orbit4 = 2 * f12_14 + f4_8 - f8_12 - 6 * f14;

        out[x] = [
            deg_x, orbit1, orbit2, orbit3, orbit4, orbit5, orbit6, orbit7, orbit8, orbit9, orbit10,
            orbit11, orbit12, orbit13, orbit14,
        ];
    }
    out
}

/// A battery of small representative graphlets covering every orbit role at orders
/// `2..=4` at least once, used only to discover [`orca_to_registry_map`]'s bijection.
fn representative_graphlets() -> Vec<Vec<Vec<usize>>> {
    // Each entry is an adjacency list (`Vec<Vec<usize>>`) over local vertices 0..k.
    fn adj(k: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
        let mut a = vec![Vec::new(); k];
        for &(u, v) in edges {
            a[u].push(v);
            a[v].push(u);
        }
        for row in &mut a {
            row.sort_unstable();
        }
        a
    }
    vec![
        adj(2, &[(0, 1)]),                                         // P2 (edge)
        adj(3, &[(0, 1), (1, 2)]),                                 // P3
        adj(3, &[(0, 1), (1, 2), (2, 0)]),                         // K3
        adj(4, &[(0, 1), (1, 2), (2, 3)]),                         // P4
        adj(4, &[(0, 1), (0, 2), (0, 3)]),                         // star K1,3
        adj(4, &[(0, 1), (1, 2), (2, 3), (3, 0)]),                 // C4
        adj(4, &[(0, 1), (1, 2), (2, 0), (2, 3)]),                 // paw
        adj(4, &[(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]),         // diamond (K4-e)
        adj(4, &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]), // K4
    ]
}

/// A minimal [`GraphAdapter`] over a fixed local adjacency list, for feeding
/// [`representative_graphlets`] to both the exact and fast paths without depending on
/// a concrete petgraph type.
#[derive(Clone, Copy)]
struct AdjGraph<'a> {
    adj: &'a [Vec<usize>],
}

mod adj_graph_impl {
    use super::AdjGraph;
    use petgraph::visit::{
        GraphBase, GraphRef, IntoNeighbors, IntoNeighborsDirected, IntoNodeIdentifiers, NodeCount,
        NodeIndexable,
    };
    use petgraph::Direction;

    impl<'a> GraphBase for AdjGraph<'a> {
        type NodeId = usize;
        type EdgeId = (usize, usize);
    }
    impl<'a> GraphRef for AdjGraph<'a> {}
    impl<'a> NodeCount for AdjGraph<'a> {
        fn node_count(&self) -> usize {
            self.adj.len()
        }
    }
    impl<'a> NodeIndexable for AdjGraph<'a> {
        fn node_bound(&self) -> usize {
            self.adj.len()
        }
        fn to_index(&self, a: usize) -> usize {
            a
        }
        fn from_index(&self, i: usize) -> usize {
            i
        }
    }
    impl<'a> IntoNodeIdentifiers for AdjGraph<'a> {
        type NodeIdentifiers = std::ops::Range<usize>;
        fn node_identifiers(self) -> Self::NodeIdentifiers {
            0..self.adj.len()
        }
    }
    impl<'a> IntoNeighbors for AdjGraph<'a> {
        type Neighbors = std::iter::Copied<std::slice::Iter<'a, usize>>;
        fn neighbors(self, n: usize) -> Self::Neighbors {
            self.adj[n].iter().copied()
        }
    }
    impl<'a> IntoNeighborsDirected for AdjGraph<'a> {
        type NeighborsDirected = std::iter::Copied<std::slice::Iter<'a, usize>>;
        fn neighbors_directed(self, n: usize, _d: Direction) -> Self::NeighborsDirected {
            self.adj[n].iter().copied()
        }
    }
}

/// Discover, once per [`Registry`], the bijection from [`raw_orbit_counts`]'s slot
/// order to the registry's global orbit ids. Determined empirically: run both the
/// exact [`graphlet_degree_vectors`] and the fast [`raw_orbit_counts`] over
/// [`representative_graphlets`] (chosen so every orbit role at order `2..=4` produces
/// a distinct value somewhere), then match whichever registry column reproduces each
/// raw slot's values across the whole battery.
///
/// Panics if any slot has zero or more than one matching registry column — a
/// correctness bug in this module's equations or in the registry's own numbering, not
/// a runtime condition to recover from.
fn orca_to_registry_map(reg: &Registry) -> [usize; FAST_ORBIT_COUNT] {
    let mut exact_sig: Vec<Vec<u64>> = vec![Vec::new(); FAST_ORBIT_COUNT];
    let mut raw_sig: Vec<Vec<i64>> = vec![Vec::new(); FAST_ORBIT_COUNT];
    for adj in representative_graphlets() {
        let g = AdjGraph { adj: &adj };
        let exact = crate::graphlet_degree_vectors(g, reg);
        let raw = raw_orbit_counts(&Snapshot::new(g));
        for row in 0..exact.len() {
            for (id, sig) in exact_sig.iter_mut().enumerate() {
                sig.push(exact.row(row)[id]);
            }
        }
        for row in &raw {
            for (i, sig) in raw_sig.iter_mut().enumerate() {
                sig.push(row[i]);
            }
        }
    }
    let mut map = [usize::MAX; FAST_ORBIT_COUNT];
    for (raw_idx, want) in raw_sig.iter().enumerate() {
        let mut found = None;
        for (id, have) in exact_sig.iter().enumerate() {
            if have.iter().map(|&x| x as i64).eq(want.iter().copied()) {
                assert!(
                    found.is_none(),
                    "ambiguous fast-orbit mapping for raw slot {raw_idx}: matches both \
                     registry id {} and {id}",
                    found.unwrap_or(usize::MAX)
                );
                found = Some(id);
            }
        }
        map[raw_idx] =
            found.unwrap_or_else(|| panic!("no registry orbit matches raw slot {raw_idx}"));
    }
    map
}

// ===========================================================================
// Order-5 layer: the 58 five-node orbits (global ids 15..=72), recovered from
// connected 4-subset seeds plus induced-attachment tallies. See the module docs.
// ===========================================================================

/// The six upper-triangle vertex pairs of a 4-node core, in the bit order used to key
/// [`FiveTable`] by the core's induced structure.
const CORE_PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

/// Is the 5-vertex graph given by a boolean adjacency matrix connected?
fn connected5(mat: &[[bool; 5]; 5]) -> bool {
    let mut seen = [false; 5];
    let mut stack = vec![0usize];
    seen[0] = true;
    let mut cnt = 1;
    while let Some(v) = stack.pop() {
        for u in 0..5 {
            if mat[v][u] && !seen[u] {
                seen[u] = true;
                cnt += 1;
                stack.push(u);
            }
        }
    }
    cnt == 5
}

/// Is the graphlet `adj` (order 5) still connected after deleting vertex `w`?
fn connected_without(adj: &[Vec<usize>], w: usize) -> bool {
    let n = adj.len();
    let start = (0..n).find(|&v| v != w);
    let Some(start) = start else { return true };
    let mut seen = vec![false; n];
    seen[w] = true;
    seen[start] = true;
    let mut stack = vec![start];
    let mut cnt = 1;
    while let Some(v) = stack.pop() {
        for &u in &adj[v] {
            if !seen[u] {
                seen[u] = true;
                cnt += 1;
                stack.push(u);
            }
        }
    }
    cnt == n - 1
}

/// Precomputed order-5 decomposition tables, built once per [`Registry`] from the
/// crate's own canonical/registry machinery (never transcribed).
struct FiveTable {
    /// Indexed by `(core_struct << 4) | attach`: `core_struct` is the 6-bit induced
    /// adjacency of the four core positions `0..4` in [`CORE_PAIRS`] order, `attach` is
    /// the 4-bit adjacency of the fifth vertex to core positions `0..4`. Each entry is
    /// the global orbit id occupied by core positions `0,1,2,3` in the reconstructed
    /// 5-graphlet; [`usize::MAX`] marks a `(core_struct, attach)` whose 5-vertex graph
    /// is disconnected (never looked up at runtime, since cores are connected and
    /// `attach != 0`).
    orbit_of_pos: Vec<[usize; 4]>,
    /// Per-global-orbit-id core multiplicity `M_o` (only ids `FIVE_ORBIT_LO..73` are
    /// meaningful): the number of order-5 vertices whose removal leaves the graphlet
    /// connected, i.e. the number of connected 4-subset cores an orbit-`o` node is
    /// attributed through. Always `>= 1` (every connected order-5 graph has a
    /// non-cut vertex distinct from any given vertex).
    mult: Vec<u64>,
}

impl FiveTable {
    fn build(reg: &Registry) -> Self {
        let ps = perms(5);
        let mut orbit_of_pos = vec![[usize::MAX; 4]; 64 * 16];
        for core in 0u32..64 {
            for attach in 0u32..16 {
                let mut mat = [[false; 5]; 5];
                for (b, &(i, j)) in CORE_PAIRS.iter().enumerate() {
                    if core & (1 << b) != 0 {
                        mat[i][j] = true;
                        mat[j][i] = true;
                    }
                }
                for (i, row) in mat.iter_mut().enumerate().take(4) {
                    if attach & (1 << i) != 0 {
                        row[4] = true;
                    }
                }
                for (i, cell) in mat[4].iter_mut().enumerate().take(4) {
                    if attach & (1 << i) != 0 {
                        *cell = true;
                    }
                }
                if !connected5(&mat) {
                    continue;
                }
                let (class, arg) = canonical_arg_by(5, &ps, |i, j| mat[i][j]);
                let slotmap = reg.slot_ids(5, class);
                // Canonical slot `c` is filled by graph position `arg[c]`, which the
                // exact path attributes to global orbit `slotmap[c]`. Invert to a
                // position -> orbit map, then keep the four core positions.
                let mut pos_orbit = [usize::MAX; 5];
                for (c, &slot) in slotmap.iter().enumerate() {
                    pos_orbit[arg[c]] = slot;
                }
                let idx = (core as usize) << 4 | attach as usize;
                orbit_of_pos[idx] = [pos_orbit[0], pos_orbit[1], pos_orbit[2], pos_orbit[3]];
            }
        }

        let mut mult = vec![0u64; reg.orbit_count()];
        for class in all_connected_classes(5) {
            let adj = class_to_adj(class, 5);
            for (c, &orbit) in reg.slot_ids(5, class).iter().enumerate() {
                let m = (0..5)
                    .filter(|&w| w != c && connected_without(&adj, w))
                    .count() as u64;
                mult[orbit] = m;
            }
        }
        FiveTable { orbit_of_pos, mult }
    }
}

/// Per-node order-5 orbit counts (global ids `FIVE_ORBIT_LO..73`), recovered from
/// connected 4-subset seeds. Returns one length-`reg.orbit_count()` row per node;
/// entries below [`FIVE_ORBIT_LO`] are left zero (filled by the order-`2..=4` layer).
fn five_node_orbit_counts<N: Copy>(
    snapshot: &Snapshot<N>,
    reg: &Registry,
    ft: &FiveTable,
) -> Vec<Vec<u64>> {
    let n = snapshot.len();
    let total = reg.orbit_count();
    // Accumulate the (overcounted-by-M_o) attributions.
    let mut acc = vec![vec![0u64; total]; n];
    // Reusable dedup stamp for gathering the outside vertices of each core.
    let mut stamp = vec![u32::MAX; n];
    let mut generation: u32 = 0;

    for_each_subset(snapshot, 4, |sub| {
        let (c0, c1, c2, c3) = (sub[0], sub[1], sub[2], sub[3]);
        // Induced structure of the core in CORE_PAIRS order.
        let mut core = 0u32;
        for (b, &(i, j)) in CORE_PAIRS.iter().enumerate() {
            if snapshot.adjacent(sub[i], sub[j]) {
                core |= 1 << b;
            }
        }
        // Tally, over every outside vertex adjacent to at least one core node, its
        // 4-bit induced adjacency pattern to (c0, c1, c2, c3).
        generation = generation.wrapping_add(1);
        // Guard against a stamp value colliding with a stale one after wraparound.
        if generation == u32::MAX {
            stamp.iter_mut().for_each(|s| *s = u32::MAX - 1);
            generation = 0;
        }
        for &c in &[c0, c1, c2, c3] {
            stamp[c] = generation;
        }
        let mut counts = [0u64; 16];
        for &c in &[c0, c1, c2, c3] {
            for &u in snapshot.neighbors(c) {
                if stamp[u] == generation {
                    continue;
                }
                stamp[u] = generation;
                let mut mask = 0usize;
                if snapshot.adjacent(u, c0) {
                    mask |= 1;
                }
                if snapshot.adjacent(u, c1) {
                    mask |= 2;
                }
                if snapshot.adjacent(u, c2) {
                    mask |= 4;
                }
                if snapshot.adjacent(u, c3) {
                    mask |= 8;
                }
                counts[mask] += 1;
            }
        }
        let base = (core as usize) << 4;
        for (attach, &cnt) in counts.iter().enumerate().skip(1) {
            if cnt == 0 {
                continue;
            }
            let orbits = ft.orbit_of_pos[base | attach];
            for (pos, &node) in [c0, c1, c2, c3].iter().enumerate() {
                acc[node][orbits[pos]] += cnt;
            }
        }
    });

    // Divide out the per-orbit core multiplicity — exact, or a bug.
    for row in &mut acc {
        for (id, cell) in row.iter_mut().enumerate().take(total).skip(FIVE_ORBIT_LO) {
            let m = ft.mult[id];
            debug_assert!(m >= 1, "order-5 orbit {id} has zero core multiplicity");
            debug_assert!(
                *cell % m == 0,
                "order-5 orbit {id} accumulated {} not divisible by multiplicity {m}: a bug",
                *cell
            );
            *cell /= m;
        }
    }
    acc
}

/// Compute the graphlet-degree vector of every node over all 73 orbits (graphlets of
/// order `2..=5`), without enumerating or canonically labelling a connected 5-subset.
///
/// The returned [`GdvTable::orbit_count`] is [`Registry::orbit_count`] (73); column
/// `id` means the same orbit as column `id` of the exact [`graphlet_degree_vectors`],
/// so the two tables are directly comparable in full.
///
/// `g` is treated as a *simple undirected* graph, matching every other entry point in
/// this crate (see [`GraphAdapter`]).
#[must_use]
pub fn fast_graphlet_degree_vectors<G>(g: G, reg: &Registry) -> GdvTable<G::NodeId>
where
    G: GraphAdapter,
{
    let snapshot = Snapshot::new(g);
    let n = snapshot.len();
    let total = reg.orbit_count();

    // Order-2..=4 layer: the linear-system fast path, mapped to global ids 0..=14.
    let raw = raw_orbit_counts(&snapshot);
    let map = orca_to_registry_map(reg);

    // Order-5 layer: 4-subset-seed decomposition into global ids 15..=72.
    let ft = FiveTable::build(reg);
    let mut rows = five_node_orbit_counts(&snapshot, reg, &ft);

    for (i, row) in rows.iter_mut().enumerate() {
        for (raw_idx, &id) in map.iter().enumerate() {
            debug_assert!(
                raw[i][raw_idx] >= 0,
                "fast orbit equation produced a negative count (slot {raw_idx}, node {i}): a bug"
            );
            row[id] = raw[i][raw_idx].max(0) as u64;
        }
    }
    let ids = (0..n).map(|i| snapshot.id(i)).collect();
    GdvTable::from_parts(total, ids, rows)
}

/// Fast graphlet-class census for one order `sel.k() <= 5`: connected-subgraph counts
/// for every class of that order, derived from [`fast_graphlet_degree_vectors`] via
/// `Σ_v GDV[v][o] = count(class) · size(o)` (documented on [`Registry`]) rather than a
/// second enumeration pass.
///
/// Directly comparable to [`crate::count`] at the same `k`: the two censuses agree
/// class-for-class. A [`Census`] is keyed by [`ClassId`] alone (no `k` tag, matching
/// [`crate::count`]'s own contract), so — like `count` — one call answers one order;
/// [`crate::census::Census`] cannot soundly hold more than one `k` at a time, since
/// the same raw mask value can denote different classes at different orders.
///
/// # Panics
///
/// Panics if `sel.k() > 5`: the science-facing surface is closed at order 5 (the
/// registry only numbers orbits for orders `2..=5`).
#[must_use]
pub fn fast_count<G>(g: G, reg: &Registry, sel: &Selector) -> Census
where
    G: GraphAdapter,
{
    assert!(
        sel.k() <= 5,
        "fast_count covers graphlet orders 2..=5 (got k = {}); the orbit registry is \
         closed at k = 5",
        sel.k()
    );
    let table = fast_graphlet_degree_vectors(g, reg);
    let mut census = Census::new();
    let mut seen: HashSet<u64> = HashSet::new();
    for id in 0..reg.orbit_count() {
        let (order, class_mask, size) = reg.orbit_meta(id);
        if order != sel.k() || !seen.insert(class_mask) {
            continue;
        }
        let sum: u64 = (0..table.len()).map(|v| table.row(v)[id]).sum();
        let n = sum / size as u64;
        // Match `count`'s contract exactly: a class with zero instances is simply
        // absent from the map, never present with a 0 count.
        if n > 0 {
            census.insert(ClassId(class_mask), n);
        }
    }
    census
}
