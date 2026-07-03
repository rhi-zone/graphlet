//! Ported, verified tests from the motif-engine gates: census permutation
//! stability, streaming `count` vs materializing `collect` memory, class counts vs
//! exhaustive ground truth (2/6/21), GDV vs an independent brute-force oracle, the
//! `s(P,C)` non-induced derivation, and the diamond catalog.

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};

use petgraph::graph::{Graph, NodeIndex, UnGraph};
use petgraph::stable_graph::StableGraph;
use petgraph::{Directed, Undirected};
use proptest::prelude::*;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use crate::canonical::{
    all_connected_classes, canonical_arg_by, canonical_by, class_to_adj, connected, perms,
};
use crate::catalog::{
    count_diamonds, count_motif, count_pattern, find_diamonds, find_motif, Induced, MotifCatalog,
    Pattern,
};
use crate::census::{count, enumerate, for_each_subset, Census, Selector};
use crate::orbit::{graphlet_degree_vectors, Registry};
use crate::rim::null_model::{
    configuration_model, configuration_model_simple, double_edge_swap, lfr_benchmark,
    watts_strogatz,
};
use crate::snapshot::Snapshot;
use crate::template::{
    count_induced_matches, count_monomorphisms, count_monomorphisms_unlabelled, monomorphisms,
    monomorphisms_unlabelled,
};
use crate::ClassId;

// ---------------------------------------------------------------------------
// Tracking global allocator (test builds only) — peak live bytes.
// ---------------------------------------------------------------------------
struct Track;
static CUR: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Track {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = System.alloc(l);
        if !p.is_null() {
            let now = CUR.fetch_add(l.size(), Ordering::Relaxed) + l.size();
            let mut prev = PEAK.load(Ordering::Relaxed);
            while now > prev {
                match PEAK.compare_exchange_weak(prev, now, Ordering::Relaxed, Ordering::Relaxed) {
                    Ok(_) => break,
                    Err(x) => prev = x,
                }
            }
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        CUR.fetch_sub(l.size(), Ordering::Relaxed);
        System.dealloc(p, l);
    }
}

#[global_allocator]
static ALLOC: Track = Track;

fn reset_peak() {
    PEAK.store(CUR.load(Ordering::Relaxed), Ordering::Relaxed);
}

// ---------------------------------------------------------------------------
// Graph builders.
// ---------------------------------------------------------------------------
fn build_un(edges: &[(usize, usize)], n: usize) -> UnGraph<(), ()> {
    let mut g = Graph::<(), (), Undirected>::new_undirected();
    let idx: Vec<_> = (0..n).map(|_| g.add_node(())).collect();
    for &(a, b) in edges {
        g.add_edge(idx[a], idx[b], ());
    }
    g
}

fn build_perm_un(edges: &[(usize, usize)], n: usize, order: &[usize]) -> UnGraph<(), ()> {
    let mut g = Graph::<(), (), Undirected>::new_undirected();
    let mut idx = vec![NodeIndex::new(0); n];
    for &logical in order {
        idx[logical] = g.add_node(());
    }
    for &(a, b) in edges {
        g.add_edge(idx[a], idx[b], ());
    }
    g
}

fn build_perm_stable(
    edges: &[(usize, usize)],
    n: usize,
    order: &[usize],
) -> StableGraph<(), (), Undirected> {
    let mut g = StableGraph::<(), (), Undirected>::default();
    let mut idx = vec![petgraph::stable_graph::NodeIndex::new(0); n];
    for &logical in order {
        idx[logical] = g.add_node(());
    }
    for &(a, b) in edges {
        g.add_edge(idx[a], idx[b], ());
    }
    g
}

fn build_weighted(edges: &[(usize, usize)], n: usize) -> Graph<char, f64, Undirected> {
    let mut g = Graph::<char, f64, Undirected>::new_undirected();
    let idx: Vec<_> = (0..n).map(|_| g.add_node('x')).collect();
    for &(a, b) in edges {
        g.add_edge(idx[a], idx[b], 1.5);
    }
    g
}

fn build_directed(edges: &[(usize, usize)], n: usize) -> Graph<(), (), Directed> {
    let mut g = Graph::<(), (), Directed>::new();
    let idx: Vec<_> = (0..n).map(|_| g.add_node(())).collect();
    for &(a, b) in edges {
        g.add_edge(idx[a], idx[b], ());
    }
    g
}

fn random_edges(n: usize, p: f64, seed: u64) -> Vec<(usize, usize)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut e = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            if rng.gen::<f64>() < p {
                e.push((i, j));
            }
        }
    }
    e
}

fn path_edges(n: usize) -> Vec<(usize, usize)> {
    (0..n - 1).map(|i| (i, i + 1)).collect()
}
fn cycle_edges(n: usize) -> Vec<(usize, usize)> {
    let mut e = path_edges(n);
    e.push((n - 1, 0));
    e
}
fn star_edges(n: usize) -> Vec<(usize, usize)> {
    (1..n).map(|i| (0, i)).collect()
}
fn complete_edges(n: usize) -> Vec<(usize, usize)> {
    (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
        .collect()
}

fn sorted(c: &Census) -> Vec<(ClassId, u64)> {
    let mut v: Vec<_> = c.iter().map(|(&k, &n)| (k, n)).collect();
    v.sort();
    v
}
fn total(c: &Census) -> u64 {
    c.values().sum()
}

// Census via the public lazy explicit-stack iterator (independent path from `count`,
// which drives the recursive visitor).
fn census_via_iter<G: crate::GraphAdapter>(g: G, k: usize) -> Census {
    let mut c: Census = HashMap::new();
    for inst in enumerate(g, &Selector::connected_k_subsets(k)) {
        *c.entry(inst.class).or_insert(0) += 1;
    }
    c
}

// ===========================================================================
// [1] class counts vs exhaustive ground truth (2 / 6 / 21).
// ===========================================================================
#[test]
fn class_counts_match_ground_truth() {
    assert_eq!(all_connected_classes(3).len(), 2);
    assert_eq!(all_connected_classes(4).len(), 6);
    assert_eq!(all_connected_classes(5).len(), 21);

    for &(n, seed) in &[(20usize, 1u64), (24, 2), (28, 3)] {
        let g = build_un(&random_edges(n, 0.5, seed), n);
        for &k in &[3usize, 4, 5] {
            let gt: HashSet<u64> = all_connected_classes(k).into_iter().collect();
            let found: HashSet<u64> = count(&g, &Selector::connected_k_subsets(k))
                .keys()
                .map(|c| c.0)
                .collect();
            assert_eq!(
                found, gt,
                "dense G(n={n}) k={k} must exhibit exactly the class set"
            );
        }
    }
}

// ===========================================================================
// [2] census permutation stability (Graph & StableGraph) + generic flavours.
// ===========================================================================
#[test]
fn census_stable_under_relabelling() {
    let n = 16;
    let edges = random_edges(n, 0.5, 42);
    for &k in &[3usize, 4, 5] {
        let identity: Vec<usize> = (0..n).collect();
        let reference = count(
            &build_perm_un(&edges, n, &identity),
            &Selector::connected_k_subsets(k),
        );
        let mut rng = StdRng::seed_from_u64(7 + k as u64);
        for _ in 0..20 {
            let mut order: Vec<usize> = (0..n).collect();
            order.shuffle(&mut rng);
            let g = build_perm_un(&edges, n, &order);
            let sg = build_perm_stable(&edges, n, &order);
            assert_eq!(
                sorted(&count(&g, &Selector::connected_k_subsets(k))),
                sorted(&reference)
            );
            assert_eq!(
                sorted(&count(&sg, &Selector::connected_k_subsets(k))),
                sorted(&reference)
            );
        }
    }
}

#[test]
fn census_generic_over_flavours_and_iter_matches_count() {
    let e = random_edges(16, 0.5, 42);
    for &k in &[3usize, 4, 5] {
        let ug = build_un(&e, 16);
        let sg = build_perm_stable(&e, 16, &(0..16).collect::<Vec<_>>());
        let wg = build_weighted(&e, 16);
        let reference = count(&ug, &Selector::connected_k_subsets(k));
        assert_eq!(sorted(&census_via_iter(&ug, k)), sorted(&reference));
        assert_eq!(sorted(&census_via_iter(&sg, k)), sorted(&reference));
        assert_eq!(sorted(&census_via_iter(&wg, k)), sorted(&reference));
    }
    // Directed graphs are analyzed on their undirected structure: compiles + runs.
    let dg = build_directed(&e, 16);
    let _ = count(&dg, &Selector::connected_k_subsets(3));
}

// ===========================================================================
// [3] streaming: count() peak memory << collect() peak memory.
// ===========================================================================
#[test]
fn count_streams_and_does_not_materialize() {
    let n = 110;
    let g = build_un(&random_edges(n, 0.14, 4), n);
    let sel = Selector::connected_k_subsets(4);

    reset_peak();
    let before = CUR.load(Ordering::Relaxed);
    let census = count(&g, &sel);
    let count_peak = PEAK.load(Ordering::Relaxed).saturating_sub(before);
    let instances = total(&census);

    reset_peak();
    let before2 = CUR.load(Ordering::Relaxed);
    let collected: Vec<_> = enumerate(&g, &sel).collect();
    let collect_peak = PEAK.load(Ordering::Relaxed).saturating_sub(before2);

    assert_eq!(
        instances as usize,
        collected.len(),
        "count and collect must see the same number of instances"
    );
    assert!(
        instances > 50_000,
        "test graph should have many instances (got {instances})"
    );
    assert!(
        collect_peak > count_peak.saturating_mul(4),
        "count peak ({count_peak}) must be far below collect peak ({collect_peak})"
    );
}

// ===========================================================================
// [4] GDV vs independent brute-force per-node oracle.
// ===========================================================================
fn combos(pool: &[usize], r: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    fn rec(
        pool: &[usize],
        r: usize,
        start: usize,
        cur: &mut Vec<usize>,
        out: &mut Vec<Vec<usize>>,
    ) {
        if cur.len() == r {
            out.push(cur.clone());
            return;
        }
        for i in start..pool.len() {
            cur.push(pool[i]);
            rec(pool, r, i + 1, cur, out);
            cur.pop();
        }
    }
    rec(pool, r, 0, &mut Vec::new(), &mut out);
    out
}

fn subset_connected(sub: &[usize], snap: &Snapshot<NodeIndex>) -> bool {
    let m = sub.len();
    let mut local = vec![Vec::new(); m];
    for (i, &a) in sub.iter().enumerate() {
        for (j, &b) in sub.iter().enumerate() {
            if i != j && snap.adjacent(a, b) {
                local[i].push(j);
            }
        }
    }
    connected(&local)
}

// Independent per-node GDV: enumerate every connected k-subset CONTAINING v by
// combinations (not ESU), attribute v's orbit. Different enumeration path.
#[allow(clippy::needless_range_loop)]
fn gdv_oracle(snap: &Snapshot<NodeIndex>, reg: &Registry) -> Vec<Vec<u64>> {
    let n = snap.len();
    let mut gdv = vec![vec![0u64; reg.orbit_count()]; n];
    for v in 0..n {
        let others: Vec<usize> = (0..n).filter(|&x| x != v).collect();
        for k in 2..=5 {
            let ps = perms(k);
            for rest in combos(&others, k - 1) {
                let mut sub = vec![v];
                sub.extend_from_slice(&rest);
                if !subset_connected(&sub, snap) {
                    continue;
                }
                let (class, arg) = canonical_arg_by(k, &ps, |i, j| snap.adjacent(sub[i], sub[j]));
                let slotmap = reg.slot_map(k, class);
                for (c, &slot) in slotmap.iter().enumerate() {
                    if sub[arg[c]] == v {
                        gdv[v][slot] += 1;
                        break;
                    }
                }
            }
        }
    }
    gdv
}

#[test]
fn gdv_matches_bruteforce_oracle() {
    let reg = Registry::build();
    assert_eq!(reg.orbit_count(), 73);

    let mut cases: Vec<(Vec<(usize, usize)>, usize)> = vec![
        (path_edges(6), 6),
        (path_edges(8), 8),
        (cycle_edges(5), 5),
        (cycle_edges(7), 7),
        (star_edges(6), 6),
        (complete_edges(4), 4),
        (complete_edges(5), 5),
        (complete_edges(6), 6),
    ];
    for seed in 0..8u64 {
        let n = 9 + (seed as usize % 4);
        cases.push((random_edges(n, 0.3, seed), n));
    }

    for (edges, n) in &cases {
        let g = build_un(edges, *n);
        let snap = Snapshot::new(&g);
        let table = graphlet_degree_vectors(&g, &reg);
        let oracle = gdv_oracle(&snap, &reg);
        for (v, orow) in oracle.iter().enumerate() {
            assert_eq!(table.row(v), orow.as_slice(), "node {v} GDV mismatch");
        }
    }
}

#[test]
fn gdv_sums_tie_to_class_census() {
    let reg = Registry::build();
    let g = build_un(&random_edges(14, 0.35, 99), 14);
    let table = graphlet_degree_vectors(&g, &reg);

    let mut class_count: HashMap<(usize, u64), u64> = HashMap::new();
    let snap = Snapshot::new(&g);
    for k in 2..=5 {
        let ps = perms(k);
        for_each_subset(&snap, k, |sub| {
            let class = canonical_by(k, &ps, |i, j| snap.adjacent(sub[i], sub[j]));
            *class_count.entry((k, class)).or_insert(0) += 1;
        });
    }
    for o in 0..reg.orbit_count() {
        let (k, class, size) = reg.orbit_meta(o);
        let sum_v: u64 = (0..table.len()).map(|v| table.row(v)[o]).sum();
        let expect = class_count.get(&(k, class)).copied().unwrap_or(0) * size as u64;
        assert_eq!(
            sum_v, expect,
            "orbit {o} sum must equal class_count * orbit_size"
        );
    }
}

// ===========================================================================
// [5] non-induced derivation via s(P,C) vs brute-force monomorphism oracle.
// ===========================================================================
fn class_adj(mask: u64, k: usize) -> Vec<Vec<usize>> {
    class_to_adj(mask, k)
}

// Brute-force labelled monomorphism count of pattern P in host snapshot.
fn mono_labelled(padj: &[Vec<usize>], snap: &Snapshot<NodeIndex>) -> u64 {
    let k = padj.len();
    let n = snap.len();
    fn rec(
        i: usize,
        k: usize,
        padj: &[Vec<usize>],
        n: usize,
        assign: &mut Vec<usize>,
        used: &mut Vec<bool>,
        snap: &Snapshot<NodeIndex>,
    ) -> u64 {
        if i == k {
            return 1;
        }
        let mut total = 0;
        for h in 0..n {
            if used[h] {
                continue;
            }
            let ok = padj[i]
                .iter()
                .filter(|&&nb| nb < i)
                .all(|&nb| snap.adjacent(h, assign[nb]));
            if !ok {
                continue;
            }
            assign[i] = h;
            used[h] = true;
            total += rec(i + 1, k, padj, n, assign, used, snap);
            used[h] = false;
        }
        total
    }
    rec(
        0,
        k,
        padj,
        n,
        &mut vec![usize::MAX; k],
        &mut vec![false; n],
        snap,
    )
}

#[test]
fn non_induced_counts_match_monomorphism_oracle() {
    // Patterns: every connected class at k=3,4 as P (P3, triangle, path/star/cycle/
    // paw/diamond/K4), against structured + fuzzed hosts.
    let mut hosts: Vec<(Vec<(usize, usize)>, usize)> = Vec::new();
    for n in [4usize, 5, 6, 7] {
        hosts.push((path_edges(n), n));
        hosts.push((cycle_edges(n), n));
        hosts.push((star_edges(n), n));
        hosts.push((complete_edges(n), n));
    }
    for seed in 0..12u64 {
        let n = 6 + (seed as usize % 4);
        hosts.push((random_edges(n, 0.35, seed), n));
    }

    for k in 3..=4usize {
        let ps = perms(k);
        for mask in all_connected_classes(k) {
            let padj = class_adj(mask, k);
            // reconstruct a Pattern from this class's edge list
            let edges: Vec<(usize, usize)> = (0..k)
                .flat_map(|i| ((i + 1)..k).map(move |j| (i, j)))
                .filter(|&(i, j)| padj[i].contains(&j))
                .collect();
            let pat = Pattern::new(k, &edges);
            // |Aut(P)| = s(P,P) computed via the same edge-preserving-bijection count
            let aut = {
                let mut c = 0u64;
                for perm in &ps {
                    let ok = (0..k).all(|i| {
                        padj[i]
                            .iter()
                            .filter(|&&j| j > i)
                            .all(|&j| padj[perm[i]].contains(&perm[j]))
                    });
                    if ok {
                        c += 1;
                    }
                }
                c
            };
            for (edges, n) in &hosts {
                if *n < k {
                    continue;
                }
                let g = build_un(edges, *n);
                let snap = Snapshot::new(&g);
                let predicted = count_pattern(&g, &pat, Induced::No);
                let oracle = mono_labelled(&padj, &snap) / aut;
                assert_eq!(
                    predicted, oracle,
                    "non-induced P(mask={mask}) k={k} on host n={n}"
                );
            }
        }
    }
}

// ===========================================================================
// [6] diamond catalog: induced & non-induced instance counts.
// ===========================================================================
#[test]
fn diamond_catalog() {
    let diamond_class = Pattern::diamond().class_id().0;
    let k4_class = canonical_by(4, &perms(4), |i, j| i != j);

    for (name, edges, n) in [
        ("diamond", vec![(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)], 4),
        ("K4", complete_edges(4), 4),
        ("K5", complete_edges(5), 5),
        ("C5", cycle_edges(5), 5),
        ("random", random_edges(10, 0.5, 3), 10),
    ] {
        let g = build_un(&edges, n);
        let census = count(&g, &Selector::connected_k_subsets(4));
        let ind = census.get(&ClassId(diamond_class)).copied().unwrap_or(0);
        let k4 = census.get(&ClassId(k4_class)).copied().unwrap_or(0);

        // induced diamonds
        assert_eq!(
            count_diamonds(&g, Induced::Yes),
            ind,
            "{name} induced count"
        );
        assert_eq!(
            find_diamonds(&g, Induced::Yes).len() as u64,
            ind,
            "{name} induced instances"
        );
        // non-induced: each K4 contributes 6 additional diamonds
        assert_eq!(
            count_diamonds(&g, Induced::No),
            ind + 6 * k4,
            "{name} non-induced count"
        );
        assert_eq!(
            find_diamonds(&g, Induced::No).len() as u64,
            ind + 6 * k4,
            "{name} non-induced instances"
        );
    }

    // Sharp cases: K4 has 0 induced, 6 non-induced diamonds; the diamond has 1/1.
    let k4 = build_un(&complete_edges(4), 4);
    assert_eq!(count_diamonds(&k4, Induced::Yes), 0);
    assert_eq!(count_diamonds(&k4, Induced::No), 6);
    let dia = build_un(&[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)], 4);
    assert_eq!(count_diamonds(&dia, Induced::Yes), 1);
    assert_eq!(count_diamonds(&dia, Induced::No), 1);
}

// ===========================================================================
// PHASE-2 CORRECTNESS HARDENING
// ===========================================================================
//
// Shared independent oracles / builders for the regression, edge-case and
// property tests below.

/// Undirected adjacency matrix (self-loops dropped, parallels deduped) — the
/// independent ground-truth structure the crate is validated against.
fn adj_matrix(edges: &[(usize, usize)], n: usize) -> Vec<Vec<bool>> {
    let mut m = vec![vec![false; n]; n];
    for &(a, b) in edges {
        if a != b {
            m[a][b] = true;
            m[b][a] = true;
        }
    }
    m
}

/// Independent canonical mask (min over k! perms of the packed upper triangle),
/// computed from the adjacency matrix — decorrelated from `canonical_by`.
fn indep_mask(m: &[Vec<bool>], sub: &[usize]) -> u64 {
    let k = sub.len();
    let mut best = u64::MAX;
    for p in &perms(k) {
        let mut mask = 0u64;
        let mut bit = 0;
        for i in 0..k {
            for j in (i + 1)..k {
                if m[sub[p[i]]][sub[p[j]]] {
                    mask |= 1 << bit;
                }
                bit += 1;
            }
        }
        best = best.min(mask);
    }
    best
}

/// Whether `sub` (global indices) induces a connected subgraph of `m`.
fn mat_connected(m: &[Vec<bool>], sub: &[usize]) -> bool {
    let k = sub.len();
    if k == 0 {
        return true;
    }
    let mut seen = vec![false; k];
    let mut stack = vec![0usize];
    seen[0] = true;
    let mut cnt = 1;
    while let Some(x) = stack.pop() {
        for y in 0..k {
            if !seen[y] && m[sub[x]][sub[y]] {
                seen[y] = true;
                cnt += 1;
                stack.push(y);
            }
        }
    }
    cnt == k
}

/// Independent census (canonical-mask -> count) by combination enumeration —
/// a fully decorrelated oracle for streaming `count`.
fn census_oracle(m: &[Vec<bool>], n: usize, k: usize) -> HashMap<u64, u64> {
    let mut out: HashMap<u64, u64> = HashMap::new();
    if k > n {
        return out;
    }
    let pool: Vec<usize> = (0..n).collect();
    for sub in combos(&pool, k) {
        if mat_connected(m, &sub) {
            *out.entry(indep_mask(m, &sub)).or_insert(0) += 1;
        }
    }
    out
}

/// Build a `StableGraph` whose live nodes carry the given undirected edge list but
/// whose raw slot numbering is riddled with holes: a dummy node is interleaved before
/// every real node, then all dummies are removed. This forces `to_index` to diverge
/// from `node_count()` — the exact StableGraph-with-removed-nodes shape.
fn build_stable_with_holes(edges: &[(usize, usize)], n: usize) -> StableGraph<(), (), Undirected> {
    let mut g = StableGraph::<(), (), Undirected>::default();
    let mut real = Vec::with_capacity(n);
    let mut dummies = Vec::with_capacity(n);
    for _ in 0..n {
        dummies.push(g.add_node(()));
        real.push(g.add_node(()));
    }
    for &(a, b) in edges {
        g.add_edge(real[a], real[b], ());
    }
    for d in dummies {
        g.remove_node(d);
    }
    g
}

/// Sorted multiset of GDV rows — node identity is not comparable across relabellings,
/// but the multiset of graphlet-degree vectors is an isomorphism invariant.
fn gdv_row_multiset<N: Copy>(t: &crate::GdvTable<N>) -> Vec<Vec<u64>> {
    let mut rows: Vec<Vec<u64>> = (0..t.len()).map(|i| t.row(i).to_vec()).collect();
    rows.sort();
    rows
}

// ---------------------------------------------------------------------------
// FIX 1 — StableGraph with removed nodes (holes) must not panic and must match
// the equivalent hole-free graph across census / GDV / diamonds, k = 2..=5.
// (Before the node_bound() fix, Snapshot::new panicked with an OOB index.)
// ---------------------------------------------------------------------------
#[test]
fn stablegraph_holes_match_holefree() {
    let reg = Registry::build();
    let cases: Vec<(Vec<(usize, usize)>, usize)> = vec![
        (complete_edges(5), 5),
        (cycle_edges(6), 6),
        (path_edges(7), 7),
        (vec![(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)], 4), // diamond
        (random_edges(9, 0.4, 11), 9),
        (random_edges(10, 0.5, 22), 10),
    ];
    for (edges, n) in &cases {
        let holed = build_stable_with_holes(edges, *n);
        let clean = build_un(edges, *n);

        // Census parity for every supported small order.
        for &k in &[2usize, 3, 4, 5] {
            let sel = Selector::connected_k_subsets(k);
            assert_eq!(
                sorted(&count(&holed, &sel)),
                sorted(&count(&clean, &sel)),
                "holed StableGraph census mismatch n={n} k={k}"
            );
        }

        // GDV parity (as an isomorphism-invariant row multiset).
        assert_eq!(
            gdv_row_multiset(&graphlet_degree_vectors(&holed, &reg)),
            gdv_row_multiset(&graphlet_degree_vectors(&clean, &reg)),
            "holed StableGraph GDV mismatch n={n}"
        );

        // Diamond parity (counts; node identities differ across the two graphs).
        for ind in [Induced::Yes, Induced::No] {
            assert_eq!(
                count_diamonds(&holed, ind),
                count_diamonds(&clean, ind),
                "holed StableGraph diamond count mismatch n={n} {ind:?}"
            );
            assert_eq!(
                find_diamonds(&holed, ind).len(),
                find_diamonds(&clean, ind).len(),
                "holed StableGraph diamond instances mismatch n={n} {ind:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// FIX 2 — Selector k-range guard enforced at the public boundary (private field
// + checked constructor). k in {0,1} is degenerate; k >= 12 overflows the u64
// canonical mask (silent corruption before the guard).
// ---------------------------------------------------------------------------
#[test]
#[should_panic(expected = "graphlet order k must be in 2")]
fn selector_rejects_k0() {
    let _ = Selector::connected_k_subsets(0);
}

#[test]
#[should_panic(expected = "graphlet order k must be in 2")]
fn selector_rejects_k1() {
    let _ = Selector::connected_k_subsets(1);
}

#[test]
#[should_panic(expected = "graphlet order k must be in 2")]
fn selector_rejects_k12_mask_overflow() {
    // k=12 packs 66 upper-triangle bits into a u64 => overflow. Must be rejected,
    // not silently wrapped.
    let _ = Selector::connected_k_subsets(12);
}

#[test]
fn selector_boundary_k_values() {
    // Lower and upper in-range boundaries construct cleanly and expose their order.
    assert_eq!(Selector::connected_k_subsets(2).k(), 2);
    assert_eq!(
        Selector::connected_k_subsets(crate::census::MAX_K).k(),
        crate::census::MAX_K
    );
    // A modest above-5 order (k=6) still enumerates without mask corruption:
    // K6 has exactly one connected class (K6 itself) counted once.
    let g = build_un(&complete_edges(6), 6);
    let census = count(&g, &Selector::connected_k_subsets(6));
    assert_eq!(census.values().sum::<u64>(), 1, "K6 has a single 6-subset");
}

// ---------------------------------------------------------------------------
// FIX 3 — inputs are treated as simple graphs: self-loops stripped, parallel
// edges deduped. Pattern::new instead REJECTS self-loop pattern edges. Pin both.
// ---------------------------------------------------------------------------
#[test]
fn self_loops_and_parallel_edges_normalized() {
    // A clean triangle: one triangle class instance at k=3.
    let clean = build_un(&[(0, 1), (1, 2), (2, 0)], 3);
    let clean_census = sorted(&count(&clean, &Selector::connected_k_subsets(3)));

    // Triangle + a self-loop on vertex 0 => identical census (self-loop stripped).
    let with_loop = build_un(&[(0, 1), (1, 2), (2, 0), (0, 0)], 3);
    assert_eq!(
        sorted(&count(&with_loop, &Selector::connected_k_subsets(3))),
        clean_census,
        "self-loop must be stripped"
    );

    // Triangle with tripled + reciprocal-directed-style parallel edges => identical.
    let with_parallels = build_un(&[(0, 1), (0, 1), (1, 0), (1, 2), (1, 2), (2, 0)], 3);
    assert_eq!(
        sorted(&count(&with_parallels, &Selector::connected_k_subsets(3))),
        clean_census,
        "parallel edges must be deduped"
    );

    // A directed reciprocal pair collapses to one undirected edge (count k=2 == 1).
    let dg = build_directed(&[(0, 1), (1, 0)], 2);
    assert_eq!(
        count(&dg, &Selector::connected_k_subsets(2))
            .values()
            .sum::<u64>(),
        1,
        "directed reciprocal edges collapse to one"
    );
}

#[test]
#[should_panic(expected = "self-loop")]
fn pattern_rejects_self_loop() {
    // The pattern side is asymmetric with the host side: it rejects self-loops.
    let _ = Pattern::new(3, &[(0, 1), (1, 2), (2, 2)]);
}

// ---------------------------------------------------------------------------
// FIX 4 — template arm returns RAW VF2 embeddings (each embedding, including
// pattern automorphisms; no node-set dedup). Pin the documented counts.
// ---------------------------------------------------------------------------
#[test]
fn template_counts_raw_embeddings() {
    use crate::template::count_induced_matches;

    // P3 (path 0-1-2) is NOT induced in a triangle host => 0 embeddings.
    let p3 = build_un(&[(0, 1), (1, 2)], 3);
    let triangle = build_un(&[(0, 1), (1, 2), (2, 0)], 3);
    assert_eq!(
        count_induced_matches(&p3, &triangle),
        0,
        "P3 not induced in K3"
    );

    // P3 on a host path a-b-c: one node-set {a,b,c}, but 2 embeddings (forward and
    // reversed) — automorphisms are NOT deduped.
    let host_path = build_un(&[(0, 1), (1, 2)], 3);
    assert_eq!(
        count_induced_matches(&p3, &host_path),
        2,
        "P3 yields |Aut(P3)|=2 raw embeddings over one node-set"
    );

    // Triangle pattern on a triangle host: 3! = 6 embeddings (|Aut(K3)| = 6).
    assert_eq!(
        count_induced_matches(&triangle, &triangle),
        6,
        "K3 in K3 yields |Aut(K3)|=6 raw embeddings"
    );

    // Two disjoint host triangles: still 6 embeddings each => 12 total, and each is a
    // raw embedding (no node-set dedup), demonstrating the template semantics.
    let two_tri = build_un(&[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)], 6);
    assert_eq!(count_induced_matches(&triangle, &two_tri), 12);
}

// ---------------------------------------------------------------------------
// HARDEN — explicit edge cases across all public entry points.
// ---------------------------------------------------------------------------
#[test]
fn edge_cases_across_entry_points() {
    let reg = Registry::build();

    // Empty graph (n = 0).
    let empty = build_un(&[], 0);
    for &k in &[2usize, 3, 4, 5] {
        assert!(count(&empty, &Selector::connected_k_subsets(k)).is_empty());
        assert_eq!(
            enumerate(&empty, &Selector::connected_k_subsets(k)).count(),
            0
        );
    }
    assert!(graphlet_degree_vectors(&empty, &reg).is_empty());
    assert_eq!(count_diamonds(&empty, Induced::Yes), 0);
    assert!(find_diamonds(&empty, Induced::No).is_empty());

    // Single isolated node.
    let single = build_un(&[], 1);
    assert!(count(&single, &Selector::connected_k_subsets(2)).is_empty());
    assert_eq!(graphlet_degree_vectors(&single, &reg).len(), 1);
    assert!(graphlet_degree_vectors(&single, &reg)
        .row(0)
        .iter()
        .all(|&x| x == 0));

    // Fewer than k nodes.
    let two = build_un(&[(0, 1)], 2);
    assert!(count(&two, &Selector::connected_k_subsets(5)).is_empty());
    // Exactly k: a single edge is the only 2-graphlet.
    assert_eq!(
        count(&two, &Selector::connected_k_subsets(2))
            .values()
            .sum::<u64>(),
        1
    );

    // Disconnected multi-component: two triangles, k=3 => two triangle instances, no
    // cross-component subset.
    let two_tri = build_un(&[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)], 6);
    assert_eq!(
        count(&two_tri, &Selector::connected_k_subsets(3))
            .values()
            .sum::<u64>(),
        2
    );

    // Directed input analyzed as undirected: directed triangle census == undirected.
    let dtri = build_directed(&[(0, 1), (1, 2), (2, 0)], 3);
    let utri = build_un(&[(0, 1), (1, 2), (2, 0)], 3);
    assert_eq!(
        sorted(&count(&dtri, &Selector::connected_k_subsets(3))),
        sorted(&count(&utri, &Selector::connected_k_subsets(3)))
    );

    // StableGraph with holes but empty edge set: no panic, empty census.
    let holed_empty = build_stable_with_holes(&[], 3);
    assert!(count(&holed_empty, &Selector::connected_k_subsets(2)).is_empty());
}

// ===========================================================================
// PROPERTY-BASED DIFFERENTIAL FUZZING (proptest).
// ===========================================================================

/// Build an undirected edge list from a bit vector over the `C(n,2)` vertex pairs.
fn edges_from_bits(n: usize, bits: &[bool]) -> Vec<(usize, usize)> {
    let mut e = Vec::new();
    let mut idx = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            if bits.get(idx).copied().unwrap_or(false) {
                e.push((i, j));
            }
            idx += 1;
        }
    }
    e
}

/// Strategy: (node count 0..=7, edge-presence bits for up to C(7,2)=21 pairs,
/// a relabelling seed).
fn graph_strategy() -> impl Strategy<Value = (usize, Vec<bool>, u64)> {
    (0usize..=7).prop_flat_map(|n| {
        let pairs = n * n.saturating_sub(1) / 2;
        (
            Just(n),
            proptest::collection::vec(any::<bool>(), pairs),
            any::<u64>(),
        )
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(160))]

    // (a) streaming count == recursive-driven count == enumerate() grouped by class,
    //     and all == the independent combination-enumeration oracle. Also (b) the
    //     class set is always a subset of the connected ground-truth classes, and (d)
    //     the census is invariant under relabelling (Graph & StableGraph-with-holes).
    #[test]
    fn prop_census_differential((n, bits, seed) in graph_strategy()) {
        let edges = edges_from_bits(n, &bits);
        let m = adj_matrix(&edges, n);
        let g = build_un(&edges, n);

        // Random relabelling + the two graph flavours that carry hole risk.
        let mut order: Vec<usize> = (0..n).collect();
        order.shuffle(&mut StdRng::seed_from_u64(seed));
        let gp = build_perm_un(&edges, n, &order);
        let holed = build_stable_with_holes(&edges, n);

        for k in 2..=5usize {
            let sel = Selector::connected_k_subsets(k);
            let oracle = census_oracle(&m, n, k);

            // count() re-keyed to bare masks.
            let by_mask: HashMap<u64, u64> = count(&g, &sel)
                .into_iter()
                .map(|(c, v)| (c.0, v))
                .collect();
            prop_assert_eq!(&by_mask, &oracle, "count vs oracle n={} k={}", n, k);

            // enumerate() grouped by ClassId, re-keyed to masks.
            let by_iter: HashMap<u64, u64> = census_via_iter(&g, k)
                .into_iter()
                .map(|(c, v)| (c.0, v))
                .collect();
            prop_assert_eq!(&by_iter, &oracle, "enumerate vs oracle n={} k={}", n, k);

            // (b) only connected ground-truth classes ever appear.
            let gt: HashSet<u64> = all_connected_classes(k).into_iter().collect();
            for mask in by_mask.keys() {
                prop_assert!(gt.contains(mask), "spurious class {} k={}", mask, k);
            }

            // (d) relabelling invariance across Graph and holed StableGraph.
            prop_assert_eq!(sorted(&count(&gp, &sel)), sorted(&count(&g, &sel)));
            prop_assert_eq!(sorted(&count(&holed, &sel)), sorted(&count(&g, &sel)));
        }
    }

    // (c) GDV == brute-force per-node GDV oracle, and Σ_v GDV[o] == class_count *
    //     orbit_size for all 73 orbits; plus holed-StableGraph GDV row-multiset parity.
    #[test]
    fn prop_gdv_differential((n, bits, _seed) in graph_strategy()) {
        let reg = Registry::build();
        let edges = edges_from_bits(n, &bits);
        let g = build_un(&edges, n);
        let snap = Snapshot::new(&g);
        let table = graphlet_degree_vectors(&g, &reg);

        // (c1) per-node GDV equals the independent combination-enumeration oracle.
        let oracle = gdv_oracle(&snap, &reg);
        for (v, orow) in oracle.iter().enumerate() {
            prop_assert_eq!(table.row(v), orow.as_slice(), "GDV mismatch node {}", v);
        }

        // (c2) Σ_v GDV[o] == class_count(k,class) * orbit_size for every orbit.
        let mut class_count: HashMap<(usize, u64), u64> = HashMap::new();
        for k in 2..=5 {
            let ps = perms(k);
            for_each_subset(&snap, k, |sub| {
                let class = canonical_by(k, &ps, |i, j| snap.adjacent(sub[i], sub[j]));
                *class_count.entry((k, class)).or_insert(0) += 1;
            });
        }
        for o in 0..reg.orbit_count() {
            let (k, class, size) = reg.orbit_meta(o);
            let sum_v: u64 = (0..table.len()).map(|v| table.row(v)[o]).sum();
            let expect = class_count.get(&(k, class)).copied().unwrap_or(0) * size as u64;
            prop_assert_eq!(sum_v, expect, "orbit {} sum mismatch", o);
        }

        // GDV row-multiset parity with the holed StableGraph.
        let holed = build_stable_with_holes(&edges, n);
        prop_assert_eq!(
            gdv_row_multiset(&graphlet_degree_vectors(&holed, &reg)),
            gdv_row_multiset(&table)
        );
    }

    // (e) s(P,C) non-induced count == labelled-monomorphism oracle / |Aut(P)| for
    //     every connected pattern P at k = 3,4,5.
    #[test]
    fn prop_non_induced_vs_monomorphism((n, bits, _seed) in graph_strategy()) {
        let edges = edges_from_bits(n, &bits);
        let g = build_un(&edges, n);
        let snap = Snapshot::new(&g);
        for k in 3..=5usize {
            if n < k {
                continue;
            }
            let ps = perms(k);
            for mask in all_connected_classes(k) {
                let padj = class_adj(mask, k);
                let pedges: Vec<(usize, usize)> = (0..k)
                    .flat_map(|i| ((i + 1)..k).map(move |j| (i, j)))
                    .filter(|&(i, j)| padj[i].contains(&j))
                    .collect();
                let pat = Pattern::new(k, &pedges);
                let aut = {
                    let mut c = 0u64;
                    for perm in &ps {
                        let ok = (0..k).all(|i| {
                            padj[i]
                                .iter()
                                .filter(|&&j| j > i)
                                .all(|&j| padj[perm[i]].contains(&perm[j]))
                        });
                        if ok {
                            c += 1;
                        }
                    }
                    c
                };
                let predicted = count_pattern(&g, &pat, Induced::No);
                let oracle = mono_labelled(&padj, &snap) / aut;
                prop_assert_eq!(predicted, oracle, "non-induced mask={} k={}", mask, k);
            }
        }
    }

    // (f) template induced-native count / |Aut(P)| == the induced census count of the
    //     pattern's class (links the VF2 arm to the census arm).
    #[test]
    fn prop_template_matches_induced_census((n, bits, _seed) in graph_strategy()) {
        use crate::template::count_induced_matches;
        let edges = edges_from_bits(n, &bits);
        let g = build_un(&edges, n);

        // A fixed battery of small connected patterns (k = 3,4).
        let patterns: [(usize, Vec<(usize, usize)>); 5] = [
            (3, vec![(0, 1), (1, 2)]),                          // P3
            (3, vec![(0, 1), (1, 2), (2, 0)]),                  // triangle
            (4, vec![(0, 1), (1, 2), (2, 3)]),                  // P4
            (4, vec![(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]),  // diamond
            (4, complete_edges(4)),                             // K4
        ];
        for (k, pedges) in &patterns {
            if n < *k {
                continue;
            }
            let pat_graph = build_un(pedges, *k);
            let pat = Pattern::new(*k, pedges);
            // |Aut(P)| = s(P,P) via the crate's own class self-count.
            let ps = perms(*k);
            let padj = {
                let mut a = vec![Vec::new(); *k];
                for &(x, y) in pedges {
                    if !a[x].contains(&y) {
                        a[x].push(y);
                        a[y].push(x);
                    }
                }
                a
            };
            let aut = ps
                .iter()
                .filter(|perm| {
                    (0..*k).all(|i| {
                        padj[i]
                            .iter()
                            .filter(|&&j| j > i)
                            .all(|&j| padj[perm[i]].contains(&perm[j]))
                    })
                })
                .count() as u64;

            let raw = count_induced_matches(&pat_graph, &g) as u64;
            let census = count(&g, &Selector::connected_k_subsets(*k));
            let induced_nodesets = census.get(&pat.class_id()).copied().unwrap_or(0);
            prop_assert_eq!(
                raw / aut,
                induced_nodesets,
                "template raw/aut vs induced census k={}",
                k
            );
            prop_assert_eq!(raw % aut, 0, "raw embeddings must be a multiple of |Aut|");
        }
    }
}

// ===========================================================================
// PHASE-1 MOTIF ENGINE: monomorphism enumerator + general motif catalog.
// ===========================================================================
//
// Independent brute-force oracles (all-injective-map enumeration over adjacency
// matrices) — fully decorrelated from the ordered-backtracking enumerator and from
// petgraph VF2.

/// Enumerate every injective map `0..pk -> 0..nh` and hand each to `f`.
fn injective_maps(pk: usize, nh: usize, mut f: impl FnMut(&[usize])) {
    fn rec(
        i: usize,
        pk: usize,
        nh: usize,
        a: &mut Vec<usize>,
        u: &mut [bool],
        f: &mut impl FnMut(&[usize]),
    ) {
        if i == pk {
            f(a);
            return;
        }
        for h in 0..nh {
            if u[h] {
                continue;
            }
            a[i] = h;
            u[h] = true;
            rec(i + 1, pk, nh, a, u, f);
            u[h] = false;
        }
    }
    if pk > nh {
        return; // no injection possible
    }
    rec(0, pk, nh, &mut vec![0; pk], &mut vec![false; nh], &mut f);
}

/// Directed adjacency matrix: `m[a][b]` iff edge `a -> b` (self-loops dropped).
fn dir_matrix(edges: &[(usize, usize)], n: usize) -> Vec<Vec<bool>> {
    let mut m = vec![vec![false; n]; n];
    for &(a, b) in edges {
        if a != b {
            m[a][b] = true;
        }
    }
    m
}

/// Brute-force labelled monomorphism count: injective maps where every (directed)
/// pattern edge lands on a host edge. `hm` is the host adjacency matrix (symmetric for
/// undirected hosts, directed otherwise); `pedges` are the pattern's directed pairs.
fn bf_mono(pedges: &[(usize, usize)], pk: usize, hm: &[Vec<bool>], nh: usize) -> usize {
    let mut c = 0usize;
    injective_maps(pk, nh, |map| {
        if pedges.iter().all(|&(i, j)| hm[map[i]][map[j]]) {
            c += 1;
        }
    });
    c
}

/// Brute-force labelled induced (edge-reflecting) subgraph-isomorphism count: injective
/// maps where an ordered image pair is a host edge *iff* the pattern pair is an edge.
fn bf_induced(pm: &[Vec<bool>], pk: usize, hm: &[Vec<bool>], nh: usize) -> usize {
    let mut c = 0usize;
    injective_maps(pk, nh, |map| {
        let ok = (0..pk).all(|i| (0..pk).all(|j| i == j || pm[i][j] == hm[map[i]][map[j]]));
        if ok {
            c += 1;
        }
    });
    c
}

/// A battery of small connected undirected patterns as (k, edges).
fn undirected_pattern_battery() -> Vec<(usize, Vec<(usize, usize)>)> {
    vec![
        (2, vec![(0, 1)]),                                 // edge
        (3, vec![(0, 1), (1, 2)]),                         // P3
        (3, complete_edges(3)),                            // triangle
        (4, vec![(0, 1), (1, 2), (2, 3)]),                 // P4
        (4, star_edges(4)),                                // claw
        (4, cycle_edges(4)),                               // C4
        (4, vec![(0, 1), (1, 2), (2, 0), (0, 3)]),         // paw
        (4, vec![(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]), // diamond
        (4, complete_edges(4)),                            // K4
        (5, path_edges(5)),                                // P5
        (5, cycle_edges(5)),                               // C5
    ]
}

/// Structured + fuzzed undirected hosts as (edges, n).
fn undirected_host_battery() -> Vec<(Vec<(usize, usize)>, usize)> {
    let mut hosts: Vec<(Vec<(usize, usize)>, usize)> = Vec::new();
    for n in [4usize, 5, 6, 7] {
        hosts.push((path_edges(n), n));
        hosts.push((cycle_edges(n), n));
        hosts.push((star_edges(n), n));
        hosts.push((complete_edges(n), n));
    }
    // Petersen graph (3-regular, 10 nodes).
    hosts.push((
        vec![
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 0), // outer C5
            (5, 7),
            (7, 9),
            (9, 6),
            (6, 8),
            (8, 5), // inner pentagram
            (0, 5),
            (1, 6),
            (2, 7),
            (3, 8),
            (4, 9), // spokes
        ],
        10,
    ));
    // 3-cube Q3 (bipartite, 8 nodes).
    hosts.push((
        vec![
            (0, 1),
            (1, 3),
            (3, 2),
            (2, 0), // bottom face
            (4, 5),
            (5, 7),
            (7, 6),
            (6, 4), // top face
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7), // verticals
        ],
        8,
    ));
    // Complete bipartite K_{2,3}.
    hosts.push((vec![(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)], 5));
    for seed in 0..12u64 {
        let n = 6 + (seed as usize % 4);
        hosts.push((random_edges(n, 0.35, seed), n));
    }
    hosts
}

// ---------------------------------------------------------------------------
// [P1-a] Monomorphism enumerator (undirected) vs brute-force oracle, over the
// adversarial + fuzzed host battery and the small-pattern battery. Also pins the
// instance/count agreement and validity of every returned embedding.
// ---------------------------------------------------------------------------
#[test]
fn monomorphism_enumerator_vs_bruteforce_undirected() {
    let patterns = undirected_pattern_battery();
    let hosts = undirected_host_battery();
    for (pk, pedges) in &patterns {
        let pat = build_un(pedges, *pk);
        for (hedges, n) in &hosts {
            let host = build_un(hedges, *n);
            let hm = adj_matrix(hedges, *n);
            let oracle = bf_mono(pedges, *pk, &hm, *n);

            let insts = monomorphisms_unlabelled(&pat, &host);
            let cnt = count_monomorphisms_unlabelled(&pat, &host);
            assert_eq!(cnt, oracle, "count k={pk} host n={n}");
            assert_eq!(insts.len(), oracle, "instances k={pk} host n={n}");

            // Every returned embedding is a valid injective edge-preserving map.
            for e in &insts {
                assert_eq!(e.len(), *pk);
                let uniq: HashSet<usize> = e.iter().copied().collect();
                assert_eq!(uniq.len(), *pk, "embedding must be injective");
                for &(i, j) in pedges {
                    assert!(
                        hm[e[i]][e[j]],
                        "embedding must preserve pattern edge ({i},{j})"
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// [P1-b] Monomorphism enumerator (DIRECTED) vs brute-force oracle. Directedness is
// honoured: a pattern arc must land on a host arc in the matching direction.
// ---------------------------------------------------------------------------
#[test]
fn monomorphism_enumerator_vs_bruteforce_directed() {
    // Directed patterns (arcs) as (k, arcs).
    let patterns: Vec<(usize, Vec<(usize, usize)>)> = vec![
        (2, vec![(0, 1)]),                         // single arc
        (3, vec![(0, 1), (1, 2)]),                 // directed P3
        (3, vec![(0, 1), (1, 2), (0, 2)]),         // feed-forward loop (FFL)
        (3, vec![(0, 1), (1, 2), (2, 0)]),         // directed 3-cycle
        (4, vec![(0, 1), (0, 2), (3, 1), (3, 2)]), // bi-fan
    ];
    let mut hosts: Vec<(Vec<(usize, usize)>, usize)> = vec![
        (vec![(0, 1), (1, 2), (0, 2)], 3),               // one FFL
        (vec![(0, 1), (1, 2), (2, 0)], 3),               // directed triangle
        (vec![(0, 1), (0, 2), (3, 1), (3, 2)], 4),       // one bi-fan
        ((0..5).map(|i| (i, (i + 1) % 5)).collect(), 5), // directed C5
    ];
    for seed in 0..14u64 {
        let n = 5 + (seed as usize % 3);
        // Random directed edges (ordered pairs, both directions independently possible).
        let mut rng = StdRng::seed_from_u64(seed + 100);
        let mut e = Vec::new();
        for a in 0..n {
            for b in 0..n {
                if a != b && rng.gen::<f64>() < 0.3 {
                    e.push((a, b));
                }
            }
        }
        hosts.push((e, n));
    }

    for (pk, parcs) in &patterns {
        let pat = build_directed(parcs, *pk);
        for (harcs, n) in &hosts {
            let host = build_directed(harcs, *n);
            let hm = dir_matrix(harcs, *n);
            let oracle = bf_mono(parcs, *pk, &hm, *n);
            let cnt = count_monomorphisms_unlabelled(&pat, &host);
            let insts = monomorphisms_unlabelled(&pat, &host);
            assert_eq!(cnt, oracle, "directed count k={pk} host n={n}");
            assert_eq!(insts.len(), oracle, "directed instances k={pk} host n={n}");
            for e in &insts {
                for &(i, j) in parcs {
                    assert!(
                        hm[e[i]][e[j]],
                        "directed embedding must preserve arc ({i},{j})"
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// [P1-c] Node / edge match predicates gate the monomorphism search.
// ---------------------------------------------------------------------------
#[test]
fn monomorphism_predicates_gate_matches() {
    fn build_labeled_dir(
        nodes: &[char],
        edges: &[(usize, usize, i32)],
    ) -> Graph<char, i32, Directed> {
        let mut g = Graph::<char, i32, Directed>::new();
        let idx: Vec<_> = nodes.iter().map(|&c| g.add_node(c)).collect();
        for &(a, b, w) in edges {
            g.add_edge(idx[a], idx[b], w);
        }
        g
    }

    // Host: directed path x -10-> y -20-> x, node weights [x, y, x].
    let host = build_labeled_dir(&['x', 'y', 'x'], &[(0, 1, 10), (1, 2, 20)]);
    // Pattern: single arc a -10-> b, node weights [x, y].
    let pat = build_labeled_dir(&['x', 'y'], &[(0, 1, 10)]);

    // Structure only: pattern arc 0->1 with p0 in {any}, p1 in {any}; host arcs 0->1,1->2.
    // 2 injective arc placements (0->1 and 1->2).
    assert_eq!(
        count_monomorphisms(&pat, &host, |_, _| true, |_, _| true),
        2
    );

    // Node match (labels equal): p0='x' -> host 'x' (0 or 2), p1='y' -> host 'y' (1).
    // Host arcs into node 1 from an 'x': only 0->1. So exactly 1.
    let node_eq = |p: &char, h: &char| p == h;
    assert_eq!(count_monomorphisms(&pat, &host, node_eq, |_, _| true), 1);

    // Edge match too (weight 10): 0->1 has weight 10 => still 1.
    let edge_eq = |p: &i32, h: &i32| p == h;
    assert_eq!(count_monomorphisms(&pat, &host, node_eq, edge_eq), 1);

    // Edge match requiring weight 20 on a weight-10 pattern arc => 0.
    let pat20 = build_labeled_dir(&['x', 'y'], &[(0, 1, 20)]);
    assert_eq!(count_monomorphisms(&pat20, &host, node_eq, edge_eq), 0);

    // The returned embedding under full matching is the arc 0->1.
    let insts = monomorphisms(&pat, &host, node_eq, edge_eq);
    assert_eq!(insts, vec![vec![0, 1]]);
}

// ---------------------------------------------------------------------------
// [P1-d] CROSS-CHECK: for connected patterns at k <= 5, the monomorphism
// enumerator's non-induced count / |Aut(P)| equals the verified s(P,C)-derived
// count_pattern(Induced::No), and the raw count equals the census-independent
// labelled-monomorphism oracle.
// ---------------------------------------------------------------------------
#[test]
fn monomorphism_cross_check_spc() {
    let hosts = undirected_host_battery();
    for k in 3..=5usize {
        let ps = perms(k);
        for mask in all_connected_classes(k) {
            let padj = class_adj(mask, k);
            let pedges: Vec<(usize, usize)> = (0..k)
                .flat_map(|i| ((i + 1)..k).map(move |j| (i, j)))
                .filter(|&(i, j)| padj[i].contains(&j))
                .collect();
            let pat = Pattern::new(k, &pedges);
            let pat_g = build_un(&pedges, k);
            let aut = ps
                .iter()
                .filter(|perm| {
                    (0..k).all(|i| {
                        padj[i]
                            .iter()
                            .filter(|&&j| j > i)
                            .all(|&j| padj[perm[i]].contains(&perm[j]))
                    })
                })
                .count() as u64;
            for (hedges, n) in &hosts {
                if *n < k {
                    continue;
                }
                let host = build_un(hedges, *n);
                let hm = adj_matrix(hedges, *n);
                let raw = count_monomorphisms_unlabelled(&pat_g, &host) as u64;
                // raw == independent labelled-monomorphism oracle.
                assert_eq!(raw, bf_mono(&pedges, k, &hm, *n) as u64, "raw vs oracle");
                // raw / |Aut(P)| == verified s(P,C) fast-path count.
                assert_eq!(
                    raw / aut,
                    count_pattern(&host, &pat, Induced::No),
                    "enumerator vs s(P,C) mask={mask} k={k} n={n}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// [P1-e] Induced arm: petgraph VF2 count vs an independent edge-reflecting
// brute-force oracle (cross-checks the VF2 delegation too).
// ---------------------------------------------------------------------------
#[test]
fn induced_matches_vs_bruteforce() {
    let patterns = undirected_pattern_battery();
    let hosts = undirected_host_battery();
    for (pk, pedges) in &patterns {
        let pat = build_un(pedges, *pk);
        let pm = adj_matrix(pedges, *pk);
        for (hedges, n) in &hosts {
            let host = build_un(hedges, *n);
            let hm = adj_matrix(hedges, *n);
            assert_eq!(
                count_induced_matches(&pat, &host),
                bf_induced(&pm, *pk, &hm, *n),
                "induced VF2 vs oracle k={pk} n={n}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// [P1-f] find_motif returns distinct occurrences whose count == count_motif, for
// both semantics, across named motifs and the adversarial host battery. Also pins
// each returned mapping as a valid, distinct-node occurrence.
// ---------------------------------------------------------------------------
#[test]
fn find_motif_matches_count_and_is_valid() {
    let named: Vec<(usize, Vec<(usize, usize)>)> = undirected_pattern_battery();
    let hosts = undirected_host_battery();
    for (pk, pedges) in &named {
        let pat = Pattern::new(*pk, pedges);
        for (hedges, n) in &hosts {
            let g = build_un(hedges, *n);
            let hm = adj_matrix(hedges, *n);
            for ind in [Induced::Yes, Induced::No] {
                let insts = find_motif(&g, &pat, ind);
                let cnt = count_motif(&g, &pat, ind);
                assert_eq!(
                    insts.len() as u64,
                    cnt,
                    "find vs count k={pk} n={n} {ind:?}"
                );

                let mut nodesets: HashSet<Vec<usize>> = HashSet::new();
                for m in &insts {
                    let idxs: Vec<usize> = m.iter().map(|nid| nid.index()).collect();
                    let uniq: HashSet<usize> = idxs.iter().copied().collect();
                    assert_eq!(uniq.len(), *pk, "occurrence nodes distinct");
                    // Every pattern edge is present in the host (both semantics).
                    for &(i, j) in pedges {
                        assert!(hm[idxs[i]][idxs[j]], "occurrence preserves edge");
                    }
                    if ind == Induced::Yes {
                        // Induced: at most one representative per node-set.
                        let mut s = idxs.clone();
                        s.sort_unstable();
                        assert!(nodesets.insert(s), "induced occurrences distinct node-sets");
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// [P1-g] Named-motif constructors and the MotifCatalog registry.
// ---------------------------------------------------------------------------
#[test]
fn named_motifs_and_catalog() {
    // Orders.
    assert_eq!(Pattern::path(5).order(), 5);
    assert_eq!(Pattern::cycle(4).order(), 4);
    assert_eq!(Pattern::star(5).order(), 5);
    assert_eq!(Pattern::complete(4).order(), 4);
    assert_eq!(Pattern::paw().order(), 4);

    // Aliases coincide as classes.
    assert_eq!(
        Pattern::triangle().class_id(),
        Pattern::complete(3).class_id()
    );
    assert_eq!(Pattern::claw().class_id(), Pattern::star(4).class_id());
    assert_eq!(
        Pattern::diamond().class_id(),
        Pattern::new(4, &[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]).class_id()
    );

    // The six connected 4-node motifs are pairwise distinct classes.
    let four: Vec<ClassId> = [
        Pattern::path(4),
        Pattern::claw(),
        Pattern::cycle(4),
        Pattern::paw(),
        Pattern::diamond(),
        Pattern::complete(4),
    ]
    .iter()
    .map(Pattern::class_id)
    .collect();
    let uniq: HashSet<ClassId> = four.iter().copied().collect();
    assert_eq!(uniq.len(), 6, "the six 4-node motifs are distinct classes");

    // Structural sanity: a single C4 host has exactly one induced 4-cycle, zero K4.
    let c4 = build_un(&cycle_edges(4), 4);
    assert_eq!(count_motif(&c4, &Pattern::cycle(4), Induced::Yes), 1);
    assert_eq!(count_motif(&c4, &Pattern::complete(4), Induced::Yes), 0);
    // Star S5 host: exactly one induced claw around the center choosing 3 of 4 leaves? No —
    // K_{1,4} has C(4,3)=4 induced claws.
    let s5 = build_un(&star_edges(5), 5);
    assert_eq!(count_motif(&s5, &Pattern::claw(), Induced::Yes), 4);

    // Registry.
    let mut cat = MotifCatalog::standard();
    assert_eq!(cat.len(), 12);
    assert!(cat.get("diamond").is_some());
    assert!(cat.get("nope").is_none());
    assert_eq!(
        cat.get("triangle").unwrap().class_id(),
        Pattern::triangle().class_id()
    );
    let prev = cat.register("mine", Pattern::cycle(5));
    assert!(prev.is_none());
    assert_eq!(cat.len(), 13);
    assert_eq!(
        cat.get("mine").unwrap().class_id(),
        Pattern::cycle(5).class_id()
    );
    // names() sorted and contains registered entries.
    let names = cat.names();
    assert!(names.windows(2).all(|w| w[0] <= w[1]), "names sorted");
    assert!(names.contains(&"mine") && names.contains(&"k5"));

    let empty = MotifCatalog::new();
    assert!(empty.is_empty());
}

// ---------------------------------------------------------------------------
// [P1-h] Motif-engine edge cases: pattern larger than host, empty / single-node /
// disconnected hosts, across the enumerator and find_motif.
// ---------------------------------------------------------------------------
#[test]
fn motif_engine_edge_cases() {
    // Pattern larger than host: no injection possible.
    let k4_pat = build_un(&complete_edges(4), 4);
    let triangle = build_un(&complete_edges(3), 3);
    assert_eq!(count_monomorphisms_unlabelled(&k4_pat, &triangle), 0);
    assert!(monomorphisms_unlabelled(&k4_pat, &triangle).is_empty());
    let dia = Pattern::diamond();
    let tri_host = build_un(&complete_edges(3), 3);
    assert!(find_motif(&tri_host, &dia, Induced::No).is_empty());
    assert_eq!(count_motif(&tri_host, &dia, Induced::Yes), 0);

    // Empty host.
    let empty: UnGraph<(), ()> = build_un(&[], 0);
    let p3 = build_un(&[(0, 1), (1, 2)], 3);
    assert_eq!(count_monomorphisms_unlabelled(&p3, &empty), 0);
    assert!(find_motif(&empty, &Pattern::path(3), Induced::No).is_empty());

    // Single isolated node vs an edge pattern.
    let single = build_un(&[], 1);
    let edge = build_un(&[(0, 1)], 2);
    assert_eq!(count_monomorphisms_unlabelled(&edge, &single), 0);

    // Disconnected host: two triangles. A triangle pattern has 2 distinct occurrences
    // (one per component) and 12 raw monomorphisms (6 per component).
    let two_tri = build_un(&[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)], 6);
    let tri_pat = build_un(&complete_edges(3), 3);
    assert_eq!(count_monomorphisms_unlabelled(&tri_pat, &two_tri), 12);
    assert_eq!(
        find_motif(&two_tri, &Pattern::triangle(), Induced::Yes).len(),
        2
    );
    assert_eq!(
        find_motif(&two_tri, &Pattern::triangle(), Induced::No).len(),
        2
    );

    // Self-loop policy: the template arm honours a literal host self-loop (unlike the
    // census, which strips them). A self-loop pattern needs a host self-loop.
    let mut loop_pat = Graph::<(), (), Undirected>::new_undirected();
    let lp = loop_pat.add_node(());
    loop_pat.add_edge(lp, lp, ());
    let mut host_loop = Graph::<(), (), Undirected>::new_undirected();
    let a = host_loop.add_node(());
    let b = host_loop.add_node(());
    host_loop.add_edge(a, a, ());
    host_loop.add_edge(a, b, ());
    // The single-vertex-with-loop pattern maps only to the looped host vertex.
    assert_eq!(count_monomorphisms_unlabelled(&loop_pat, &host_loop), 1);
    assert_eq!(
        monomorphisms_unlabelled(&loop_pat, &host_loop),
        vec![vec![0]]
    );
}

// ===========================================================================
// PHASE-1 PROPERTY-BASED DIFFERENTIAL FUZZING.
// ===========================================================================

/// Build a directed edge list (ordered pairs) from a bit vector over all n*(n-1)
/// ordered off-diagonal pairs.
fn dir_edges_from_bits(n: usize, bits: &[bool]) -> Vec<(usize, usize)> {
    let mut e = Vec::new();
    let mut idx = 0;
    for i in 0..n {
        for j in 0..n {
            if i != j {
                if bits.get(idx).copied().unwrap_or(false) {
                    e.push((i, j));
                }
                idx += 1;
            }
        }
    }
    e
}

fn dir_graph_strategy() -> impl Strategy<Value = (usize, Vec<bool>)> {
    (0usize..=6).prop_flat_map(|n| {
        let pairs = n * n.saturating_sub(1);
        (Just(n), proptest::collection::vec(any::<bool>(), pairs))
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(120))]

    // (g) monomorphism enumerator (undirected) == brute-force oracle, and instance
    //     count == streaming count, over the small-pattern battery and fuzzed hosts.
    #[test]
    fn prop_monomorphism_differential_undirected((n, bits, _s) in graph_strategy()) {
        let hedges = edges_from_bits(n, &bits);
        let hm = adj_matrix(&hedges, n);
        let host = build_un(&hedges, n);
        for (pk, pedges) in undirected_pattern_battery() {
            if pk > n {
                prop_assert_eq!(count_monomorphisms_unlabelled(&build_un(&pedges, pk), &host), 0);
                continue;
            }
            let pat = build_un(&pedges, pk);
            let oracle = bf_mono(&pedges, pk, &hm, n);
            let cnt = count_monomorphisms_unlabelled(&pat, &host);
            let insts = monomorphisms_unlabelled(&pat, &host);
            prop_assert_eq!(cnt, oracle, "count k={} n={}", pk, n);
            prop_assert_eq!(insts.len(), oracle, "instances k={} n={}", pk, n);
        }
    }

    // (h) monomorphism enumerator (DIRECTED) == brute-force oracle, honouring arc
    //     direction, over fuzzed directed hosts and a directed-pattern battery.
    #[test]
    fn prop_monomorphism_differential_directed((n, bits) in dir_graph_strategy()) {
        let harcs = dir_edges_from_bits(n, &bits);
        let hm = dir_matrix(&harcs, n);
        let host = build_directed(&harcs, n);
        let patterns: Vec<(usize, Vec<(usize, usize)>)> = vec![
            (2, vec![(0, 1)]),
            (3, vec![(0, 1), (1, 2)]),
            (3, vec![(0, 1), (1, 2), (0, 2)]),   // FFL
            (3, vec![(0, 1), (1, 2), (2, 0)]),   // directed C3
            (4, vec![(0, 1), (0, 2), (3, 1), (3, 2)]), // bi-fan
        ];
        for (pk, parcs) in patterns {
            if pk > n {
                continue;
            }
            let pat = build_directed(&parcs, pk);
            prop_assert_eq!(
                count_monomorphisms_unlabelled(&pat, &host),
                bf_mono(&parcs, pk, &hm, n),
                "directed k={} n={}", pk, n
            );
        }
    }

    // (i) find_motif distinct-occurrence count == count_motif (the s(P,C)/census fast
    //     path), for both semantics and every named motif; and induced occurrences hit
    //     distinct node-sets.
    #[test]
    fn prop_find_motif_matches_count((n, bits, _s) in graph_strategy()) {
        let edges = edges_from_bits(n, &bits);
        let g = build_un(&edges, n);
        for (pk, pedges) in undirected_pattern_battery() {
            let pat = Pattern::new(pk, &pedges);
            for ind in [Induced::Yes, Induced::No] {
                let insts = find_motif(&g, &pat, ind);
                prop_assert_eq!(insts.len() as u64, count_motif(&g, &pat, ind),
                    "k={} n={} {:?}", pk, n, ind);
                if ind == Induced::Yes {
                    let mut sets: HashSet<Vec<usize>> = HashSet::new();
                    for m in &insts {
                        let mut s: Vec<usize> = m.iter().map(|nid| nid.index()).collect();
                        s.sort_unstable();
                        prop_assert!(sets.insert(s), "induced node-sets distinct");
                    }
                }
            }
        }
    }
}

// ===========================================================================
// PHASE-2 NULL MODEL TESTS
// ===========================================================================

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Stub-degree of node i: counts each edge endpoint, counting self-loop as 2.
/// This is the correct invariant for the raw configuration model.
fn stub_degree_of(g: &UnGraph<(), ()>, i: usize) -> usize {
    use petgraph::visit::EdgeRef as _;
    g.edge_references()
        .map(|e| {
            let a = e.source().index();
            let b = e.target().index();
            if a == b {
                if a == i {
                    2
                } else {
                    0
                }
            } else {
                usize::from(a == i) + usize::from(b == i)
            }
        })
        .sum()
}

/// Degree of node i in a simple graph (no self-loops, no parallel edges).
fn simple_degree(g: &UnGraph<(), ()>, i: usize) -> usize {
    g.edges(NodeIndex::new(i)).count()
}

/// Whether g has no self-loops.
fn no_self_loops(g: &UnGraph<(), ()>) -> bool {
    use petgraph::visit::EdgeRef as _;
    g.edge_references()
        .all(|e| e.source().index() != e.target().index())
}

/// Whether g has no parallel edges (requires simple graph, no self-loops).
fn no_parallel_edges(g: &UnGraph<(), ()>) -> bool {
    use petgraph::visit::EdgeRef as _;
    let mut seen: HashSet<(usize, usize)> = HashSet::new();
    for e in g.edge_references() {
        let a = e.source().index();
        let b = e.target().index();
        if a == b {
            continue;
        }
        let k = if a < b { (a, b) } else { (b, a) };
        if !seen.insert(k) {
            return false;
        }
    }
    true
}

/// Edge set as a sorted vec of canonical pairs.
fn edge_set_sorted(g: &UnGraph<(), ()>) -> Vec<(usize, usize)> {
    use petgraph::visit::EdgeRef as _;
    let mut v: Vec<(usize, usize)> = g
        .edge_references()
        .map(|e| {
            let a = e.source().index();
            let b = e.target().index();
            if a < b {
                (a, b)
            } else {
                (b, a)
            }
        })
        .collect();
    v.sort_unstable();
    v
}

// ---------------------------------------------------------------------------
// Configuration model — raw
// ---------------------------------------------------------------------------

#[test]
fn config_model_raw_exact_degrees() {
    let mut rng = StdRng::seed_from_u64(1);
    // Even-sum degree sequences.
    for deg_seq in &[
        vec![2usize, 2, 2, 2],
        vec![1, 1, 2, 2],
        vec![3, 3, 2, 2, 2, 2],
        vec![4, 4, 4, 4, 4, 4],
        vec![1, 1, 1, 1, 1, 1, 1, 1],
    ] {
        let g = configuration_model(deg_seq, &mut rng);
        assert_eq!(g.node_count(), deg_seq.len(), "node count");
        assert_eq!(
            g.edge_count(),
            deg_seq.iter().sum::<usize>() / 2,
            "edge count"
        );
        for (i, &d) in deg_seq.iter().enumerate() {
            assert_eq!(stub_degree_of(&g, i), d, "stub-degree mismatch at node {i}");
        }
    }
}

#[test]
#[should_panic(expected = "even stub sum")]
fn config_model_odd_sum_panics() {
    let mut rng = StdRng::seed_from_u64(0);
    let _ = configuration_model(&[1, 2], &mut rng); // sum = 3, odd
}

// ---------------------------------------------------------------------------
// Configuration model — simple
// ---------------------------------------------------------------------------

#[test]
fn config_model_simple_no_loops_no_parallel() {
    let mut rng = StdRng::seed_from_u64(2);
    for deg_seq in &[
        vec![2usize, 2, 2, 2],
        vec![3, 3, 3, 3, 3, 3],
        vec![1, 1, 2, 2, 2, 2],
        vec![4, 4, 4, 4, 4, 4, 4, 4],
    ] {
        let g = configuration_model_simple(deg_seq, &mut rng);
        assert_eq!(g.node_count(), deg_seq.len(), "node count");
        assert!(no_self_loops(&g), "no self-loops");
        assert!(no_parallel_edges(&g), "no parallel edges");
        // Realized degree <= degree_seq[i].
        for (i, &d) in deg_seq.iter().enumerate() {
            assert!(
                simple_degree(&g, i) <= d,
                "realized degree > requested degree at node {i}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Double-edge-swap
// ---------------------------------------------------------------------------

#[test]
fn des_degree_preservation() {
    let mut rng = StdRng::seed_from_u64(3);
    let n = 12;
    let edges = random_edges(n, 0.4, 7);
    let g = build_un(&edges, n);
    let orig_degrees: Vec<usize> = (0..n).map(|i| simple_degree(&g, i)).collect();
    let swapped = double_edge_swap(&g, 200, &mut rng);
    let new_degrees: Vec<usize> = (0..n).map(|i| simple_degree(&swapped, i)).collect();
    assert_eq!(new_degrees, orig_degrees, "degrees must be preserved");
}

#[test]
fn des_no_self_loops_or_parallel() {
    let n = 16;
    let edges = random_edges(n, 0.4, 8);
    let g = build_un(&edges, n);
    for swaps in [50, 200, 500] {
        let mut r = StdRng::seed_from_u64(swaps as u64);
        let swapped = double_edge_swap(&g, swaps, &mut r);
        assert!(no_self_loops(&swapped), "no self-loops after {swaps} swaps");
        assert!(
            no_parallel_edges(&swapped),
            "no parallel edges after {swaps} swaps"
        );
    }
}

#[test]
fn des_edge_count_preserved() {
    let mut rng = StdRng::seed_from_u64(5);
    let n = 14;
    let edges = random_edges(n, 0.35, 9);
    let g = build_un(&edges, n);
    let m = g.edge_count();
    let swapped = double_edge_swap(&g, 300, &mut rng);
    assert_eq!(swapped.edge_count(), m, "edge count preserved");
}

#[test]
fn des_mixing_evidence() {
    // After enough swaps on a graph with many edges, the edge set should change.
    let n = 20;
    let edges = random_edges(n, 0.4, 11);
    let g = build_un(&edges, n);
    let orig = edge_set_sorted(&g);
    let mut rng = StdRng::seed_from_u64(42);
    let swapped = double_edge_swap(&g, 500, &mut rng);
    let new = edge_set_sorted(&swapped);
    assert_ne!(orig, new, "edge set must change after sufficient swaps");
}

#[test]
fn des_empty_and_single_edge_passthrough() {
    let mut rng = StdRng::seed_from_u64(6);
    // Empty graph.
    let empty: UnGraph<(), ()> = build_un(&[], 0);
    let r = double_edge_swap(&empty, 10, &mut rng);
    assert_eq!(r.node_count(), 0);
    assert_eq!(r.edge_count(), 0);
    // Single edge: no valid swap.
    let single = build_un(&[(0, 1)], 2);
    let r2 = double_edge_swap(&single, 10, &mut rng);
    assert_eq!(r2.edge_count(), 1);
}

// ---------------------------------------------------------------------------
// Watts-Strogatz
// ---------------------------------------------------------------------------

#[test]
fn ws_node_and_edge_count() {
    let mut rng = StdRng::seed_from_u64(7);
    for &(n, k) in &[(10usize, 4usize), (20, 6), (50, 4), (100, 8)] {
        let g = watts_strogatz(n, k, 0.3, &mut rng);
        assert_eq!(g.node_count(), n, "node count n={n} k={k}");
        assert_eq!(g.edge_count(), n * k / 2, "edge count n={n} k={k}");
    }
}

#[test]
fn ws_p0_is_pure_ring_lattice() {
    let n = 20;
    let k = 4;
    let mut rng = StdRng::seed_from_u64(8);
    let g = watts_strogatz(n, k, 0.0, &mut rng);
    assert_eq!(g.node_count(), n);
    assert_eq!(g.edge_count(), n * k / 2);
    let half_k = k / 2;
    for i in 0..n {
        for j in 1..=half_k {
            let nb = (i + j) % n;
            assert!(
                g.contains_edge(NodeIndex::new(i), NodeIndex::new(nb)),
                "ring lattice edge ({i},{nb}) missing"
            );
        }
    }
    // No self-loops or parallel edges in the ring lattice.
    assert!(no_self_loops(&g));
    assert!(no_parallel_edges(&g));
}

#[test]
fn ws_simple_at_all_p() {
    let mut rng = StdRng::seed_from_u64(9);
    let n = 30;
    let k = 4;
    for &p in &[0.0, 0.1, 0.5, 1.0] {
        let g = watts_strogatz(n, k, p, &mut rng);
        assert_eq!(g.node_count(), n, "p={p}");
        assert_eq!(g.edge_count(), n * k / 2, "edge count p={p}");
        assert!(no_self_loops(&g), "self-loop at p={p}");
        assert!(no_parallel_edges(&g), "parallel edge at p={p}");
    }
}

#[test]
fn ws_p1_not_ring() {
    // At p=1 the ring lattice structure is broken; check via multiple seeds.
    let n = 40;
    let k = 4;
    let half_k = k / 2;
    let mut found_non_ring = false;
    for seed in 0..20u64 {
        let mut rng = StdRng::seed_from_u64(seed + 100);
        let g = watts_strogatz(n, k, 1.0, &mut rng);
        // Check if at least one expected lattice edge is missing.
        let any_missing = (0..n).any(|i| {
            (1..=half_k).any(|j| {
                let nb = (i + j) % n;
                !g.contains_edge(NodeIndex::new(i), NodeIndex::new(nb))
            })
        });
        if any_missing {
            found_non_ring = true;
            break;
        }
    }
    assert!(found_non_ring, "p=1 graph should not be a ring lattice");
}

#[test]
#[should_panic(expected = "k must be even")]
fn ws_odd_k_panics() {
    let mut rng = StdRng::seed_from_u64(0);
    let _ = watts_strogatz(10, 3, 0.1, &mut rng);
}

#[test]
#[should_panic(expected = "n must be greater than k")]
fn ws_n_le_k_panics() {
    let mut rng = StdRng::seed_from_u64(0);
    let _ = watts_strogatz(4, 4, 0.1, &mut rng);
}

// ---------------------------------------------------------------------------
// LFR benchmark
// ---------------------------------------------------------------------------

#[test]
fn lfr_node_and_community_count() {
    let mut rng = StdRng::seed_from_u64(10);
    let n = 100;
    let (g, community) = lfr_benchmark(n, 5.0, 15, 0.1, 2.5, 1.5, 10, 50, &mut rng);
    assert_eq!(g.node_count(), n, "node count");
    assert_eq!(community.len(), n, "community vec length");
}

#[test]
fn lfr_community_labels_in_range() {
    let mut rng = StdRng::seed_from_u64(11);
    let n = 80;
    let (_, community) = lfr_benchmark(n, 4.0, 12, 0.2, 2.5, 1.5, 8, 30, &mut rng);
    let max_label = *community.iter().max().unwrap();
    // There must be at least 1 community and labels start at 0.
    assert!(max_label < n, "community labels must be < n");
    // Every node has a valid label (some community was assigned).
    assert_eq!(community.len(), n);
}

#[test]
fn lfr_degree_distribution_sanity() {
    // Degrees should be in [1, max_degree] and the mean should be in a loose range.
    let mut rng = StdRng::seed_from_u64(12);
    let n = 200;
    let max_degree = 20;
    let avg_degree = 6.0;
    let (g, _) = lfr_benchmark(n, avg_degree, max_degree, 0.1, 2.5, 1.5, 10, 60, &mut rng);
    assert_eq!(g.node_count(), n);
    let degrees: Vec<usize> = (0..n).map(|i| g.edges(NodeIndex::new(i)).count()).collect();
    // All degrees <= max_degree (guaranteed by power-law sampling bound).
    assert!(
        degrees.iter().all(|&d| d <= max_degree * 2),
        "some degree exceeds 2*max_degree (LFR allows degree growth from external matching)"
    );
    // Mean degree is non-zero.
    let mean: f64 = degrees.iter().sum::<usize>() as f64 / n as f64;
    assert!(mean > 0.5, "mean degree must be positive, got {mean}");
}

#[test]
fn lfr_mixing_fraction_roughly_mu() {
    // For each node, fraction of edges going outside its community ≈ mu.
    let mu = 0.15f64;
    let mut rng = StdRng::seed_from_u64(13);
    let n = 200;
    let (g, community) = lfr_benchmark(n, 6.0, 20, mu, 2.5, 1.5, 10, 60, &mut rng);
    use petgraph::visit::EdgeRef as _;
    let mut external_fractions: Vec<f64> = Vec::new();
    for i in 0..n {
        let deg = g.edges(NodeIndex::new(i)).count();
        if deg == 0 {
            continue;
        }
        let external = g
            .edges(NodeIndex::new(i))
            .filter(|e| {
                let nb = if e.source().index() == i {
                    e.target().index()
                } else {
                    e.source().index()
                };
                community[nb] != community[i]
            })
            .count();
        external_fractions.push(external as f64 / deg as f64);
    }
    if !external_fractions.is_empty() {
        let realized_mu = external_fractions.iter().sum::<f64>() / external_fractions.len() as f64;
        // Loose tolerance: LFR approximations can deviate.
        assert!(
            realized_mu < mu + 0.25,
            "realized mixing {realized_mu:.3} much higher than mu={mu}"
        );
    }
}

// ---------------------------------------------------------------------------
// Proptest: double-edge-swap degree preservation
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Degree sequence of every node is unchanged by double_edge_swap, for all
    /// random simple graphs and swap counts.
    #[test]
    fn prop_des_degree_preservation((n, bits, seed) in graph_strategy()) {
        let edges = edges_from_bits(n, &bits);
        let g = build_un(&edges, n);
        let orig_deg: Vec<usize> = (0..n).map(|i| simple_degree(&g, i)).collect();
        let n_swaps = n.saturating_mul(5).max(10);
        let mut rng = StdRng::seed_from_u64(seed);
        let swapped = double_edge_swap(&g, n_swaps, &mut rng);
        let new_deg: Vec<usize> = (0..n).map(|i| simple_degree(&swapped, i)).collect();
        prop_assert_eq!(new_deg, orig_deg, "degree preservation n={}", n);
    }

    /// No self-loops or parallel edges in double_edge_swap output.
    #[test]
    fn prop_des_simple_output((n, bits, seed) in graph_strategy()) {
        let edges = edges_from_bits(n, &bits);
        let g = build_un(&edges, n);
        let mut rng = StdRng::seed_from_u64(seed);
        let swapped = double_edge_swap(&g, 50, &mut rng);
        prop_assert!(no_self_loops(&swapped), "self-loop in output n={}", n);
        prop_assert!(no_parallel_edges(&swapped), "parallel edge in output n={}", n);
    }
}

// ---------------------------------------------------------------------------
// Proptest: configuration model raw stub-degree exactness
// ---------------------------------------------------------------------------

/// Strategy: a vector of 2–8 degrees each in 1..=6, with even total sum.
fn even_degree_seq_strategy() -> impl Strategy<Value = Vec<usize>> {
    proptest::collection::vec(1usize..=6, 2..=8).prop_map(|mut v| {
        if v.iter().sum::<usize>() % 2 != 0 {
            v[0] += 1; // bump first element to make sum even
        }
        v
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// Raw configuration model stub-degrees exactly equal the requested sequence.
    #[test]
    fn prop_config_model_raw_exact_stubs(deg_seq in even_degree_seq_strategy(), seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        let g = configuration_model(&deg_seq, &mut rng);
        prop_assert_eq!(g.node_count(), deg_seq.len());
        prop_assert_eq!(g.edge_count(), deg_seq.iter().sum::<usize>() / 2);
        for (i, &d) in deg_seq.iter().enumerate() {
            prop_assert_eq!(
                stub_degree_of(&g, i), d,
                "stub-degree mismatch at node {} (deg_seq={:?})", i, deg_seq
            );
        }
    }

    /// Simple variant: no self-loops, no parallel edges, realized degree ≤ requested.
    #[test]
    fn prop_config_model_simple_valid(deg_seq in even_degree_seq_strategy(), seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        let g = configuration_model_simple(&deg_seq, &mut rng);
        prop_assert_eq!(g.node_count(), deg_seq.len());
        prop_assert!(no_self_loops(&g), "self-loop in simple variant");
        prop_assert!(no_parallel_edges(&g), "parallel edge in simple variant");
        for (i, &d) in deg_seq.iter().enumerate() {
            prop_assert!(
                simple_degree(&g, i) <= d,
                "realized degree {} > requested {} at node {}",
                simple_degree(&g, i), d, i
            );
        }
    }
}
