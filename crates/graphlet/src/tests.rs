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
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use crate::canonical::{
    all_connected_classes, canonical_arg_by, canonical_by, class_to_adj, connected, perms,
};
use crate::catalog::{count_diamonds, count_pattern, find_diamonds, Induced, Pattern};
use crate::census::{count, enumerate, for_each_subset, Census, Selector};
use crate::orbit::{graphlet_degree_vectors, Registry};
use crate::snapshot::Snapshot;
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
