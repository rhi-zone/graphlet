//! Canonical labelling of small graphlets.
//!
//! A k-node induced subgraph is labelled by the minimum, over all `k!` vertex
//! permutations, of the packed upper-triangle adjacency bitmask. The minimum is
//! insertion-order invariant, so two subgraphs share a [`ClassId`] iff they are
//! isomorphic. Exhaustive over `k!` — intended for `k ≤ 5`.

/// A stable graphlet-class identifier: the canonical (minimum) adjacency bitmask.
///
/// Comparable and hashable; its numeric value is an implementation detail (it is a
/// canonical mask, *not* ORCA's published class ordering).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ClassId(pub u64);

/// All `k!` permutations of `0..k`.
pub(crate) fn perms(k: usize) -> Vec<Vec<usize>> {
    let mut res = Vec::new();
    let mut cur: Vec<usize> = (0..k).collect();
    fn go(cur: &mut Vec<usize>, i: usize, res: &mut Vec<Vec<usize>>) {
        if i == cur.len() {
            res.push(cur.clone());
            return;
        }
        for j in i..cur.len() {
            cur.swap(i, j);
            go(cur, i + 1, res);
            cur.swap(i, j);
        }
    }
    go(&mut cur, 0, &mut res);
    res
}

/// The `k(k-1)/2` upper-triangle vertex pairs in bit order.
pub(crate) fn pairs(k: usize) -> Vec<(usize, usize)> {
    (0..k)
        .flat_map(|i| ((i + 1)..k).map(move |j| (i, j)))
        .collect()
}

/// Pack the induced upper-triangle bitmask under one permutation, where canonical
/// slot `c` is filled by local vertex `perm[c]` and adjacency of local vertices is
/// probed by `is_adj`.
fn mask_of(k: usize, perm: &[usize], is_adj: &impl Fn(usize, usize) -> bool) -> u64 {
    let mut m = 0u64;
    let mut bit = 0;
    for i in 0..k {
        for j in (i + 1)..k {
            if is_adj(perm[i], perm[j]) {
                m |= 1 << bit;
            }
            bit += 1;
        }
    }
    m
}

/// Packed upper-triangle mask under a single permutation (used for automorphism
/// detection: `p` is an automorphism of a class iff its mask equals the class mask).
pub(crate) fn mask_of_by(k: usize, perm: &[usize], is_adj: impl Fn(usize, usize) -> bool) -> u64 {
    mask_of(k, perm, &is_adj)
}

/// Canonical mask of a k-vertex induced subgraph given an adjacency predicate over
/// its local vertices `0..k`.
pub(crate) fn canonical_by(
    k: usize,
    ps: &[Vec<usize>],
    is_adj: impl Fn(usize, usize) -> bool,
) -> u64 {
    ps.iter().map(|p| mask_of(k, p, &is_adj)).min().unwrap()
}

/// Canonical mask plus a witnessing arg-permutation: `arg[c]` is the local vertex
/// feeding canonical slot `c`. Any arg achieving the minimum is a valid
/// instance→canonical isomorphism; since orbit ids are automorphism-invariant, the
/// choice among ties does not matter.
pub(crate) fn canonical_arg_by(
    k: usize,
    ps: &[Vec<usize>],
    is_adj: impl Fn(usize, usize) -> bool,
) -> (u64, Vec<usize>) {
    let mut best = u64::MAX;
    let mut arg = ps[0].clone();
    for p in ps {
        let m = mask_of(k, p, &is_adj);
        if m < best {
            best = m;
            arg = p.clone();
        }
    }
    (best, arg)
}

/// The `k×k` adjacency of the graphlet class with the given canonical mask.
pub(crate) fn class_to_adj(mask: u64, k: usize) -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::new(); k];
    for (b, &(i, j)) in pairs(k).iter().enumerate() {
        if mask & (1 << b) != 0 {
            adj[i].push(j);
            adj[j].push(i);
        }
    }
    adj
}

/// Whether an adjacency (as neighbor lists) is connected.
pub(crate) fn connected(adj: &[Vec<usize>]) -> bool {
    let n = adj.len();
    if n == 0 {
        return true;
    }
    let mut seen = vec![false; n];
    let mut stack = vec![0usize];
    seen[0] = true;
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
    cnt == n
}

/// Every connected graphlet class at order `k`, as canonical masks (sorted).
///
/// Exhaustive over all `2^(k(k-1)/2)` labelled graphs — the independent ground truth
/// for class counts (k=3→2, k=4→6, k=5→21).
pub(crate) fn all_connected_classes(k: usize) -> Vec<u64> {
    let ps = perms(k);
    let np = pairs(k).len();
    let mut classes = std::collections::HashSet::new();
    for mask in 0u64..(1u64 << np) {
        let adj = class_to_adj(mask, k);
        if !connected(&adj) {
            continue;
        }
        classes.insert(canonical_by(k, &ps, |i, j| adj[i].contains(&j)));
    }
    let mut v: Vec<u64> = classes.into_iter().collect();
    v.sort_unstable();
    v
}
