//! Canonical labelling of small directed graphlets.
//!
//! Generalizes [`crate::canonical`] from an unordered upper-triangle mask to an
//! **ordered**-pair mask: a k-node induced directed subgraph is labelled by the
//! minimum, over all `k!` vertex permutations, of the packed `k(k-1)`-bit arc mask (one
//! bit per ordered pair `(i, j)`, `i != j`). Two directed subgraphs share a
//! [`DirectedClassId`] iff they are isomorphic *as directed graphs* (an isomorphism
//! must map arcs to arcs preserving direction). Exhaustive over `k!` permutations —
//! intended for `k <= 5` (`k = 3` for the triad census, `k in 4..=5` for the directed
//! graphlet census); `k(k-1)` bits fits comfortably in `u64` up to `k = 8`, but
//! `k!`-exhaustive canonicalization is only cheap through `k = 4` — at `k = 5` it is
//! still exact but noticeably slower (the per-instance canonicalization exhausts `120`
//! permutations, and the one-time [`all_weakly_connected_classes`] ground-truth sweep
//! at registry-build time exhausts all `2^20` labelled digraphs).

/// A stable directed-graphlet-class identifier: the canonical (minimum) ordered-pair
/// arc bitmask. Comparable and hashable; its numeric value is an implementation
/// detail, not a published class ordering.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct DirectedClassId(pub u64);

/// The `k(k-1)` ordered vertex pairs `(i, j)`, `i != j`, in bit order.
pub(crate) fn directed_pairs(k: usize) -> Vec<(usize, usize)> {
    (0..k)
        .flat_map(|i| (0..k).filter(move |&j| j != i).map(move |j| (i, j)))
        .collect()
}

/// Pack the induced ordered-pair arc bitmask under one permutation, where canonical
/// slot `c` is filled by local vertex `perm[c]` and arc presence is probed by `has_arc`.
fn mask_of(k: usize, perm: &[usize], has_arc: &impl Fn(usize, usize) -> bool) -> u64 {
    let mut m = 0u64;
    let mut bit = 0u32;
    for i in 0..k {
        for j in 0..k {
            if i == j {
                continue;
            }
            if has_arc(perm[i], perm[j]) {
                m |= 1 << bit;
            }
            bit += 1;
        }
    }
    m
}

/// Packed ordered-pair mask under a single permutation (used for directed
/// automorphism detection: `p` is a directed automorphism of a class iff its mask
/// equals the class mask).
pub(crate) fn mask_of_by(k: usize, perm: &[usize], has_arc: impl Fn(usize, usize) -> bool) -> u64 {
    mask_of(k, perm, &has_arc)
}

/// Canonical mask of a k-vertex induced directed subgraph given an arc predicate over
/// its local vertices `0..k`.
pub(crate) fn canonical_by(
    k: usize,
    ps: &[Vec<usize>],
    has_arc: impl Fn(usize, usize) -> bool,
) -> u64 {
    ps.iter().map(|p| mask_of(k, p, &has_arc)).min().unwrap()
}

/// Canonical mask plus a witnessing arg-permutation: `arg[c]` is the local vertex
/// feeding canonical slot `c`.
pub(crate) fn canonical_arg_by(
    k: usize,
    ps: &[Vec<usize>],
    has_arc: impl Fn(usize, usize) -> bool,
) -> (u64, Vec<usize>) {
    let mut best = u64::MAX;
    let mut arg = ps[0].clone();
    for p in ps {
        let m = mask_of(k, p, &has_arc);
        if m < best {
            best = m;
            arg.clone_from(p);
        }
    }
    (best, arg)
}

/// The `k`-vertex out-adjacency of the directed-graphlet class with the given
/// canonical mask.
pub(crate) fn class_to_arcs(mask: u64, k: usize) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::new(); k];
    for (b, &(i, j)) in directed_pairs(k).iter().enumerate() {
        if mask & (1 << b) != 0 {
            out[i].push(j);
        }
    }
    out
}

/// Whether a directed adjacency (as out-neighbor lists) is *weakly* connected: its
/// underlying undirected union is connected. Reuses [`crate::canonical::connected`],
/// the crate's single already-verified connectivity check.
pub(crate) fn weakly_connected(out: &[Vec<usize>]) -> bool {
    let k = out.len();
    let mut und = vec![Vec::new(); k];
    for (i, row) in out.iter().enumerate() {
        for &j in row {
            und[i].push(j);
            und[j].push(i);
        }
    }
    crate::canonical::connected(&und)
}

/// Every weakly-connected directed-graphlet class at order `k`, as canonical masks
/// (sorted).
///
/// Exhaustive over all `2^(k(k-1))` labelled directed graphs — the independent ground
/// truth for directed class counts. At `k = 3` this includes all 13 weakly-connected
/// triad types (the 3 disconnected types — 003, 012, 102 — are excluded here but
/// covered separately by the full 16-type [`crate::rim::directed::triad`] census,
/// which is *not* connectivity-restricted).
pub(crate) fn all_weakly_connected_classes(k: usize) -> Vec<u64> {
    use crate::canonical::perms;
    use std::collections::HashSet;

    let ps = perms(k);
    let np = directed_pairs(k).len();
    let mut classes = HashSet::new();
    for mask in 0u64..(1u64 << np) {
        let out = class_to_arcs(mask, k);
        if !weakly_connected(&out) {
            continue;
        }
        classes.insert(canonical_by(k, &ps, |i, j| out[i].contains(&j)));
    }
    let mut v: Vec<u64> = classes.into_iter().collect();
    v.sort_unstable();
    v
}
