//! The directed triad census: the standard 16 isomorphism classes of 3-node directed
//! graphs (Holland & Leinhardt 1970; the MAN — Mutual/Asymmetric/Null dyad-count —
//! naming popularized by Davis & Leinhardt and implemented identically in `sna`
//! (R) and `igraph`'s `triad_census`).
//!
//! Unlike the crate's directed graphlet census (which restricts to *weakly-connected*
//! subsets, the graphlet convention), the triad census counts **every** 3-subset,
//! connected or not — 3 of the 16 types (003, 012, 102) are disconnected. This is the
//! standard social-network-analysis quantity: e.g. `sum(census) == C(n, 3)`.
//!
//! Each dyad (unordered node pair) is independently Null (no arc either way), Asym
//! (exactly one arc), or Mutual (both arcs); the multiset of the three dyads' types
//! (M/A/N counts) fixes the base code, and four codes (021, 111, 030, 120) split
//! further into named subtypes by *which* node plays the distinguished role:
//!
//! - **021D/U/C**: with two asymmetric dyads and one null dyad, the two asymmetric
//!   dyads share a "center" node; both arcs point away from it (**D**own, out-star),
//!   both point into it (**U**p, in-star), or one each (**C**hain).
//! - **111D/U**: with one mutual dyad and one asymmetric dyad (the third dyad null),
//!   the asymmetric arc either points into the mutual pair (**D**own) or out of it
//!   (**U**p).
//! - **030T/C**: with three asymmetric dyads, either one node has out-degree 2 within
//!   the triad (**T**ransitive) or every node has out-degree 1 (**C**yclic).
//! - **120D/U/C**: like 021D/U/C, but the previously-null dyad is now mutual instead.
//!
//! (201, 210, 300 have no further subtype: with two, one, or zero null dyads fixed by
//! two or three mutual dyads, every configuration on 3 labelled nodes is isomorphic to
//! every other.)

use super::snapshot::{DirectedGraphAdapter, DirectedSnapshot};

/// The 16 standard triad (isomorphism class of a 3-node directed graph) types.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[allow(missing_docs)] // self-documenting via `label`/the module docs
pub enum TriadType {
    T003,
    T012,
    T102,
    T021D,
    T021U,
    T021C,
    T111D,
    T111U,
    T030T,
    T030C,
    T201,
    T120D,
    T120U,
    T120C,
    T210,
    T300,
}

impl TriadType {
    /// All 16 types, in the standard `sna`/`igraph` census order.
    #[must_use]
    pub fn all() -> [TriadType; 16] {
        use TriadType::*;
        [
            T003, T012, T102, T021D, T021U, T021C, T111D, T111U, T030T, T030C, T201, T120D, T120U,
            T120C, T210, T300,
        ]
    }

    /// The standard three-digit(-plus-letter) label, e.g. `"021C"`, `"300"`.
    #[must_use]
    pub fn label(self) -> &'static str {
        use TriadType::*;
        match self {
            T003 => "003",
            T012 => "012",
            T102 => "102",
            T021D => "021D",
            T021U => "021U",
            T021C => "021C",
            T111D => "111D",
            T111U => "111U",
            T030T => "030T",
            T030C => "030C",
            T201 => "201",
            T120D => "120D",
            T120U => "120U",
            T120C => "120C",
            T210 => "210",
            T300 => "300",
        }
    }

    /// This type's index into [`TriadType::all`] / [`TriadCensus`]'s internal order.
    fn index(self) -> usize {
        TriadType::all().iter().position(|&t| t == self).unwrap()
    }
}

/// A dyad's type: whether the pair `(a, b)` has no arc, one arc, or both.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Dyad {
    Null,
    Asym,
    Mutual,
}

fn dyad(ab: bool, ba: bool) -> Dyad {
    match (ab, ba) {
        (false, false) => Dyad::Null,
        (true, true) => Dyad::Mutual,
        _ => Dyad::Asym,
    }
}

/// Classify one labelled triple `(x, y, z)` by its directed adjacency (`has_arc(i, j)`
/// tells whether the arc `i -> j` is present, for `i, j` in `{x, y, z}`) into one of
/// the 16 standard types. Isomorphism-invariant: relabelling `x, y, z` never changes
/// the returned type.
#[must_use]
pub fn classify(x: usize, y: usize, z: usize, has_arc: impl Fn(usize, usize) -> bool) -> TriadType {
    let dxy = dyad(has_arc(x, y), has_arc(y, x));
    let dxz = dyad(has_arc(x, z), has_arc(z, x));
    let dyz = dyad(has_arc(y, z), has_arc(z, y));
    let dyads = [dxy, dxz, dyz];
    let m = dyads.iter().filter(|&&d| d == Dyad::Mutual).count();
    let a = dyads.iter().filter(|&&d| d == Dyad::Asym).count();

    // (node, node) pairs matching each dyad slot, so subtype logic can name endpoints.
    let pairs = [(x, y), (x, z), (y, z)];

    // For 021/120: the two asymmetric dyads share exactly one "center" node.
    let center_subtype = || -> TriadType {
        let asym_pairs: Vec<(usize, usize)> = (0..3)
            .filter(|&i| dyads[i] == Dyad::Asym)
            .map(|i| pairs[i])
            .collect();
        let (p, q) = (asym_pairs[0], asym_pairs[1]);
        // The shared node between the two asymmetric dyads.
        let center = if p.0 == q.0 || p.0 == q.1 { p.0 } else { p.1 };
        let other = |pair: (usize, usize)| if pair.0 == center { pair.1 } else { pair.0 };
        let (o1, o2) = (other(p), other(q));
        let out1 = has_arc(center, o1);
        let out2 = has_arc(center, o2);
        if out1 && out2 {
            TriadType::T021D
        } else if !out1 && !out2 {
            TriadType::T021U
        } else {
            TriadType::T021C
        }
    };

    match (m, a) {
        (0, 0) => TriadType::T003,
        (0, 1) => TriadType::T012,
        (1, 0) => TriadType::T102,
        (0, 2) => center_subtype(),
        (1, 1) => {
            // One mutual dyad {p, q}; the asymmetric dyad connects the third node r to
            // whichever of p/q (the remaining dyad is null).
            let asym_idx = (0..3).find(|&i| dyads[i] == Dyad::Asym).unwrap();
            let (u, v) = pairs[asym_idx];
            if has_arc(u, v) {
                // u -> v: is v part of the mutual dyad (arc points INTO it)?
                let mutual_idx = (0..3).find(|&i| dyads[i] == Dyad::Mutual).unwrap();
                let (mp, mq) = pairs[mutual_idx];
                if v == mp || v == mq {
                    TriadType::T111D
                } else {
                    TriadType::T111U
                }
            } else {
                let mutual_idx = (0..3).find(|&i| dyads[i] == Dyad::Mutual).unwrap();
                let (mp, mq) = pairs[mutual_idx];
                if u == mp || u == mq {
                    TriadType::T111D
                } else {
                    TriadType::T111U
                }
            }
        }
        (0, 3) => {
            // All three dyads asymmetric: transitive (some node has out-degree 2 among
            // the three arcs) or cyclic (every node has out-degree 1).
            let out_deg = |v: usize| {
                [x, y, z]
                    .iter()
                    .filter(|&&u| u != v && has_arc(v, u))
                    .count()
            };
            if [x, y, z].iter().any(|&v| out_deg(v) == 2) {
                TriadType::T030T
            } else {
                TriadType::T030C
            }
        }
        (2, 0) => TriadType::T201,
        (1, 2) => center_subtype_120(&dyads, &pairs, &has_arc),
        (2, 1) => TriadType::T210,
        (3, 0) => TriadType::T300,
        _ => unreachable!("M={m} A={a} impossible for 3 dyads"),
    }
}

/// 120D/U/C subtype: identical center-node logic to 021, just with the two
/// asymmetric dyads (not null ones) supplying the D/U/C distinction — the mutual dyad
/// plays no role beyond fixing the base code.
fn center_subtype_120(
    dyads: &[Dyad; 3],
    pairs: &[(usize, usize); 3],
    has_arc: &impl Fn(usize, usize) -> bool,
) -> TriadType {
    let asym_pairs: Vec<(usize, usize)> = (0..3)
        .filter(|&i| dyads[i] == Dyad::Asym)
        .map(|i| pairs[i])
        .collect();
    let (p, q) = (asym_pairs[0], asym_pairs[1]);
    let center = if p.0 == q.0 || p.0 == q.1 { p.0 } else { p.1 };
    let other = |pair: (usize, usize)| if pair.0 == center { pair.1 } else { pair.0 };
    let (o1, o2) = (other(p), other(q));
    let out1 = has_arc(center, o1);
    let out2 = has_arc(center, o2);
    if out1 && out2 {
        TriadType::T120D
    } else if !out1 && !out2 {
        TriadType::T120U
    } else {
        TriadType::T120C
    }
}

/// A class -> count map over the 16 standard triad types (dense array, indexed by
/// each [`TriadType`]'s internal ordering).
#[derive(Clone, Debug)]
pub struct TriadCensus {
    counts: [u64; 16],
}

impl TriadCensus {
    /// The count for one triad type.
    #[inline]
    #[must_use]
    pub fn get(&self, t: TriadType) -> u64 {
        self.counts[t.index()]
    }

    /// Total triads counted (should equal `C(n, 3)` for an `n`-node host).
    #[must_use]
    pub fn total(&self) -> u64 {
        self.counts.iter().sum()
    }

    /// Iterate `(type, count)` pairs in the standard census order.
    pub fn iter(&self) -> impl Iterator<Item = (TriadType, u64)> + '_ {
        TriadType::all().into_iter().map(|t| (t, self.get(t)))
    }
}

/// Compute the directed triad census of `g`: every 3-subset of nodes, classified into
/// one of the 16 standard types (connected or not).
///
/// `O(n^3)` in the node count (every unordered triple is visited once); this is the
/// direct/naive algorithm — the standard reference for correctness. A subquadratic
/// algorithm (Batagelj & Mrvar 2001, `O(m * maxdeg)`) is a possible future scalability
/// upgrade, not attempted here: nothing about the *counts* this module returns is
/// approximate.
///
/// `g` is treated as a *simple directed* graph (self-loop arcs stripped, parallel
/// same-direction arcs deduped) — see [`DirectedGraphAdapter`].
#[must_use]
pub fn triad_census<G>(g: G) -> TriadCensus
where
    G: DirectedGraphAdapter,
{
    let snapshot = DirectedSnapshot::new(g);
    let n = snapshot.len();
    let mut counts = [0u64; 16];
    let has_arc = |a: usize, b: usize| snapshot.has_arc(a, b);
    for x in 0..n {
        for y in (x + 1)..n {
            for z in (y + 1)..n {
                let t = classify(x, y, z, has_arc);
                counts[t.index()] += 1;
            }
        }
    }
    TriadCensus { counts }
}
