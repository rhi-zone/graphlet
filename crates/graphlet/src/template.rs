//! Template matching — the parallel arm delegating to petgraph's VF2.
//!
//! petgraph's `subgraph_isomorphisms_iter` returns **node-induced** subgraph
//! isomorphisms natively (verified: its docstring states "'subgraph' always means a
//! 'node-induced subgraph'", its `is_feasible` rejects extra host edges, and
//! empirically a P3 pattern finds 0 matches in a triangle host). So the induced arm
//! is free — we delegate and do no filtering.
//!
//! **Non-induced (monomorphism) matching of an arbitrary template is deliberately
//! not implemented.** It cannot be recovered by filtering petgraph's output (induced
//! ⊂ monomorphism — a post-filter can only shrink the set), the k-bounded `s(P,C)`
//! trick from [`crate::catalog`] does not apply to an unbounded template, and no
//! grounding consumer needs it today (science domains are induced; small named
//! software patterns are served by the catalog arm). It is reserved as a future
//! additive method, never an erroring runtime toggle — see ADR-0290.

use petgraph::algo::subgraph_isomorphisms_iter;
use petgraph::graph::Graph;
use petgraph::EdgeType;

/// All node-induced matches of `pattern` in `host`, each a vector indexed by pattern
/// node with the matched host node index. Induced-native (no filtering).
///
/// # Counting semantics: raw embeddings, not distinct node-sets
///
/// Each returned vector is one VF2 *embedding* (an ordered injection of pattern nodes
/// into host nodes). Pattern automorphisms are **not** deduped: a symmetric pattern
/// matched onto one host node-set yields `|Aut(pattern)|` separate embeddings here. For
/// example a P3 (path `0–1–2`) on a triangle host yields 0 (P3 is not induced in a
/// triangle), but a P3 on a host path `a–b–c` yields 2 embeddings (`a,b,c` and `c,b,a`)
/// over the single node-set `{a,b,c}`. This differs from [`crate::catalog`], which
/// reports *distinct occurrences* (node-sets) by dividing out `|Aut(P)|`.
///
/// `node_match` / `edge_match` gate a pattern element against a host element (return
/// `true` to allow); pass `|_, _| true` to match on structure alone.
#[must_use]
pub fn induced_matches<Np, Ep, Nh, Eh, Ty, NM, EM>(
    pattern: &Graph<Np, Ep, Ty>,
    host: &Graph<Nh, Eh, Ty>,
    mut node_match: NM,
    mut edge_match: EM,
) -> Vec<Vec<usize>>
where
    Ty: EdgeType,
    NM: FnMut(&Np, &Nh) -> bool,
    EM: FnMut(&Ep, &Eh) -> bool,
{
    match subgraph_isomorphisms_iter(&pattern, &host, &mut node_match, &mut edge_match) {
        Some(it) => it.collect(),
        None => Vec::new(),
    }
}

/// All node-induced matches of `pattern` in `host` on structure alone (no node/edge
/// predicates). See [`induced_matches`].
#[must_use]
pub fn induced_matches_unlabelled<Np, Ep, Nh, Eh, Ty>(
    pattern: &Graph<Np, Ep, Ty>,
    host: &Graph<Nh, Eh, Ty>,
) -> Vec<Vec<usize>>
where
    Ty: EdgeType,
{
    induced_matches(pattern, host, |_, _| true, |_, _| true)
}

/// Count node-induced matches of `pattern` in `host` on structure alone.
///
/// Counts raw VF2 embeddings, including pattern automorphisms (no node-set dedup) —
/// see [`induced_matches`].
#[must_use]
pub fn count_induced_matches<Np, Ep, Nh, Eh, Ty>(
    pattern: &Graph<Np, Ep, Ty>,
    host: &Graph<Nh, Eh, Ty>,
) -> usize
where
    Ty: EdgeType,
{
    match subgraph_isomorphisms_iter(&pattern, &host, &mut |_, _| true, &mut |_, _| true) {
        Some(it) => it.count(),
        None => 0,
    }
}
