//! Template matching — the arbitrary-graph arm.
//!
//! This arm matches an arbitrary `petgraph` query graph against a host graph, in two
//! honestly-distinct semantics, and unlike the census/[`catalog`](crate::catalog) arm
//! it is **not** bounded to `k ≤ 5`, honours node/edge weight predicates, and preserves
//! directedness.
//!
//! - **Induced** ([`induced_matches`]): delegated to petgraph's VF2
//!   `subgraph_isomorphisms_iter`, which is **node-induced native** (its docstring
//!   states "'subgraph' always means a 'node-induced subgraph'", its `is_feasible`
//!   rejects extra host edges, and empirically a P3 pattern finds 0 matches in a
//!   triangle host). So the induced arm is free — we delegate and do no filtering.
//! - **Non-induced / monomorphism** ([`monomorphisms`]): a real ordered-backtracking
//!   subgraph-monomorphism enumerator (this crate's own code — petgraph provides no
//!   monomorphism search, and it *cannot* be recovered by filtering the induced output
//!   since induced ⊂ monomorphism, a post-filter only shrinks). It returns every
//!   injective, **edge-preserving** (not edge-reflecting) mapping: every pattern edge
//!   maps to a host edge in the matching direction, but extra host edges among the
//!   image are allowed. Directed and undirected, with node/edge match predicates.
//!
//! ADR-0290 originally deferred this monomorphism arm for want of a grounding consumer;
//! this module activates it as the arbitrary-template engine (the bounded-`k` catalog
//! arm still serves small named patterns via the verified `s(P,C)` census derivation).
//!
//! # Counting semantics: raw embeddings, not distinct node-sets
//!
//! Both arms return **raw embeddings**: each result is one ordered injection of pattern
//! nodes into host nodes, and pattern automorphisms are **not** deduped. A symmetric
//! pattern matched onto one image yields `|Aut(pattern)|` separate embeddings. For a
//! distinct-occurrence count, divide the raw count by `|Aut(pattern)|` (the catalog arm
//! and [`crate::catalog::find_motif`] do this dedup for the bounded-`k` patterns).

use petgraph::graph::{EdgeIndex, Graph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::{Direction, EdgeType};

use petgraph::algo::subgraph_isomorphisms_iter;

/// All node-induced matches of `pattern` in `host`, each a vector indexed by pattern
/// node with the matched host node index. Induced-native via petgraph VF2 (no
/// filtering).
///
/// See the [module docs](self) for the raw-embedding counting semantics (pattern
/// automorphisms are not deduped).
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

// ---------------------------------------------------------------------------
// Monomorphism (non-induced) enumerator — this crate's own ordered backtracking.
// ---------------------------------------------------------------------------

/// How a placed pattern edge constrains the vertex being placed at a position.
enum Dir {
    /// A pattern self-loop on the current vertex: host must have the loop `hv → hv`.
    SelfLoop,
    /// Pattern edge `other → current`: host must have `assign[other] → hv`.
    FromOther,
    /// Pattern edge `current → other`: host must have `hv → assign[other]`.
    ToOther,
}

/// One adjacency constraint checked when a vertex is placed: the already-placed pattern
/// vertex it must connect to, the required direction, and the pattern edge (for
/// `edge_match`).
struct Cons {
    other: usize,
    dir: Dir,
    edge: EdgeIndex,
}

/// A backtracking plan: the pattern-vertex processing order (connected-first, for
/// pruning), the per-position adjacency constraints against already-placed vertices,
/// and the per-vertex out/in degrees for a necessary-condition degree prune.
struct Plan {
    order: Vec<usize>,
    cons: Vec<Vec<Cons>>,
    pat_out: Vec<usize>,
    pat_in: Vec<usize>,
}

/// Build the [`Plan`] for `pattern`: order the vertices connected-first (each newly
/// placed vertex maximizes edges to the already-placed set, so constraints bind early),
/// then attach each pattern edge as a constraint on the later of its two endpoints.
fn plan<Np, Ep, Ty: EdgeType>(pattern: &Graph<Np, Ep, Ty>) -> Plan {
    let pk = pattern.node_count();
    let directed = Ty::is_directed();
    let mut pnbr = vec![Vec::new(); pk];
    let mut pat_out = vec![0usize; pk];
    let mut pat_in = vec![0usize; pk];
    for e in pattern.edge_references() {
        let (si, ti) = (e.source().index(), e.target().index());
        if si == ti {
            continue; // self-loops excluded from degree prune / neighbor lists
        }
        pnbr[si].push(ti);
        pnbr[ti].push(si);
        if directed {
            pat_out[si] += 1;
            pat_in[ti] += 1;
        }
    }
    if !directed {
        for v in 0..pk {
            pat_out[v] = pnbr[v].len();
            pat_in[v] = pnbr[v].len();
        }
    }

    // Connected-first greedy order: pick the unplaced vertex with the most edges to the
    // already-placed set (tie-break on total degree), so a component is grown from a
    // seed and every non-seed vertex arrives with at least one binding constraint.
    let mut order = Vec::with_capacity(pk);
    let mut placed = vec![false; pk];
    while order.len() < pk {
        let mut best = None;
        let mut best_score = (-1i64, -1i64);
        for v in 0..pk {
            if placed[v] {
                continue;
            }
            let conn = pnbr[v].iter().filter(|&&u| placed[u]).count() as i64;
            let deg = pnbr[v].len() as i64;
            if (conn, deg) > best_score {
                best_score = (conn, deg);
                best = Some(v);
            }
        }
        let v = best.expect("an unplaced vertex must exist while order is incomplete");
        placed[v] = true;
        order.push(v);
    }
    let mut pos_of = vec![0usize; pk];
    for (p, &v) in order.iter().enumerate() {
        pos_of[v] = p;
    }

    let mut cons: Vec<Vec<Cons>> = (0..pk).map(|_| Vec::new()).collect();
    for e in pattern.edge_references() {
        let (si, ti, ei) = (e.source().index(), e.target().index(), e.id());
        if si == ti {
            cons[pos_of[si]].push(Cons {
                other: si,
                dir: Dir::SelfLoop,
                edge: ei,
            });
            continue;
        }
        let (pi, pj) = (pos_of[si], pos_of[ti]);
        if pi < pj {
            // `si` placed earlier; the current vertex at position `pj` is `ti`, and the
            // pattern edge `si → ti` requires host `assign[si] → assign[ti]`.
            cons[pj].push(Cons {
                other: si,
                dir: Dir::FromOther,
                edge: ei,
            });
        } else {
            // `ti` placed earlier; the current vertex at position `pi` is `si`.
            cons[pi].push(Cons {
                other: ti,
                dir: Dir::ToOther,
                edge: ei,
            });
        }
    }
    Plan {
        order,
        cons,
        pat_out,
        pat_in,
    }
}

/// Live backtracking state, threaded through the search so the recursion body stays a
/// single method rather than a free function with a dozen arguments.
struct Search<'a, Np, Ep, Nh, Eh, Ty, NM, EM> {
    pattern: &'a Graph<Np, Ep, Ty>,
    host: &'a Graph<Nh, Eh, Ty>,
    plan: &'a Plan,
    host_out: Vec<usize>,
    host_in: Vec<usize>,
    node_match: NM,
    edge_match: EM,
    assign: Vec<usize>,
    used: Vec<bool>,
    count_only: bool,
    count: usize,
    out: Vec<Vec<usize>>,
}

impl<Np, Ep, Nh, Eh, Ty, NM, EM> Search<'_, Np, Ep, Nh, Eh, Ty, NM, EM>
where
    Ty: EdgeType,
    NM: FnMut(&Np, &Nh) -> bool,
    EM: FnMut(&Ep, &Eh) -> bool,
{
    fn go(&mut self, pos: usize) {
        let pk = self.plan.order.len();
        if pos == pk {
            self.count += 1;
            if !self.count_only {
                self.out.push(self.assign.clone());
            }
            return;
        }
        let pv = self.plan.order[pos];
        let nh = self.host.node_count();
        for hv in 0..nh {
            if self.used[hv] {
                continue;
            }
            // Necessary-condition degree prune (self-loops excluded from both sides).
            if self.plan.pat_out[pv] > self.host_out[hv] || self.plan.pat_in[pv] > self.host_in[hv]
            {
                continue;
            }
            if !(self.node_match)(
                &self.pattern[NodeIndex::new(pv)],
                &self.host[NodeIndex::new(hv)],
            ) {
                continue;
            }
            let mut ok = true;
            for c in &self.plan.cons[pos] {
                let ei = match c.dir {
                    Dir::SelfLoop => self.host.find_edge(NodeIndex::new(hv), NodeIndex::new(hv)),
                    Dir::FromOther => self
                        .host
                        .find_edge(NodeIndex::new(self.assign[c.other]), NodeIndex::new(hv)),
                    Dir::ToOther => self
                        .host
                        .find_edge(NodeIndex::new(hv), NodeIndex::new(self.assign[c.other])),
                };
                match ei {
                    None => {
                        ok = false;
                        break;
                    }
                    Some(hei) => {
                        if !(self.edge_match)(&self.pattern[c.edge], &self.host[hei]) {
                            ok = false;
                            break;
                        }
                    }
                }
            }
            if !ok {
                continue;
            }
            self.assign[pv] = hv;
            self.used[hv] = true;
            self.go(pos + 1);
            self.used[hv] = false;
        }
    }
}

/// Per-node undirected degree of the host (self-loops excluded), used for pruning.
fn host_degrees<Nh, Eh, Ty: EdgeType>(host: &Graph<Nh, Eh, Ty>) -> (Vec<usize>, Vec<usize>) {
    let nh = host.node_count();
    let directed = Ty::is_directed();
    let mut out = vec![0usize; nh];
    let mut inc = vec![0usize; nh];
    for h in 0..nh {
        let v = NodeIndex::new(h);
        if directed {
            out[h] = host
                .neighbors_directed(v, Direction::Outgoing)
                .filter(|u| u.index() != h)
                .count();
            inc[h] = host
                .neighbors_directed(v, Direction::Incoming)
                .filter(|u| u.index() != h)
                .count();
        } else {
            let d = host.neighbors(v).filter(|u| u.index() != h).count();
            out[h] = d;
            inc[h] = d;
        }
    }
    (out, inc)
}

/// All monomorphisms (non-induced subgraph matches) of `pattern` in `host`, each a
/// vector indexed by pattern node with the matched host node index.
///
/// A monomorphism is an **injective, edge-preserving** map: distinct pattern nodes go to
/// distinct host nodes, and every pattern edge maps to a host edge in the matching
/// direction (for a directed `Ty`) — but **extra** host edges among the image are
/// allowed (unlike the induced [`induced_matches`], which rejects them). So the induced
/// matches are exactly the subset of these whose image induces no extra edges.
///
/// See the [module docs](self) for the raw-embedding counting semantics (pattern
/// automorphisms are not deduped; divide the count by `|Aut(pattern)|` for distinct
/// occurrences).
///
/// `node_match` / `edge_match` gate a pattern element against a host element (return
/// `true` to allow); pass `|_, _| true` to match on structure alone.
#[must_use]
pub fn monomorphisms<Np, Ep, Nh, Eh, Ty, NM, EM>(
    pattern: &Graph<Np, Ep, Ty>,
    host: &Graph<Nh, Eh, Ty>,
    node_match: NM,
    edge_match: EM,
) -> Vec<Vec<usize>>
where
    Ty: EdgeType,
    NM: FnMut(&Np, &Nh) -> bool,
    EM: FnMut(&Ep, &Eh) -> bool,
{
    let plan = plan(pattern);
    let (host_out, host_in) = host_degrees(host);
    let pk = pattern.node_count();
    let nh = host.node_count();
    let mut search = Search {
        pattern,
        host,
        plan: &plan,
        host_out,
        host_in,
        node_match,
        edge_match,
        assign: vec![0usize; pk],
        used: vec![false; nh],
        count_only: false,
        count: 0,
        out: Vec::new(),
    };
    search.go(0);
    search.out
}

/// All monomorphisms of `pattern` in `host` on structure alone (no predicates). See
/// [`monomorphisms`].
#[must_use]
pub fn monomorphisms_unlabelled<Np, Ep, Nh, Eh, Ty>(
    pattern: &Graph<Np, Ep, Ty>,
    host: &Graph<Nh, Eh, Ty>,
) -> Vec<Vec<usize>>
where
    Ty: EdgeType,
{
    monomorphisms(pattern, host, |_, _| true, |_, _| true)
}

/// Count monomorphisms of `pattern` in `host`, streaming (embeddings are counted, never
/// materialized). See [`monomorphisms`] for the raw-embedding semantics.
#[must_use]
pub fn count_monomorphisms<Np, Ep, Nh, Eh, Ty, NM, EM>(
    pattern: &Graph<Np, Ep, Ty>,
    host: &Graph<Nh, Eh, Ty>,
    node_match: NM,
    edge_match: EM,
) -> usize
where
    Ty: EdgeType,
    NM: FnMut(&Np, &Nh) -> bool,
    EM: FnMut(&Ep, &Eh) -> bool,
{
    let plan = plan(pattern);
    let (host_out, host_in) = host_degrees(host);
    let pk = pattern.node_count();
    let nh = host.node_count();
    let mut search = Search {
        pattern,
        host,
        plan: &plan,
        host_out,
        host_in,
        node_match,
        edge_match,
        assign: vec![0usize; pk],
        used: vec![false; nh],
        count_only: true,
        count: 0,
        out: Vec::new(),
    };
    search.go(0);
    search.count
}

/// Count monomorphisms of `pattern` in `host` on structure alone. See
/// [`count_monomorphisms`].
#[must_use]
pub fn count_monomorphisms_unlabelled<Np, Ep, Nh, Eh, Ty>(
    pattern: &Graph<Np, Ep, Ty>,
    host: &Graph<Nh, Eh, Ty>,
) -> usize
where
    Ty: EdgeType,
{
    count_monomorphisms(pattern, host, |_, _| true, |_, _| true)
}
