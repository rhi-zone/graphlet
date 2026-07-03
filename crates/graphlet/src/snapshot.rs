//! The generic boundary: an owned `O(V+E)` adjacency snapshot.
//!
//! Every petgraph graph flavour is reduced, once, to `0..n` neighbor lists at the
//! constructor. All enumeration and labelling then runs on plain indices; genericity
//! is confined to this one seam. petgraph's only `O(1)` borrowing adjacency probe is
//! its `O(V²)` adjacency matrix, infeasible at scale — so the census core owns a
//! snapshot rather than borrowing the graph live.

use std::collections::HashSet;

use petgraph::visit::{IntoNeighborsDirected, IntoNodeIdentifiers, NodeCount, NodeIndexable};
use petgraph::Direction;

/// The single trait-bound set the census substrate is generic over.
///
/// Satisfied by `Graph` and `StableGraph`, for both `Directed` and `Undirected`,
/// with arbitrary node/edge weights (weights are never read).
///
/// # Input treated as a simple undirected graph
///
/// The snapshot normalizes every input to a *simple, undirected* graph, so the whole
/// census substrate shares one precondition:
///
/// - **Directed → undirected.** In- and out-neighborhoods are unioned; a directed
///   graph is analyzed on its underlying undirected structure.
/// - **Self-loops stripped.** An edge `v–v` contributes nothing (a vertex is never its
///   own neighbor). This is *silent* on the host side (unlike [`crate::catalog::Pattern::new`],
///   which rejects self-loop pattern edges outright).
/// - **Parallel edges deduped.** Multiple edges between the same pair — including a
///   directed reciprocal pair `a→b, b→a` — collapse to a single undirected edge.
pub trait GraphAdapter:
    IntoNodeIdentifiers + IntoNeighborsDirected + NodeIndexable + NodeCount + Copy
{
}

impl<G> GraphAdapter for G where
    G: IntoNodeIdentifiers + IntoNeighborsDirected + NodeIndexable + NodeCount + Copy
{
}

/// An owned adjacency snapshot in `0..n` index space, plus the map back to the host
/// `NodeId`s. Built in `O(V+E)`; independent of the number of subgraph instances.
#[derive(Clone, Debug)]
pub struct Snapshot<N> {
    /// `ids[i]` is the host `NodeId` for index `i`.
    ids: Vec<N>,
    /// `adj[i]` is the sorted undirected neighbor indices of `i`.
    adj: Vec<Vec<usize>>,
    /// `nbr[i]` is the same neighborhood as a membership set.
    nbr: Vec<HashSet<usize>>,
}

impl<N: Copy> Snapshot<N> {
    /// Reduce any [`GraphAdapter`] to an owned undirected adjacency snapshot.
    ///
    /// The result is a *simple* graph: self-loops are stripped and parallel edges
    /// (including directed reciprocals) are deduped; directed inputs are unioned into
    /// their undirected structure. See [`GraphAdapter`] for the full precondition.
    ///
    /// Raw host slot indices are compacted to a dense `0..n` space, so `StableGraph`
    /// inputs with removed nodes (holes in the slot numbering) are handled correctly.
    pub fn new<G>(g: G) -> Self
    where
        G: GraphAdapter<NodeId = N>,
    {
        // `node_count()` is the number of *live* nodes; `to_index` returns a raw slot
        // that, for a `StableGraph` with holes, can exceed `node_count()`. Size the
        // remap by `node_bound()` (an upper bound on raw slots) and compact live slots
        // to a dense `0..n`, so downstream index space is hole-free.
        let bound = g.node_bound();
        let mut compact = vec![usize::MAX; bound];
        let mut ids: Vec<N> = Vec::new();
        for v in g.node_identifiers() {
            compact[g.to_index(v)] = ids.len();
            ids.push(v);
        }
        let n = ids.len();
        let mut adj = vec![Vec::new(); n];
        for v in g.node_identifiers() {
            let vi = compact[g.to_index(v)];
            for dir in [Direction::Outgoing, Direction::Incoming] {
                for u in g.neighbors_directed(v, dir) {
                    let ui = compact[g.to_index(u)];
                    if ui != vi && !adj[vi].contains(&ui) {
                        adj[vi].push(ui);
                    }
                }
            }
        }
        for row in &mut adj {
            row.sort_unstable();
        }
        let nbr = adj.iter().map(|r| r.iter().copied().collect()).collect();
        Snapshot { ids, adj, nbr }
    }

    /// Number of nodes.
    #[inline]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether the graph is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Host `NodeId` for an index.
    #[inline]
    pub fn id(&self, i: usize) -> N {
        self.ids[i]
    }

    /// Undirected neighbor indices of `i`.
    #[inline]
    pub fn neighbors(&self, i: usize) -> &[usize] {
        &self.adj[i]
    }

    /// Whether indices `a` and `b` are adjacent.
    #[inline]
    pub fn adjacent(&self, a: usize, b: usize) -> bool {
        self.nbr[a].contains(&b)
    }
}
