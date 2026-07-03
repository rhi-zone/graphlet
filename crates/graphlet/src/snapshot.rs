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
/// with arbitrary node/edge weights (weights are never read). Directed graphs are
/// analyzed on their underlying undirected structure.
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
    pub fn new<G>(g: G) -> Self
    where
        G: GraphAdapter<NodeId = N>,
    {
        let n = g.node_count();
        let mut ids: Vec<Option<N>> = vec![None; n];
        let mut adj = vec![Vec::new(); n];
        for v in g.node_identifiers() {
            let vi = g.to_index(v);
            ids[vi] = Some(v);
            for dir in [Direction::Outgoing, Direction::Incoming] {
                for u in g.neighbors_directed(v, dir) {
                    let ui = g.to_index(u);
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
        let ids = ids
            .into_iter()
            .map(|o| o.expect("node_identifiers covers 0..node_count"))
            .collect();
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
