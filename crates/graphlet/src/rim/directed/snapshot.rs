//! Directed adjacency snapshot: the directed analogue of [`crate::snapshot::Snapshot`].
//!
//! Reduces a petgraph digraph to owned `0..n` index space, keeping *both* the raw
//! directed arc relation (asymmetric — `has_arc(a, b)` need not imply `has_arc(b, a)`)
//! and the underlying undirected neighbor union. The latter lets weakly-connected
//! enumeration reuse the same ESU traversal shape as the undirected census substrate:
//! structure comes from the undirected union, class *labelling* comes from the
//! directed arc relation.

use std::collections::HashSet;

use petgraph::visit::{EdgeRef, IntoEdgeReferences, IntoNodeIdentifiers, NodeCount, NodeIndexable};

/// The trait-bound set the directed census substrate needs: a simple directed graph
/// (self-loop arcs ignored; parallel arcs in the same direction deduped), satisfied by
/// petgraph `Graph`/`StableGraph` with `Directed` edge type and arbitrary weights.
pub trait DirectedGraphAdapter:
    IntoNodeIdentifiers + IntoEdgeReferences + NodeIndexable + NodeCount + Copy
{
}

impl<G> DirectedGraphAdapter for G where
    G: IntoNodeIdentifiers + IntoEdgeReferences + NodeIndexable + NodeCount + Copy
{
}

/// An owned directed adjacency snapshot in `0..n` index space.
#[derive(Clone, Debug)]
pub struct DirectedSnapshot<N> {
    ids: Vec<N>,
    /// `arc[i]` sorted out-neighbor indices of `i` (directed).
    arc: Vec<Vec<usize>>,
    /// `arc_set[i]` the same out-neighborhood as a membership set (`has_arc` probe).
    arc_set: Vec<HashSet<usize>>,
    /// `und[i]` sorted undirected neighbor-union indices of `i` (arc either direction).
    und: Vec<Vec<usize>>,
    /// `und_set[i]` the same undirected neighborhood as a membership set.
    und_set: Vec<HashSet<usize>>,
}

impl<N: Copy> DirectedSnapshot<N> {
    /// Reduce any [`DirectedGraphAdapter`] to an owned directed adjacency snapshot.
    ///
    /// Self-loop arcs are dropped; parallel arcs in the same direction are deduped
    /// (their multiplicity carries no structural signal for canonical labelling).
    /// Raw host slot indices are compacted to a dense `0..n` space, so `StableGraph`
    /// inputs with removed nodes (holes in the slot numbering) are handled correctly.
    #[must_use]
    pub fn new<G>(g: G) -> Self
    where
        G: DirectedGraphAdapter<NodeId = N>,
    {
        let bound = g.node_bound();
        let mut compact = vec![usize::MAX; bound];
        let mut ids: Vec<N> = Vec::new();
        for v in g.node_identifiers() {
            compact[g.to_index(v)] = ids.len();
            ids.push(v);
        }
        let n = ids.len();
        let mut arc_set: Vec<HashSet<usize>> = vec![HashSet::new(); n];
        let mut und_set: Vec<HashSet<usize>> = vec![HashSet::new(); n];
        for e in g.edge_references() {
            let a = compact[g.to_index(e.source())];
            let b = compact[g.to_index(e.target())];
            if a == b {
                continue; // self-loop stripped
            }
            arc_set[a].insert(b);
            und_set[a].insert(b);
            und_set[b].insert(a);
        }
        let sorted = |s: &[HashSet<usize>]| -> Vec<Vec<usize>> {
            s.iter()
                .map(|set| {
                    let mut v: Vec<usize> = set.iter().copied().collect();
                    v.sort_unstable();
                    v
                })
                .collect()
        };
        let arc = sorted(&arc_set);
        let und = sorted(&und_set);
        DirectedSnapshot {
            ids,
            arc,
            arc_set,
            und,
            und_set,
        }
    }

    /// Number of nodes.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether the graph is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Host `NodeId` for an index.
    #[inline]
    #[must_use]
    pub fn id(&self, i: usize) -> N {
        self.ids[i]
    }

    /// Sorted out-neighbor indices of `i` (directed).
    #[inline]
    #[must_use]
    pub fn out_neighbors(&self, i: usize) -> &[usize] {
        &self.arc[i]
    }

    /// Whether the arc `a -> b` is present.
    #[inline]
    #[must_use]
    pub fn has_arc(&self, a: usize, b: usize) -> bool {
        self.arc_set[a].contains(&b)
    }

    /// Sorted undirected neighbor-union indices of `i` (arc either direction).
    #[inline]
    #[must_use]
    pub fn undirected_neighbors(&self, i: usize) -> &[usize] {
        &self.und[i]
    }

    /// Whether `a` and `b` are adjacent in the undirected union (arc either direction).
    #[inline]
    #[must_use]
    pub fn undirected_adjacent(&self, a: usize, b: usize) -> bool {
        self.und_set[a].contains(&b)
    }
}
