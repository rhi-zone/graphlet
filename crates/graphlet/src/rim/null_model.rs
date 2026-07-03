//! Null-model / random-graph generators for significance testing.
//!
//! All generators return [`UnGraph<(), ()>`] (petgraph undirected unit graph) and
//! accept `rng: &mut impl Rng` so the caller controls reproducibility via a seeded
//! RNG. Runtime dependencies: `petgraph` and `rand` only.

use petgraph::graph::{NodeIndex, UnGraph};
use petgraph::visit::EdgeRef;
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::HashSet;

/// Canonical undirected edge key: always `(min, max)`.
#[inline]
fn ekey(a: usize, b: usize) -> (usize, usize) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

// ---------------------------------------------------------------------------
// Configuration model
// ---------------------------------------------------------------------------

/// Configuration model — raw (multigraph) variant.
///
/// Constructs a random multigraph whose stub-degree sequence exactly matches
/// `degree_seq` using stub-pairing (Molloy–Reed 1995):
///
/// 1. Build a flat stub list where node `i` appears `degree_seq[i]` times.
/// 2. Shuffle the list uniformly.
/// 3. Pair consecutive stubs `(stubs[2k], stubs[2k+1])` as edges.
///
/// **May produce self-loops and parallel edges.** Use
/// [`configuration_model_simple`] to erase them (at the cost of approximate
/// realized degrees — see that function's doc).
///
/// **Stub-degree guarantee:** for every node `i`, counting self-loops as 2
/// (each self-loop consumes two stubs):
/// ```text
/// Σ_{e incident to i, counting self-loops twice} 1  ==  degree_seq[i]
/// ```
/// This is exact and deterministic — it follows from the construction.
///
/// # Panics
///
/// Panics if `degree_seq.iter().sum::<usize>()` is odd (stubs cannot be fully
/// paired).
pub fn configuration_model(degree_seq: &[usize], rng: &mut impl Rng) -> UnGraph<(), ()> {
    let n = degree_seq.len();
    let sum: usize = degree_seq.iter().sum();
    assert!(
        sum.is_multiple_of(2),
        "configuration model requires an even stub sum; got sum={sum}"
    );

    let mut g = UnGraph::with_capacity(n, sum / 2);
    for _ in 0..n {
        g.add_node(());
    }

    let mut stubs: Vec<usize> = degree_seq
        .iter()
        .enumerate()
        .flat_map(|(i, &d)| std::iter::repeat_n(i, d))
        .collect();
    stubs.shuffle(rng);

    for pair in stubs.chunks_exact(2) {
        g.add_edge(NodeIndex::new(pair[0]), NodeIndex::new(pair[1]), ());
    }
    g
}

/// Configuration model — simple (loop-free, no parallel edges) variant.
///
/// Runs the raw configuration model then erases every self-loop and every
/// parallel edge beyond the first. The erased stubs are discarded, so the
/// realized degree of node `i` after simplification satisfies:
///
/// ```text
/// realized_degree[i]  <=  degree_seq[i]
/// ```
///
/// **Erasure bound:** in expectation the fraction of erased edges is
/// `O(Σ d_i² / (Σ d_i)²)`, which vanishes for sparse sequences with bounded
/// second moment (Erdős–Rényi regime). For heavy-tailed or very dense sequences
/// the realized degrees can deviate substantially from `degree_seq`.
///
/// # Panics
///
/// Same as [`configuration_model`]: panics if the degree-sequence sum is odd.
pub fn configuration_model_simple(degree_seq: &[usize], rng: &mut impl Rng) -> UnGraph<(), ()> {
    let raw = configuration_model(degree_seq, rng);
    let n = raw.node_count();
    let mut g = UnGraph::with_capacity(n, raw.edge_count());
    for _ in 0..n {
        g.add_node(());
    }
    let mut seen: HashSet<(usize, usize)> = HashSet::new();
    for e in raw.edge_references() {
        let a = e.source().index();
        let b = e.target().index();
        if a == b {
            continue; // drop self-loops
        }
        if seen.insert(ekey(a, b)) {
            g.add_edge(NodeIndex::new(a), NodeIndex::new(b), ());
        }
    }
    g
}

// ---------------------------------------------------------------------------
// Double-edge-swap
// ---------------------------------------------------------------------------

/// Degree-preserving edge randomization by double-edge swap.
///
/// Starting from a *simple* undirected graph (no self-loops, no parallel edges),
/// attempts `n_swaps` random swaps. Each attempt:
///
/// 1. Picks two distinct edges `a–b` and `c–d` uniformly at random.
/// 2. Randomly chooses rewiring `a–c, b–d` **or** `a–d, b–c`.
/// 3. **Rejects** the swap if any candidate endpoint pair is a self-loop, if a
///    candidate edge already exists (parallel edge), or if the swap is a no-op
///    (a new edge coincides with an old one).
///
/// **Invariant:** every node's degree is exactly preserved across all accepted
/// swaps. The output is always a simple undirected graph.
///
/// Returns a new graph (the input is not mutated). If the input has fewer than
/// 2 edges, the graph is returned unchanged.
pub fn double_edge_swap(
    graph: &UnGraph<(), ()>,
    n_swaps: usize,
    rng: &mut impl Rng,
) -> UnGraph<(), ()> {
    let n = graph.node_count();
    let mut edges: Vec<(usize, usize)> = graph
        .edge_references()
        .map(|e| (e.source().index(), e.target().index()))
        .collect();
    let m = edges.len();

    let mut edge_set: HashSet<(usize, usize)> = edges.iter().map(|&(a, b)| ekey(a, b)).collect();

    if m >= 2 {
        for _ in 0..n_swaps {
            // Sample two distinct edge indices.
            let i = rng.gen_range(0..m);
            let mut j = rng.gen_range(0..m - 1);
            if j >= i {
                j += 1;
            }

            let (a, b) = edges[i];
            let (c, d) = edges[j];

            let (new_e1, new_e2) = if rng.gen_bool(0.5) {
                ((a, c), (b, d))
            } else {
                ((a, d), (b, c))
            };

            // Reject: self-loops.
            if new_e1.0 == new_e1.1 || new_e2.0 == new_e2.1 {
                continue;
            }

            let k1 = ekey(new_e1.0, new_e1.1);
            let k2 = ekey(new_e2.0, new_e2.1);
            let old1 = ekey(a, b);
            let old2 = ekey(c, d);

            // Reject: no-op (a new edge equals an old one, so structure unchanged).
            if k1 == old1 || k1 == old2 || k2 == old1 || k2 == old2 {
                continue;
            }

            // Reject: the two new edges are identical (would create a self-loop or
            // parallel edge implicitly).
            if k1 == k2 {
                continue;
            }

            // Reject: parallel edge would be created.
            if edge_set.contains(&k1) || edge_set.contains(&k2) {
                continue;
            }

            // Apply.
            edge_set.remove(&old1);
            edge_set.remove(&old2);
            edge_set.insert(k1);
            edge_set.insert(k2);
            edges[i] = new_e1;
            edges[j] = new_e2;
        }
    }

    let mut g = UnGraph::with_capacity(n, m);
    for _ in 0..n {
        g.add_node(());
    }
    for &(a, b) in &edges {
        g.add_edge(NodeIndex::new(a), NodeIndex::new(b), ());
    }
    g
}

// ---------------------------------------------------------------------------
// Watts-Strogatz
// ---------------------------------------------------------------------------

/// Watts–Strogatz small-world graph.
///
/// Constructs a random graph on `n` nodes:
///
/// 1. **Ring lattice:** each node `i` connects to `(i ± j) mod n` for
///    `j ∈ 1..=k/2`, producing exactly `n·k/2` distinct undirected edges.
/// 2. **Rewiring:** for each edge in the lattice (node-major iteration order),
///    independently rewire it with probability `p` to a uniformly chosen target
///    node, rejecting the source itself and any already-connected target. If no
///    valid target is found within `n` attempts, the original edge is kept.
///
/// **Invariants:**
/// - `graph.node_count() == n`
/// - `graph.edge_count() == n * k / 2` (rewiring preserves the edge count:
///   each rewired edge is replaced by exactly one new edge; rejections keep the
///   original).
/// - `p = 0.0`: pure ring lattice — edge `(i, (i + j) mod n)` exists for every
///   `i` and `j ∈ 1..=k/2`.
/// - `p = 1.0`: near-random; no ring-lattice structure is guaranteed.
///
/// # Panics
///
/// Panics if `k` is odd, `n ≤ k`, or `n < 2`.
pub fn watts_strogatz(n: usize, k: usize, p: f64, rng: &mut impl Rng) -> UnGraph<(), ()> {
    assert!(
        k.is_multiple_of(2),
        "k must be even for Watts–Strogatz; got k={k}"
    );
    assert!(
        n > k,
        "n must be greater than k for the ring lattice; got n={n}, k={k}"
    );
    assert!(n >= 2, "n must be at least 2; got n={n}");

    let half_k = k / 2;

    // Build ring-lattice edge list.
    // For n > k = 2·half_k, farthest neighbor is < n/2 away, so no two (i,j)
    // pairs produce the same undirected edge — the list has exactly n·half_k
    // distinct entries.
    let mut edges: Vec<(usize, usize)> = Vec::with_capacity(n * half_k);
    let mut edge_set: HashSet<(usize, usize)> = HashSet::with_capacity(n * half_k);
    for i in 0..n {
        for j in 1..=half_k {
            let nb = (i + j) % n;
            edges.push((i, nb));
            edge_set.insert(ekey(i, nb));
        }
    }

    // Rewire.
    for edge in &mut edges {
        if p > 0.0 && rng.gen::<f64>() < p {
            let (u, old_v) = *edge;
            let old_k = ekey(u, old_v);
            let mut new_v = None;
            for _ in 0..n {
                let cand = rng.gen_range(0..n);
                if cand != u && !edge_set.contains(&ekey(u, cand)) {
                    new_v = Some(cand);
                    break;
                }
            }
            if let Some(v) = new_v {
                edge_set.remove(&old_k);
                edge_set.insert(ekey(u, v));
                *edge = (u, v);
            }
            // If no valid target found, keep the original edge unchanged.
        }
    }

    let mut g = UnGraph::with_capacity(n, n * half_k);
    for _ in 0..n {
        g.add_node(());
    }
    for &(a, b) in &edges {
        g.add_edge(NodeIndex::new(a), NodeIndex::new(b), ());
    }
    g
}

// ---------------------------------------------------------------------------
// LFR benchmark
// ---------------------------------------------------------------------------

/// LFR (Lancichinetti–Fortunato–Radicchi) benchmark graph with planted
/// community structure.
///
/// Returns `(graph, community)` where `community[i]` is the 0-indexed
/// community label of node `i`.
///
/// **Algorithm outline:**
///
/// 1. Sample `n` degrees from a discrete power law on `[1, max_degree]` with
///    exponent `gamma`. Adjust one degree by ±1 to make the total even.
/// 2. Generate community sizes from a discrete power law on
///    `[min_community, max_community]` with exponent `beta`, filling `n` nodes
///    greedily. The last community absorbs remainder if it falls short of
///    `min_community`.
/// 3. Shuffle node-to-community assignment.
/// 4. For each node `i` with degree `d[i]`, split:
///    - *internal* `d_in[i] = round((1 − μ) · d[i])`, clamped to
///      `[0, |C_i| − 1]`.
///    - *external* `d_ext[i] = d[i] − d_in[i]`.
/// 5. Within each community, parity-correct `Σ d_in` by bumping one node,
///    then run the raw stub-pairing configuration model. **Self-loops within a
///    community are retained.**
/// 6. Match external stubs across communities greedily: shuffle all external
///    stubs; scan for the next unmatched stub from a *different* community.
///    Unmatched stubs are silently dropped when all remaining stubs belong to
///    the same community (see limitations).
///
/// **Known limitations and deviations from the original paper:**
///
/// - *Degree approximation:* the realized degree distribution approximates
///   the target power law but the mean equals `avg_degree` only in expectation
///   (the parameter is advisory and not enforced).
/// - *Parity perturbation:* for each community with an odd internal-stub sum,
///   one node's `d_in` is bumped by ±1.
/// - *Feasibility clamping:* `d_in[i]` is hard-clamped to `|C_i| − 1`.
/// - *Internal self-loops:* the stub-pairing within communities may produce
///   self-loops. Post-process with [`double_edge_swap`] if simplicity is
///   required.
/// - *External stub failure:* when a community holds > 50 % of the remaining
///   external stubs, greedy matching cannot pair all stubs and the unmatched
///   are dropped. The realized mixing fraction may then be lower than `mu`.
///
/// # Panics
///
/// Panics if `max_degree == 0`, `min_community == 0`,
/// `min_community > max_community`, `n == 0`, or `!(0.0..=1.0).contains(&mu)`.
#[allow(clippy::too_many_arguments)]
pub fn lfr_benchmark(
    n: usize,
    avg_degree: f64,
    max_degree: usize,
    mu: f64,
    gamma: f64,
    beta: f64,
    min_community: usize,
    max_community: usize,
    rng: &mut impl Rng,
) -> (UnGraph<(), ()>, Vec<usize>) {
    assert!(n > 0, "n must be positive");
    assert!(max_degree > 0, "max_degree must be positive");
    assert!(min_community > 0, "min_community must be positive");
    assert!(
        min_community <= max_community,
        "min_community ({min_community}) must be ≤ max_community ({max_community})"
    );
    assert!(
        (0.0..=1.0).contains(&mu),
        "mu must be in [0.0, 1.0]; got {mu}"
    );
    let _ = avg_degree; // advisory only; actual mean depends on power-law samples

    // Step 1: degree sequence.
    let mut degrees: Vec<usize> = (0..n)
        .map(|_| sample_discrete_power_law(1, max_degree, gamma, rng))
        .collect();
    // Ensure even sum for the configuration model.
    if degrees.iter().sum::<usize>() % 2 != 0 {
        let idx = rng.gen_range(0..n);
        if degrees[idx] < max_degree {
            degrees[idx] += 1;
        } else {
            degrees[idx] = degrees[idx].saturating_sub(1);
        }
    }

    // Step 2: community sizes.
    let community_sizes = gen_community_sizes(n, min_community, max_community, beta, rng);
    let num_communities = community_sizes.len();

    // Step 3: assign nodes to communities.
    let mut node_order: Vec<usize> = (0..n).collect();
    node_order.shuffle(rng);
    let mut community = vec![0usize; n];
    {
        let mut start = 0;
        for (c, &sz) in community_sizes.iter().enumerate() {
            for &node in &node_order[start..start + sz] {
                community[node] = c;
            }
            start += sz;
        }
    }

    // Step 4: split into internal / external stubs.
    let mut d_in: Vec<usize> = (0..n)
        .map(|i| {
            let c = community[i];
            let c_size = community_sizes[c];
            let max_in = c_size.saturating_sub(1);
            let raw = ((1.0 - mu) * degrees[i] as f64).round() as usize;
            raw.min(max_in)
        })
        .collect();
    let mut d_ext: Vec<usize> = (0..n).map(|i| degrees[i].saturating_sub(d_in[i])).collect();

    // Step 5: internal edges via configuration model per community.
    // Collect nodes per community.
    let mut community_nodes: Vec<Vec<usize>> = vec![Vec::new(); num_communities];
    for (node, &c) in community.iter().enumerate() {
        community_nodes[c].push(node);
    }

    let mut all_edges: Vec<(usize, usize)> = Vec::new();

    for (c, nodes) in community_nodes.iter().enumerate() {
        if nodes.len() <= 1 {
            continue;
        }
        let mut local_din: Vec<usize> = nodes.iter().map(|&v| d_in[v]).collect();
        // Parity-correct the internal stub sum for this community.
        let sum: usize = local_din.iter().sum();
        if !sum.is_multiple_of(2) {
            // Bump the node with the smallest d_in by 1 (or decrement if at cap).
            let max_in = community_sizes[c].saturating_sub(1);
            if let Some((li, _)) = local_din
                .iter()
                .enumerate()
                .filter(|&(_, &v)| v < max_in)
                .min_by_key(|&(_, &v)| v)
            {
                local_din[li] += 1;
                d_in[nodes[li]] += 1;
                d_ext[nodes[li]] = degrees[nodes[li]].saturating_sub(d_in[nodes[li]]);
            } else if let Some((li, _)) = local_din
                .iter()
                .enumerate()
                .filter(|&(_, &v)| v > 0)
                .min_by_key(|&(_, &v)| v)
            {
                local_din[li] -= 1;
                d_in[nodes[li]] -= 1;
                d_ext[nodes[li]] = degrees[nodes[li]].saturating_sub(d_in[nodes[li]]);
            }
        }
        // Stub-pairing within the community (raw — may produce self-loops).
        let mut stubs: Vec<usize> = local_din
            .iter()
            .enumerate()
            .flat_map(|(li, &d)| std::iter::repeat_n(nodes[li], d))
            .collect();
        stubs.shuffle(rng);
        for pair in stubs.chunks_exact(2) {
            all_edges.push((pair[0], pair[1]));
        }
    }

    // Step 6: external stub matching.
    let mut ext_stubs: Vec<usize> = (0..n)
        .flat_map(|i| std::iter::repeat_n(i, d_ext[i]))
        .collect();
    ext_stubs.shuffle(rng);

    // Greedy scan: pair stub i with the next unmatched stub from a different community.
    let mut matched = vec![false; ext_stubs.len()];
    let mut i = 0;
    while i < ext_stubs.len() {
        if matched[i] {
            i += 1;
            continue;
        }
        let u = ext_stubs[i];
        let cu = community[u];
        let found =
            (i + 1..ext_stubs.len()).find(|&j| !matched[j] && community[ext_stubs[j]] != cu);
        if let Some(j) = found {
            all_edges.push((u, ext_stubs[j]));
            matched[i] = true;
            matched[j] = true;
        }
        i += 1;
    }

    // Build output graph.
    let mut g = UnGraph::with_capacity(n, all_edges.len());
    for _ in 0..n {
        g.add_node(());
    }
    for &(a, b) in &all_edges {
        g.add_edge(NodeIndex::new(a), NodeIndex::new(b), ());
    }
    (g, community)
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Sample from the discrete power law on `{min_val, ..., max_val}` with weights
/// proportional to `x^{-exponent}` via linear CDF scan.
fn sample_discrete_power_law(
    min_val: usize,
    max_val: usize,
    exponent: f64,
    rng: &mut impl Rng,
) -> usize {
    debug_assert!(min_val <= max_val && min_val >= 1);
    let mut total = 0.0f64;
    let mut cdf: Vec<f64> = (min_val..=max_val)
        .map(|x| {
            total += (x as f64).powf(-exponent);
            total
        })
        .collect();
    let r = rng.gen::<f64>() * total;
    // Normalize last entry to avoid floating-point overshoot.
    *cdf.last_mut().unwrap() = f64::INFINITY;
    for (i, &c) in cdf.iter().enumerate() {
        if r <= c {
            return min_val + i;
        }
    }
    max_val
}

/// Generate community sizes summing to exactly `n` by drawing from a discrete power
/// law on `[min_c, max_c]` greedily. The last community absorbs any remainder that
/// falls below `min_c` (documented limitation in [`lfr_benchmark`]).
fn gen_community_sizes(
    n: usize,
    min_c: usize,
    max_c: usize,
    beta: f64,
    rng: &mut impl Rng,
) -> Vec<usize> {
    let mut sizes: Vec<usize> = Vec::new();
    let mut total = 0usize;
    while total < n {
        let remaining = n - total;
        if remaining < min_c {
            // Absorb into the last community or create one.
            match sizes.last_mut() {
                Some(last) => *last += remaining,
                None => sizes.push(remaining),
            }
            break;
        }
        let sz = sample_discrete_power_law(min_c, max_c.min(remaining), beta, rng);
        sizes.push(sz);
        total += sz;
    }
    if sizes.is_empty() {
        sizes.push(n);
    }
    sizes
}
