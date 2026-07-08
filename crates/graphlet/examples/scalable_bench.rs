//! Reproducible wall-clock comparison of the scalable fast path
//! ([`graphlet::rim::scalable::fast_graphlet_degree_vectors`]) against the exact
//! [`graphlet::graphlet_degree_vectors`], both computing all 73 orbits (k = 2..=5) on
//! undirected G(n, p) random graphs.
//!
//! Run in release (the exact path is expensive enough that a debug build is
//! misleading):
//!
//! ```text
//! cargo run --release -p graphlet --example scalable_bench
//! ```
//!
//! Uses a fixed seeded RNG so the graphs (and therefore the timings' relative shape)
//! are reproducible across runs and machines; absolute wall-clock numbers will still
//! vary with hardware.
//!
//! The speedup is not a fixed multiplier: it grows with `n` and with edge density.
//! The exact path canonicalizes every connected k-subset it visits (a `k!`-permutation
//! search per instance, dominant at k=5), so its cost climbs steeply as the number of
//! connected 5-subsets grows with `n` and `p`. The fast path never canonicalizes a
//! 5-subset at all (see the module docs on `graphlet::rim::scalable`): it derives all
//! 73 orbits from per-vertex/per-edge/per-4-subset tallies, so its own cost grows far
//! more slowly.
//!
//! That asymmetry is the whole story, so the benchmark makes it explicit. The exact
//! path is only timed while it stays under a wall-clock budget ([`EXACT_BUDGET_SECS`]);
//! past the first point that would exceed it the exact column reads `impractical` and
//! only the fast path is timed, so the fast path can be pushed to sizes where the exact
//! path would take minutes. The rows are ordered by `n` at each density so the widening
//! gap, and the point where exact drops out, are both visible directly. Do not read a
//! single row's ratio as representative of the whole crate.

use std::time::Instant;

use petgraph::graph::{NodeIndex, UnGraph};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use graphlet::rim::scalable::fast_graphlet_degree_vectors;
use graphlet::{graphlet_degree_vectors, Registry};

/// Wall-clock budget for the exact path. Once a point's exact timing exceeds this, no
/// larger point is timed on the exact path (its cost only grows), and the exact column
/// reads `impractical` from there on while the fast path keeps being timed.
const EXACT_BUDGET_SECS: f64 = 45.0;

/// Build an undirected G(n, p) Erdos-Renyi random graph with a caller-supplied RNG,
/// for reproducibility independent of any other generator in the crate.
fn gnp(n: usize, p: f64, rng: &mut impl Rng) -> UnGraph<(), ()> {
    let mut g = UnGraph::with_capacity(n, 0);
    for _ in 0..n {
        g.add_node(());
    }
    for i in 0..n {
        for j in (i + 1)..n {
            if rng.gen_bool(p) {
                g.add_edge(NodeIndex::new(i), NodeIndex::new(j), ());
            }
        }
    }
    g
}

fn main() {
    // Fixed seed: reproducible graphs across machines and runs.
    let mut rng = StdRng::seed_from_u64(0xDECA_5CA1_AB1E_u64);
    // Registry construction is shared, deterministic, and reused across every graph
    // below (as intended by both APIs), so it is built once, outside every timed
    // region.
    let reg = Registry::build();

    // Each density gets its own ascending-n sweep so the exact-path budget can retire
    // it independently: the dense (p = 0.3) sweep blows the budget at a smaller n than
    // the sparse (p = 0.1) sweep, and the fast path is timed at every point regardless.
    let sweeps: &[(f64, &[usize])] = &[
        (0.1, &[20, 40, 60, 80, 100, 150, 200]),
        (0.3, &[20, 40, 60, 80, 100, 150]),
    ];

    println!(
        "{:>4} {:>5} {:>14} {:>12} {:>10}",
        "n", "p", "exact_ms", "fast_ms", "speedup"
    );

    for &(p, sizes) in sweeps {
        // Once the exact path exceeds the budget at some n, skip it for all larger n in
        // this sweep (cost is monotone in n) and keep timing the fast path alone.
        let mut exact_retired = false;
        for &n in sizes {
            let g = gnp(n, p, &mut rng);

            let exact_dur = if exact_retired {
                None
            } else {
                let t0 = Instant::now();
                let exact = graphlet_degree_vectors(&g, &reg);
                let d = t0.elapsed();
                let fast_check = fast_graphlet_degree_vectors(&g, &reg);
                // Sanity check: the two paths must agree, or the benchmark is
                // measuring nothing meaningful.
                assert_eq!(exact.len(), fast_check.len());
                for i in 0..exact.len() {
                    assert_eq!(
                        exact.row(i),
                        fast_check.row(i),
                        "fast/exact GDV mismatch at n={n}, p={p}, node {i}"
                    );
                }
                if d.as_secs_f64() > EXACT_BUDGET_SECS {
                    exact_retired = true;
                }
                Some(d)
            };

            let t1 = Instant::now();
            let _fast = fast_graphlet_degree_vectors(&g, &reg);
            let fast_dur = t1.elapsed();

            match exact_dur {
                Some(ex) => {
                    let speedup = ex.as_secs_f64() / fast_dur.as_secs_f64().max(1e-12);
                    println!(
                        "{:>4} {:>5.2} {:>14.3} {:>12.3} {:>9.1}x",
                        n,
                        p,
                        ex.as_secs_f64() * 1000.0,
                        fast_dur.as_secs_f64() * 1000.0,
                        speedup
                    );
                }
                None => {
                    println!(
                        "{:>4} {:>5.2} {:>14} {:>12.3} {:>10}",
                        n,
                        p,
                        "impractical",
                        fast_dur.as_secs_f64() * 1000.0,
                        "-"
                    );
                }
            }
        }
    }
}
