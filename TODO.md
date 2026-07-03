# TODO

Open threads for `graphlet`, from ADR-0290 (rhi ecosystem) and the motif-engine gates.

## Open gates (live, not resolved)

- **Scalable k=5.** Naive per-subset canonicalization is untenable at biological
  scale. Needs ORCA orbit-count equations or a g-trie. (`rim::scalable`)
- **ORCA-permutation alignment.** Internal orbit ids are stable but not ORCA's
  published numbering — a mechanical relabelling. (`rim::scalable`)
- **Throughput benchmark.** Owning-snapshot vs. borrowing traversal throughput was
  never benchmarked (only peak-memory streaming was measured).
- **Directed k ≥ 4.** Deferred by design; k=3 triads are settled. (`rim::directed`)

## Rim — TRUE algorithmic voids (build)

- **Null-model generators** (`rim::null_model`): configuration model, degree-preserving
  double-edge-swap, Watts–Strogatz, LFR. These use `rand`.
- **Graph kernels** (`rim::kernels`): the graphlet kernel reduces to census vectors;
  shortest-path and WL kernels are siblings. Do **not** rebuild the WL hash core —
  observe/use `wl_isomorphism` (NIH-corrected).
- **Motif significance** (`rim::significance`): z-scores / empirical p-values against a
  null-model ensemble.

## Rim — cohesion re-homing (small — own them)

- **Neighborhood statistics** (`rim::neighborhood`): link-prediction indices, degree
  assortativity, rich-club, local/average clustering + triangle counting.
  - **Live threat:** if `franken_networkx` internals ship as a real crates.io crate
    over petgraph types, the cohesion case for re-homing these weakens. The true
    algorithmic voids above are unaffected.

## Deferred by decision (no beneficiary today)

- **Non-induced (monomorphism) arbitrary-template enumerator.** Reserved as a future
  additive method / type-gated entry — never an erroring runtime toggle. Gate: a
  concrete consumer needing non-induced matching of a pattern that is (a) not in the
  catalog and (b) needs actual instances, not counts.

## Pre-publish

- Verify `graphlet` name availability on crates.io (crates.io API was unreachable at
  scaffold time — data-access policy).
