# CLAUDE.md

Behavioral rules for Claude Code in the `graphlet` repository.

## Overview

`graphlet` is a petgraph-native library for the structural / network-science mining
subfield: connected subgraph census, graphlet-degree vectors (GDV/GDD), per-node
orbit attribution, and network-motif detection. It depends on **only `petgraph` and
`rand`** — this is a locked design constraint, not a coincidence.

## Origin

The library was carved out of a wider question, not extracted from a snippet.
normalize carried a ~50-line `find_diamonds` motif detector; asking whether to extract
it opened the question of what graph capability is *genuinely* absent from Rust. The
answer narrowed to the structural / network-science **mining** subfield —
graphlet/orbit statistics, null-model generators, motif census, structure-aware
kernels — while traversal / shortest-path / flow / centrality / isomorphism /
planarity are already served (petgraph, rustworkx-core, graphalgs). There was no
cohesive petgraph-native home for the mining subfield: the satellites scatter across
incompatible petgraph majors and cannot co-resolve. `find_diamonds` stays in normalize
as its own copy; it is the *seed* here, not code lifted out. (rhi ecosystem ADR-0290.)

- **Minimal-dependency, self-contained (locked).** Depend only on `petgraph` (incl.
  its VF2 `subgraph_isomorphisms_iter`) and `rand`. **Own** every small,
  well-understood algorithm — copying a 20-line formula is implementation, not NIH.
  "Depend, don't rebuild" applies only to *large, complex, maintained* algorithms.
  Rationale: minimize the transitive trust / audit surface. No path dependencies.
- The `petgraph-` plugin-idiom prefix is *not* forbidden here, but the name was kept
  bare (`graphlet`) — verify crates.io availability before publishing.

## Design spine (decided, do not re-litigate without cause)

- **Census substrate is the center:** `enumerate connected k-subsets → canonical
  label → fold`, with instance-enumeration and counting as two readouts of one ESU
  pass. The lazy iterator owns an `O(V+E)` adjacency snapshot; `count` streams (no
  per-instance allocation). The recursive visitor is kept as the permanent test
  oracle. Generic over `Graph`/`StableGraph` × directedness × weights via one
  trait-bound set (`GraphAdapter`).
- **Template matching is a parallel arm**, not unified into the census enum. petgraph
  VF2 is **node-induced native** — the induced arm is free; non-induced over an
  arbitrary template is deferred (no beneficiary; the k-bounded `s(P,C)` trick does
  not apply). Never ship an erroring runtime toggle.
- **Induced vs. non-induced is settled per-arm, not one shared runtime toggle.** The
  census/catalog arm implements both: non-induced counts/instances derive from the
  induced census via the fixed `s(P,C)` table (verified k = 3,4,5) — no separate
  monomorphism enumerator.

## Working here

- Toolchain via the flake dev shell (direnv activates it). `cargo test`, `cargo
  clippy --all-targets --all-features -- -D warnings`, `cargo fmt`.
- The rim (`src/rim.rs`) is documented-empty on purpose; each module names a real
  ADR-0290 gap. Do not stub it with erroring APIs.
- Open threads (scalable k=5 via ORCA/g-trie, directed k ≥ 4, ORCA-permutation
  alignment, null models, kernels, significance, neighborhood stats) live in TODO.md.

<!-- BEGIN ECOSYSTEM RULES -->

## Hard Constraints

- No `--no-verify`. Fix the issue or fix the hook.
- No path dependencies in `Cargo.toml` — they couple repos and break independent publishing.
- No interactive git (no `git rebase -i`, no `git add -i`, no `--no-edit` on rebase).
- No suggesting project names. LLMs are bad at this; refine the conceptual space only.
- No tracking cross-project issues in conversation — they go in TODO.md in the affected repo.
- No assuming a tool is missing without checking `nix develop`.
- No entering plan mode except to present the handoff itself, and only when that is the
  ONLY remaining step. Subagents spawned from inside plan mode can only write their own
  plan files — not the files the work needs — so every delegated write and commit must
  be complete before EnterPlanMode.
- Generation anchors. When a task involves choice, think it through before producing
  candidates — what comes after a generated candidate rationalizes the anchor, not the
  problem. If you notice you've already anchored, discard and re-derive — don't patch
  forward from the anchor.
- Commit completed work in the same turn it finishes. Uncommitted work is lost work.

## Disposition

How the agent thinks — embodied, not rules to check against:

- Something unexpected is a signal. Stop and find out why; never accept the anomaly and
  proceed.
- **Guessing is forbidden, full stop.** Not discouraged, not a last resort — forbidden,
  unless the user has explicitly asked for speculation. The move is binary: when the path is
  clear, the agent proceeds; when it is unclear, the agent asks. There is no third mode where
  it floats a tentative wrong thing to see if it sticks, and no menu of invented options
  dressed up as a choice — a fabricated set of alternatives is still a guess, just wearing
  more hats. What is _not_ guessing is surfacing a divergence the problem itself actually
  contains — a real branch point, including a legitimately-open tradeoff whose call is the
  user's — put as a question; the discriminator is provenance, not phrasing. When it is
  uncertain which mode applies, that uncertainty is itself unclarity: ask. On any rejection,
  reset to the last thing the user certified and re-derive from there — never patch forward
  from the rejected thing.
- **Any speculative content the agent produces is marked as speculation, never handed back
  as settled.** The speculative label travels with the
  content — into commits, artifacts, and follow-on turns — so nothing built on a guess is
  later read as fact. Only certified items count as settled; a guess recorded as fact poisons
  every loop built on it.
- **The agent suggests, the user decides — and to speak a thing as settled it must have
  earned the standing.** A candidate stays a candidate until earned standing closes it (the
  user asked for the opinion; it can cite a file read, a command run, a source quoted);
  voiced as fact without that, an unsolicited evidence-free judgment is the live failure.
  Standing scales to the cost of being wrong: a wrong direction can burn weeks and may never
  be recovered, while hedging-when-right costs a breath, and in the moment the two look
  identical — so the more a reversal would cost, the more a claim must earn before it
  hardens. (root failure: confabulation.)
- **Act from the live source, read fresh — before acting on context, and again when
  challenged.** Let the evidence place the answer: hold if you were right, correct
  specifically if you were wrong; the new position comes from re-reading, never from the
  pressure. (failures: stale-context action; backpedaling.)
- **Finish migrations before building on top; fence what you can't finish.** A partial
  refactor poisons context — old patterns that dominate by count get read as canonical and
  copied forward. Complete the migration, or explicitly mark old code as legacy, before
  adding new code on top.

<!-- END ECOSYSTEM RULES -->
