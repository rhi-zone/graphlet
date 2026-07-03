# Named motifs

The catalog arm queries named motifs over the census substrate. It threads
`Induced` as a real parameter — **both** values are honoured:

```rust
use graphlet::catalog::{count_diamonds, find_diamonds, count_pattern, Induced, Pattern};

let induced     = count_diamonds(&g, Induced::Yes); // exact K4-minus-an-edge subgraphs
let non_induced = count_diamonds(&g, Induced::No);  // diamond monomorphisms (K4s count too)

let occurrences = find_diamonds(&g, Induced::Yes);  // each: spine (shared edge) + tips

// Arbitrary connected catalog pattern:
let pat = Pattern::new(4, &[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]);
let n = count_pattern(&g, &pat, Induced::No);
```

## Induced vs. non-induced

- **Induced** counts are read straight off the census.
- **Non-induced** (monomorphism) counts are derived from the induced census by a
  fixed per-`(P,C)` table `s(P,C)` — the number of edge-preserving bijections of the
  pattern `P` into graphlet class `C`. The identity

  ```
  mono_labelled(P in G) = Σ_C indCount(C) · s(P,C)
  ```

  was verified against an independent brute-force monomorphism oracle (1105/1105 over
  45 hosts, k = 3,4,5). There is **no separate monomorphism enumerator** — the
  non-induced readout is a bounded post-pass. For example, every `K4` contains exactly
  6 non-induced diamonds (its `C(4,2)` tip-pairs).

Both sides report distinct occurrences, obtained from the labelled identity by
dividing by `|Aut(P)| = s(P,P)`.
