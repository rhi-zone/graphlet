---
layout: home

hero:
  name: graphlet
  text: Structural graph mining for petgraph
  tagline: Subgraph census, graphlet-degree vectors (GDV/GDD), per-node orbits, and network-motif detection — one census substrate, minimal dependencies.
  actions:
    - theme: brand
      text: Introduction
      link: /introduction
    - theme: alt
      text: View on GitHub
      link: https://github.com/rhi-zone/graphlet

features:
  - title: One census substrate
    details: enumerate connected k-subsets → canonically label → fold. Instance enumeration and counting are two readouts of a single pass; counting streams, so memory tracks graph size, not the number of instances.
  - title: Orbits, GDV & GDD
    details: Per-node attribution to all 73 automorphism orbits (k ≤ 5, undirected), verified against an independent brute-force oracle.
  - title: Named motifs, honest semantics
    details: Induced and non-induced (monomorphism) counts for catalog patterns — the non-induced readout derived from the induced census via a fixed s(P,C) table, no separate enumerator.
  - title: Minimal dependencies
    details: Depends only on petgraph and rand. Generic over Graph / StableGraph, directed / undirected, and arbitrary weights through one trait-bound set.
---
