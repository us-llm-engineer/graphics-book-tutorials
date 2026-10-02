# Bounding volume hierarchy

Sources: [`include/wf/bvh.hpp`](../../include/wf/bvh.hpp) (traversal and intersection, header-only), [`src/bvh_build.cpp`](../../src/bvh_build.cpp) (build).

## Build

`build_bvh` constructs a binary tree over all primitives (spheres and triangles share one primitive id space) with a binned surface-area-heuristic split and at most four primitives per leaf; identical primitives still respect the limit. Nodes are stored in preorder (48 bytes each: box, `skip`, `prim_start`, `prim_count`): the left child of node `N` is `N+1`, the right child of a left child is its `skip`, and a right child shares its parent's `skip`. The `skip` of the last node on the right-most path is `-1`.

## Stackless traversal

The loop visits a node; if the ray hits its box it continues to `N+1` (descending to the left child, testing leaf primitives on arrival), otherwise or after a leaf it jumps to `skip`. No per-thread stack is needed, which keeps register pressure and local-memory traffic low in both GPU arms. The idea follows the stackless-traversal formulation in the literature [4] (see the repository README references).

## Oracles and invariants

- `intersect_bruteforce` tests every primitive; `intersect` and the oracle must agree on primitive and distance within 1e-5 relative for 100000 random rays on every scene.
- `validate_bvh` checks that every primitive appears exactly once, boxes contain children and primitives, skip links point forward or to `-1`, and the preorder property holds.

## Measured quality

Figure 8 reports leaf-depth distribution, mean child-to-parent surface-area ratio per depth and SAH cost for the three scenes (data: `results/stats/stats.json`, section `bvh`). The area ratio sits near one half at every depth, the signature of balanced spatial splits.
