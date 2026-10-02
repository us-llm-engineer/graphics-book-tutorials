#include "wf/bvh.hpp"
#include "wf/scene.hpp"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace wf {

struct BBox {
    Vec3 lo, hi;
    BBox() : lo{kInf, kInf, kInf}, hi{-kInf, -kInf, -kInf} {}
    BBox(Vec3 p) : lo(p), hi(p) {}
    void expand(Vec3 p) {
        lo.x = fminf3(lo.x, p.x); lo.y = fminf3(lo.y, p.y); lo.z = fminf3(lo.z, p.z);
        hi.x = fmaxf3(hi.x, p.x); hi.y = fmaxf3(hi.y, p.y); hi.z = fmaxf3(hi.z, p.z);
    }
    void expand(BBox b) {
        lo.x = fminf3(lo.x, b.lo.x); lo.y = fminf3(lo.y, b.lo.y); lo.z = fminf3(lo.z, b.lo.z);
        hi.x = fmaxf3(hi.x, b.hi.x); hi.y = fmaxf3(hi.y, b.hi.y); hi.z = fmaxf3(hi.z, b.hi.z);
    }
    float surface_area() const {
        Vec3 d = hi - lo;
        return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
    }
};

static BBox prim_bbox(const Scene& scene, int prim) {
    if (prim < (int)scene.spheres.size()) {
        const Sphere& s = scene.spheres[prim];
        BBox b; b.lo = s.center - Vec3{s.radius, s.radius, s.radius};
        b.hi = s.center + Vec3{s.radius, s.radius, s.radius};
        return b;
    }
    const Triangle& t = scene.triangles[prim - (int)scene.spheres.size()];
    BBox b(scene.vertices[t.i0]);
    b.expand(scene.vertices[t.i1]); b.expand(scene.vertices[t.i2]);
    return b;
}

static Vec3 prim_centroid(const Scene& scene, int prim) {
    if (prim < (int)scene.spheres.size()) {
        return scene.spheres[prim].center;
    }
    const Triangle& t = scene.triangles[prim - (int)scene.spheres.size()];
    Vec3 c = scene.vertices[t.i0] + scene.vertices[t.i1] + scene.vertices[t.i2];
    return c * (1.0f / 3.0f);
}

// Temporary node in the intermediate tree (before flattening to preorder).
struct TempNode {
    Vec3 lo, hi;
    int left, right;      // child indices, or -1 for leaves
    int prim_start, prim_count;
    int subtree_size;     // number of nodes in this subtree (computed bottom-up)
};

// Step 1: Build the temporary tree using binned SAH.
// Returns the root node index in temp_nodes.
static int build_temp_tree(const Scene& scene, std::vector<TempNode>& temp_nodes,
                          std::vector<int>& prim_order, std::vector<int> prims) {
    if (prims.empty()) return -1;

    int node_id = (int)temp_nodes.size();
    TempNode node{};
    node.left = -1;
    node.right = -1;
    node.subtree_size = 1;

    // Compute bounds and centroid bounds.
    BBox bounds, centroid_bounds;
    for (int p : prims) {
        BBox pb = prim_bbox(scene, p);
        bounds.expand(pb);
        centroid_bounds.expand(prim_centroid(scene, p));
    }
    node.lo = bounds.lo;
    node.hi = bounds.hi;

    // If leaf: store primitives and return.
    if ((int)prims.size() <= 4) {
        node.prim_start = (int)prim_order.size();
        node.prim_count = (int)prims.size();
        for (int p : prims) prim_order.push_back(p);
        temp_nodes.push_back(node);
        return node_id;
    }

    // Internal node: split using binned SAH.
    node.prim_count = 0;
    node.prim_start = -1;

    Vec3 centroid_extent = centroid_bounds.hi - centroid_bounds.lo;
    int split_axis = 0;
    if (centroid_extent.y > centroid_extent.x) split_axis = 1;
    if (centroid_extent.z > centroid_extent.x && centroid_extent.z > centroid_extent.y) split_axis = 2;

    // Degenerate case: identical centroids (or nearly so). Split by primitive index.
    if (centroid_extent.x < 1e-6f && centroid_extent.y < 1e-6f && centroid_extent.z < 1e-6f) {
        size_t mid = prims.size() / 2;
        std::vector<int> left(prims.begin(), prims.begin() + mid);
        std::vector<int> right(prims.begin() + mid, prims.end());

        temp_nodes.push_back(node);
        int left_id = build_temp_tree(scene, temp_nodes, prim_order, left);
        int right_id = build_temp_tree(scene, temp_nodes, prim_order, right);

        if (left_id >= 0) {
            temp_nodes[node_id].left = left_id;
            temp_nodes[node_id].subtree_size += temp_nodes[left_id].subtree_size;
        }
        if (right_id >= 0) {
            temp_nodes[node_id].right = right_id;
            temp_nodes[node_id].subtree_size += temp_nodes[right_id].subtree_size;
        }
        return node_id;
    }

    // Normal SAH binning.
    float axis_min = (split_axis == 0) ? centroid_bounds.lo.x :
                     (split_axis == 1) ? centroid_bounds.lo.y : centroid_bounds.lo.z;
    float axis_max = (split_axis == 0) ? centroid_bounds.hi.x :
                     (split_axis == 1) ? centroid_bounds.hi.y : centroid_bounds.hi.z;
    float axis_range = axis_max - axis_min;

    const int n_bins = 8;
    struct Bin { BBox bounds; int count = 0; } bins[n_bins] = {};

    for (int p : prims) {
        Vec3 c = prim_centroid(scene, p);
        float coord = (split_axis == 0) ? c.x : (split_axis == 1) ? c.y : c.z;
        int bin_id = (int)((coord - axis_min) / axis_range * (n_bins - 0.0001f));
        bin_id = (bin_id < 0) ? 0 : (bin_id >= n_bins) ? n_bins - 1 : bin_id;
        bins[bin_id].bounds.expand(prim_bbox(scene, p));
        bins[bin_id].count++;
    }

    float best_cost = 1e9f;
    int best_split = 1;
    for (int split = 1; split < n_bins; ++split) {
        BBox left_bounds, right_bounds;
        int left_count = 0, right_count = 0;
        for (int i = 0; i < split; ++i) {
            left_bounds.expand(bins[i].bounds);
            left_count += bins[i].count;
        }
        for (int i = split; i < n_bins; ++i) {
            right_bounds.expand(bins[i].bounds);
            right_count += bins[i].count;
        }
        float cost = (left_count * left_bounds.surface_area() +
                     right_count * right_bounds.surface_area()) / bounds.surface_area();
        if (cost < best_cost) { best_cost = cost; best_split = split; }
    }

    std::vector<int> left, right;
    for (int p : prims) {
        Vec3 c = prim_centroid(scene, p);
        float coord = (split_axis == 0) ? c.x : (split_axis == 1) ? c.y : c.z;
        int bin_id = (int)((coord - axis_min) / axis_range * (n_bins - 0.0001f));
        bin_id = (bin_id < 0) ? 0 : (bin_id >= n_bins) ? n_bins - 1 : bin_id;
        if (bin_id < best_split) left.push_back(p);
        else right.push_back(p);
    }

    // Fallback: split by index if SAH split is empty.
    if (left.empty() || right.empty()) {
        size_t mid = prims.size() / 2;
        left.assign(prims.begin(), prims.begin() + mid);
        right.assign(prims.begin() + mid, prims.end());
    }

    temp_nodes.push_back(node);
    int left_id = build_temp_tree(scene, temp_nodes, prim_order, left);
    int right_id = build_temp_tree(scene, temp_nodes, prim_order, right);

    if (left_id >= 0) {
        temp_nodes[node_id].left = left_id;
        temp_nodes[node_id].subtree_size += temp_nodes[left_id].subtree_size;
    }
    if (right_id >= 0) {
        temp_nodes[node_id].right = right_id;
        temp_nodes[node_id].subtree_size += temp_nodes[right_id].subtree_size;
    }

    return node_id;
}

// Step 2: Flatten temporary tree to preorder with skip links.
// Returns the number of nodes appended.
static int flatten(const std::vector<TempNode>& temp_nodes, int tmp_node, int skip,
                  std::vector<BvhNode>& out) {
    if (tmp_node < 0) return 0;

    const TempNode& tn = temp_nodes[tmp_node];
    int idx = (int)out.size();

    BvhNode node;
    node.lo = tn.lo;
    node.hi = tn.hi;
    node.skip = skip;
    node.prim_start = tn.prim_start;
    node.prim_count = tn.prim_count;

    out.push_back(node);

    if (tn.prim_count > 0) {
        // Leaf: no children.
        return 1;
    }

    // Internal node: flatten children in order (left, then right).
    // Left child is at idx+1.
    // Right child is at idx + 1 + subtree_size(left).
    int right_idx = idx + 1 + (tn.left >= 0 ? temp_nodes[tn.left].subtree_size : 0);

    int count = 1;
    if (tn.left >= 0) {
        count += flatten(temp_nodes, tn.left, right_idx, out);
    }
    if (tn.right >= 0) {
        count += flatten(temp_nodes, tn.right, skip, out);
    }

    return count;
}

void build_bvh(Scene& scene) {
    if (scene.spheres.empty() && scene.triangles.empty()) {
        throw std::runtime_error("Cannot build BVH: no primitives");
    }

    scene.nodes.clear();
    scene.prim_order.clear();

    int n_prims = (int)scene.spheres.size() + (int)scene.triangles.size();
    std::vector<int> prims(n_prims);
    for (int i = 0; i < n_prims; ++i) prims[i] = i;

    // Step 1: Build temporary tree.
    std::vector<TempNode> temp_nodes;
    std::vector<int> prim_order;
    int root_id = build_temp_tree(scene, temp_nodes, prim_order, prims);

    // Step 2: Flatten to preorder with skip links.
    if (root_id >= 0) {
        flatten(temp_nodes, root_id, -1, scene.nodes);
    }

    scene.prim_order = prim_order;
}

std::string validate_bvh(const Scene& scene) {
    int n_prims = (int)scene.spheres.size() + (int)scene.triangles.size();

    // Check primitive coverage.
    std::vector<int> count(n_prims, 0);
    for (int p : scene.prim_order) {
        if (p < 0 || p >= n_prims) return "primitive index out of range";
        count[p]++;
    }
    if ((int)scene.prim_order.size() != n_prims) return "not all primitives in BVH";
    for (int c : count) if (c != 1) return "primitive appears multiple times or not at all";

    // Helper: check if a box is contained within another (with small slack for floating point).
    auto box_contains = [](Vec3 parent_lo, Vec3 parent_hi, Vec3 child_lo, Vec3 child_hi) -> bool {
        const float slack = 1e-5f;
        return parent_lo.x - slack <= child_lo.x && child_hi.x <= parent_hi.x + slack &&
               parent_lo.y - slack <= child_lo.y && child_hi.y <= parent_hi.y + slack &&
               parent_lo.z - slack <= child_lo.z && child_hi.z <= parent_hi.z + slack;
    };

    // Check node structure.
    for (int i = 0; i < (int)scene.nodes.size(); ++i) {
        const BvhNode& n = scene.nodes[i];

        // Check skip links.
        if (n.skip < -1 || n.skip >= (int)scene.nodes.size()) return "skip link out of bounds";
        if (n.skip != -1 && n.skip <= i) return "skip link does not point forward";

        if (n.prim_count > 0) {
            // Leaf node.
            if (n.prim_count > 4) return "leaf has more than 4 primitives";
            if (n.prim_start < 0 || n.prim_start + n.prim_count > (int)scene.prim_order.size())
                return "leaf prim_start/count out of bounds";

            // Check that leaf box contains its primitives.
            for (int k = 0; k < n.prim_count; ++k) {
                int prim = scene.prim_order[n.prim_start + k];
                BBox pb = prim_bbox(scene, prim);
                if (!box_contains(n.lo, n.hi, pb.lo, pb.hi))
                    return "leaf box does not contain its primitive";
            }
        } else {
            // Internal node.
            if (n.prim_start != -1) return "internal node has prim_start != -1";
            if (i + 1 >= (int)scene.nodes.size()) return "left child (N+1) out of bounds";

            // Left child is at i+1; its skip should be its right sibling.
            int right_child = scene.nodes[i + 1].skip;
            if (right_child <= i + 1 || right_child >= (int)scene.nodes.size())
                return "right child index invalid";

            // Right child should share parent's skip.
            if (scene.nodes[right_child].skip != n.skip)
                return "right child does not share parent's skip";

            // Check that parent box contains child boxes.
            const BvhNode& left = scene.nodes[i + 1];
            const BvhNode& right = scene.nodes[right_child];
            if (!box_contains(n.lo, n.hi, left.lo, left.hi))
                return "parent box does not contain left child";
            if (!box_contains(n.lo, n.hi, right.lo, right.hi))
                return "parent box does not contain right child";
        }
    }

    // Check that the last node has skip -1.
    if (!scene.nodes.empty() && scene.nodes.back().skip != -1)
        return "last node skip is not -1";

    return "";
}

}  // namespace wf
