// CPU unit tests for the mesh-distance dispatch (ff::cpu::dt_mesh).
//
// A normal (B, D) batch of query points is evaluated against a single shared
// 3D triangle mesh (vertices (N, D), faces (M, D)). Distances are compared to
// an analytic point-to-triangle squared-distance reference, and the returned
// nearest_vertex is compared to the nearest of the triangle's own vertices.
//
// This exercises the dispatch fixes:
//   - nb_faces / nb_vertices taken from axis 0 (M / N), not the last axis (D)
//   - the query batch (B) is independent of N / M (no bogus batch-equality check)
//
// Build (from fastfields-cpu-lib):
//   clang++ -std=c++11 -O2 -I. tests/test_distance_mesh.cpp distance.cpp \
//       -o build/test_distance_mesh && ./build/test_distance_mesh

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <array>
#include <limits>
#include <random>
#include <stdexcept>
#include "dlpack.h"
#include "distance.h"

namespace {

constexpr double INF = std::numeric_limits<double>::infinity();

template <typename T>
DLTensor make_cpu_tensor(T* data, std::vector<int64_t>& shape,
                         std::vector<int64_t>& strides, uint8_t code, uint8_t bits)
{
    DLTensor t;
    t.data                 = static_cast<void*>(data);
    t.device.device_type   = kDLCPU;
    t.device.device_id     = 0;
    t.ndim                 = static_cast<int32_t>(shape.size());
    t.dtype.code           = code;
    t.dtype.bits           = bits;
    t.dtype.lanes          = 1;
    t.shape                = shape.data();
    t.strides              = strides.data();
    t.byte_offset          = 0;
    return t;
}

std::vector<int64_t> contiguous_strides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> s(shape.size());
    int64_t acc = 1;
    for (int64_t d = (int64_t)shape.size() - 1; d >= 0; --d) {
        s[d] = acc;
        acc *= shape[d];
    }
    return s;
}

int g_failures = 0;
int g_checks   = 0;

void check_close(double a, double b, const char* what, double tol = 1e-3)
{
    ++g_checks;
    double diff  = std::fabs(a - b);
    double scale = std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
    if (diff / scale > tol) {
        ++g_failures;
        std::printf("  MISMATCH [%s]: got %.6g expected %.6g\n", what, a, b);
    }
}

void check_true(bool cond, const char* what)
{
    ++g_checks;
    if (!cond) { ++g_failures; std::printf("  FAIL [%s]\n", what); }
}

using Vec3 = std::array<double, 3>;
Vec3 sub(const Vec3& a, const Vec3& b){ return {a[0]-b[0], a[1]-b[1], a[2]-b[2]}; }
Vec3 add(const Vec3& a, const Vec3& b){ return {a[0]+b[0], a[1]+b[1], a[2]+b[2]}; }
Vec3 mul(const Vec3& a, double s){ return {a[0]*s, a[1]*s, a[2]*s}; }
double dot(const Vec3& a, const Vec3& b){ return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }

// Closest point on triangle (a,b,c) to p (Ericson, Real-Time Collision
// Detection). Returns squared distance.
double point_tri_sqdist(const Vec3& p, const Vec3& a, const Vec3& b, const Vec3& c)
{
    Vec3 ab = sub(b, a), ac = sub(c, a), ap = sub(p, a);
    double d1 = dot(ab, ap), d2 = dot(ac, ap);
    Vec3 closest;
    if (d1 <= 0 && d2 <= 0) { closest = a; }
    else {
        Vec3 bp = sub(p, b);
        double d3 = dot(ab, bp), d4 = dot(ac, bp);
        if (d3 >= 0 && d4 <= d3) { closest = b; }
        else {
            Vec3 cp = sub(p, c);
            double d5 = dot(ab, cp), d6 = dot(ac, cp);
            if (d6 >= 0 && d5 <= d6) { closest = c; }
            else {
                double vc = d1*d4 - d3*d2;
                if (vc <= 0 && d1 >= 0 && d3 <= 0) {
                    double v = d1 / (d1 - d3);
                    closest = add(a, mul(ab, v));
                } else {
                    double vb = d5*d2 - d1*d6;
                    if (vb <= 0 && d2 >= 0 && d6 <= 0) {
                        double w = d2 / (d2 - d6);
                        closest = add(a, mul(ac, w));
                    } else {
                        double va = d3*d6 - d5*d4;
                        if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
                            double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
                            closest = add(b, mul(sub(c, b), w));
                        } else {
                            double denom = 1.0 / (va + vb + vc);
                            double v = vb * denom, w = vc * denom;
                            closest = add(a, add(mul(ab, v), mul(ac, w)));
                        }
                    }
                }
            }
        }
    }
    Vec3 d = sub(p, closest);
    return dot(d, d);
}

template <typename scalar_t, typename index_t>
void run_triangle(uint8_t bits, uint8_t ibits, unsigned seed, bool naive)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-3.0, 3.0);

    // A single (non-degenerate) triangle.
    Vec3 v0 = {0.0, 0.0, 0.0};
    Vec3 v1 = {2.0, 0.0, 0.0};
    Vec3 v2 = {0.0, 1.5, 0.0};
    std::array<Vec3, 3> V = {v0, v1, v2};

    const int64_t N = 3, M = 1, D = 3, B = 20;

    std::vector<scalar_t> vertices(N * D);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t d = 0; d < D; ++d)
            vertices[n * D + d] = (scalar_t)V[n][d];
    std::vector<index_t> faces = {0, 1, 2}; // one triangle

    std::vector<scalar_t> loc(B * D);
    std::vector<Vec3>     loc_d(B);
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t d = 0; d < D; ++d) {
            double val = u(rng);
            loc_d[b][d] = val;
            loc[b * D + d] = (scalar_t)val;
        }
    }

    std::vector<scalar_t> dist_out(B, 0);
    std::vector<index_t>  near_out(B, (index_t)-1);

    std::vector<int64_t> sh_loc  = {B, D}, st_loc  = contiguous_strides(sh_loc);
    std::vector<int64_t> sh_vert = {N, D}, st_vert = contiguous_strides(sh_vert);
    std::vector<int64_t> sh_face = {M, D}, st_face = contiguous_strides(sh_face);
    std::vector<int64_t> sh_out  = {B},    st_out  = contiguous_strides(sh_out);

    DLTensor t_loc  = make_cpu_tensor(loc.data(),      sh_loc,  st_loc,  kDLFloat, bits);
    DLTensor t_vert = make_cpu_tensor(vertices.data(), sh_vert, st_vert, kDLFloat, bits);
    DLTensor t_face = make_cpu_tensor(faces.data(),    sh_face, st_face, kDLInt,   ibits);
    DLTensor t_dist = make_cpu_tensor(dist_out.data(), sh_out,  st_out,  kDLFloat, bits);
    DLTensor t_near = make_cpu_tensor(near_out.data(), sh_out,  st_out,  kDLInt,   ibits);

    // Signed distance (the dt_mesh default). The kernel returns the *Euclidean*
    // (signed) distance, so |dist| == distance to the triangle. The signed path
    // is also the one that populates nearest_vertex.
    ff::cpu::dt_mesh(t_dist, t_near, t_loc, t_vert, t_face,
                     /*_signed=*/true, /*naive=*/naive, 0);

    for (int64_t b = 0; b < B; ++b) {
        double ref = std::sqrt(point_tri_sqdist(loc_d[b], V[0], V[1], V[2]));
        check_close(std::fabs((double)dist_out[b]), ref,
                    naive ? "mesh.dist.naive" : "mesh.dist.tree");

        // Nearest triangle vertex to the query point.
        index_t ref_v = 0; double best = INF;
        for (int64_t n = 0; n < N; ++n) {
            Vec3 dd = sub(loc_d[b], V[n]);
            double dn = dot(dd, dd);
            if (dn < best) { best = dn; ref_v = (index_t)n; }
        }
        check_true(near_out[b] == ref_v,
                   naive ? "mesh.near.naive" : "mesh.near.tree");
    }
}

// Multi-triangle mesh with faces given in a deliberately NON-BVH-sorted order.
// Regression for three bugs the single-triangle test could not see:
//   * signed-dt tree path must query the *reordered* faces (build_sdt) -- else
//     it returns wrong distances for unsorted meshes;
//   * the dtype dispatch must pick the index width from faces, not loc, so
//     mixed float/int widths (e.g. float32 coords + int64 faces) work;
//   * the *unsigned* path must actually write nearest_vertex.
template <typename scalar_t, typename index_t>
void run_multi(uint8_t bits, uint8_t ibits, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uv(-2.0, 2.0);
    std::uniform_real_distribution<double> uq(-4.0, 4.0);

    const int64_t N = 12, D = 3, B = 30;
    std::vector<Vec3> Vd(N);
    std::vector<scalar_t> vertices(N * D);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t d = 0; d < D; ++d) {
            double val = uv(rng);
            Vd[n][d] = val;
            vertices[n * D + d] = (scalar_t)val;
        }

    // 8 triangles, listed in an order that is not BVH-sorted, so build_tree's
    // in-place sort genuinely reorders them.
    std::vector<std::array<index_t,3>> tris = {
        {0,5,9},{7,1,3},{11,2,4},{6,8,10},{2,7,0},{9,4,1},{10,3,5},{8,11,6}
    };
    const int64_t M = (int64_t)tris.size();
    std::vector<index_t> faces(M * D);
    for (int64_t m = 0; m < M; ++m)
        for (int64_t d = 0; d < D; ++d) faces[m * D + d] = tris[m][d];

    std::vector<scalar_t> loc(B * D);
    std::vector<Vec3>     loc_d(B);
    for (int64_t b = 0; b < B; ++b)
        for (int64_t d = 0; d < D; ++d) {
            double val = uq(rng);
            loc_d[b][d] = val;
            loc[b * D + d] = (scalar_t)val;
        }

    std::vector<int64_t> sh_loc={B,D}, st_loc=contiguous_strides(sh_loc);
    std::vector<int64_t> sh_vert={N,D}, st_vert=contiguous_strides(sh_vert);
    std::vector<int64_t> sh_face={M,D}, st_face=contiguous_strides(sh_face);
    std::vector<int64_t> sh_out={B},    st_out=contiguous_strides(sh_out);

    DLTensor t_loc  = make_cpu_tensor(loc.data(),      sh_loc,  st_loc,  kDLFloat, bits);
    DLTensor t_vert = make_cpu_tensor(vertices.data(), sh_vert, st_vert, kDLFloat, bits);
    DLTensor t_face = make_cpu_tensor(faces.data(),    sh_face, st_face, kDLInt,   ibits);

    // Oracle: unsigned distance = min over all triangles; nearest face too.
    auto oracle = [&](const Vec3& p, int64_t& face_out){
        double best = INF; face_out = 0;
        for (int64_t m = 0; m < M; ++m) {
            double dm = point_tri_sqdist(p, Vd[tris[m][0]], Vd[tris[m][1]], Vd[tris[m][2]]);
            if (dm < best) { best = dm; face_out = m; }
        }
        return std::sqrt(best);
    };

    // Signed dt via the tree (default) and the naive path: |dist| must match the
    // brute-force oracle for this unsorted mesh (A1 regression).
    for (int naive = 0; naive < 2; ++naive) {
        std::vector<scalar_t> dist_out(B, 0);
        DLTensor t_dist = make_cpu_tensor(dist_out.data(), sh_out, st_out, kDLFloat, bits);
        DLTensor t_null; t_null.data = nullptr; t_null.ndim = 0;
        ff::cpu::dt_mesh(t_dist, t_null, t_loc, t_vert, t_face,
                         /*_signed=*/true, /*naive=*/(bool)naive, 0);
        for (int64_t b = 0; b < B; ++b) {
            int64_t f; double ref = oracle(loc_d[b], f);
            check_close(std::fabs((double)dist_out[b]), ref,
                        naive ? "mesh.multi.dist.naive" : "mesh.multi.dist.tree");
        }
    }

    // Unsigned dt must populate nearest_vertex (A5) and match the oracle
    // distance; the written index must be a real vertex of the nearest face.
    {
        std::vector<scalar_t> dist_out(B, 0);
        std::vector<index_t>  near_out(B, (index_t)-1);
        DLTensor t_dist = make_cpu_tensor(dist_out.data(), sh_out, st_out, kDLFloat, bits);
        DLTensor t_near = make_cpu_tensor(near_out.data(), sh_out, st_out, kDLInt, ibits);
        ff::cpu::dt_mesh(t_dist, t_near, t_loc, t_vert, t_face,
                         /*_signed=*/false, /*naive=*/false, 0);
        for (int64_t b = 0; b < B; ++b) {
            int64_t f; double ref = oracle(loc_d[b], f);
            check_close((double)dist_out[b], ref, "mesh.multi.udist");
            check_true(near_out[b] != (index_t)-1, "mesh.multi.near.written");
            bool of_face = (near_out[b] == tris[f][0]) ||
                           (near_out[b] == tris[f][1]) ||
                           (near_out[b] == tris[f][2]);
            check_true(of_face && near_out[b] >= 0 && near_out[b] < (index_t)N,
                       "mesh.multi.near.valid");
        }
    }
}

// --- B4: negative / validation tests --------------------------------------
// A single valid (float32 coords + int32 faces) setup, perturbed per test.
// Bad dtype: float16 coordinates must throw at the scalar dispatch.
void test_bad_dtype_throws()
{
    const int64_t N = 3, M = 1, D = 3, B = 4;
    std::vector<uint16_t> loc(B*D,0), vert(N*D,0), dist(B,0);  // float16 payload
    std::vector<int32_t>  faces = {0,1,2}, near(B,0);
    std::vector<int64_t> sh_loc={B,D},st_loc=contiguous_strides(sh_loc);
    std::vector<int64_t> sh_v={N,D},st_v=contiguous_strides(sh_v);
    std::vector<int64_t> sh_f={M,D},st_f=contiguous_strides(sh_f);
    std::vector<int64_t> sh_o={B},st_o=contiguous_strides(sh_o);
    DLTensor t_loc =make_cpu_tensor(loc.data(),  sh_loc,st_loc,kDLFloat,16);
    DLTensor t_vert=make_cpu_tensor(vert.data(), sh_v,  st_v,  kDLFloat,16);
    DLTensor t_face=make_cpu_tensor(faces.data(),sh_f,  st_f,  kDLInt,  32);
    DLTensor t_dist=make_cpu_tensor(dist.data(), sh_o,  st_o,  kDLFloat,16);
    DLTensor t_near=make_cpu_tensor(near.data(), sh_o,  st_o,  kDLInt,  32);
    bool threw = false;
    try { ff::cpu::dt_mesh(t_dist,t_near,t_loc,t_vert,t_face,true,false,0); }
    catch (const std::exception&) { threw = true; }
    ++g_checks;
    if (!threw) { ++g_failures; std::printf("  FAIL [distance_mesh.bad_dtype_throws]\n"); }
}

// Shape mismatch: vertices must be a rank-2 (N, D) tensor; a rank-1 vertices
// tensor must throw.
void test_shape_mismatch_throws()
{
    const int64_t N = 3, M = 1, D = 3, B = 4;
    std::vector<double> loc(B*D,0), vert(N*D,0), dist(B,0);
    std::vector<int64_t> faces = {0,1,2}, near(B,0);
    std::vector<int64_t> sh_loc={B,D},st_loc=contiguous_strides(sh_loc);
    std::vector<int64_t> sh_vbad={N*D},st_vbad=contiguous_strides(sh_vbad); // rank-1 (bad)
    std::vector<int64_t> sh_f={M,D},st_f=contiguous_strides(sh_f);
    std::vector<int64_t> sh_o={B},st_o=contiguous_strides(sh_o);
    DLTensor t_loc =make_cpu_tensor(loc.data(),  sh_loc, st_loc, kDLFloat,64);
    DLTensor t_vert=make_cpu_tensor(vert.data(), sh_vbad,st_vbad,kDLFloat,64);
    DLTensor t_face=make_cpu_tensor(faces.data(),sh_f,   st_f,   kDLInt,  64);
    DLTensor t_dist=make_cpu_tensor(dist.data(), sh_o,   st_o,   kDLFloat,64);
    DLTensor t_near=make_cpu_tensor(near.data(), sh_o,   st_o,   kDLInt,  64);
    bool threw = false;
    try { ff::cpu::dt_mesh(t_dist,t_near,t_loc,t_vert,t_face,true,false,0); }
    catch (const std::exception&) { threw = true; }
    ++g_checks;
    if (!threw) { ++g_failures; std::printf("  FAIL [distance_mesh.shape_mismatch_throws]\n"); }
}

} // namespace

int main()
{
    std::printf("mesh distance CPU tests\n");
    test_bad_dtype_throws();
    test_shape_mismatch_throws();
    for (unsigned seed = 1; seed <= 8; ++seed) {
        run_triangle<float,  int32_t>(32, 32, seed,       false);
        run_triangle<double, int64_t>(64, 64, seed + 100, false);
        run_triangle<float,  int32_t>(32, 32, seed + 200, true);
        run_triangle<double, int64_t>(64, 64, seed + 300, true);
        // Mixed float/int widths now that dispatch reads faces.dtype.bits (A2).
        run_triangle<float,  int64_t>(32, 64, seed + 400, false);
        run_triangle<double, int32_t>(64, 32, seed + 500, true);
    }
    for (unsigned seed = 1; seed <= 6; ++seed) {
        run_multi<float,  int32_t>(32, 32, seed + 600);
        run_multi<double, int64_t>(64, 64, seed + 700);
        run_multi<float,  int64_t>(32, 64, seed + 800);   // mixed widths (A2)
    }
    std::printf("checks: %d, failures: %d\n", g_checks, g_failures);
    if (g_failures) { std::printf("FAILED\n"); return 1; }
    std::printf("PASSED\n");
    return 0;
}
