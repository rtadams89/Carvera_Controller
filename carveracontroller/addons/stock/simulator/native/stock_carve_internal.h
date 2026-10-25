/* Helpers shared by the carvers: small math utilities, and the test for
 * "is this point inside the tool while it moves along a straight line".
 */
#ifndef CARVERA_STOCK_CARVE_INTERNAL_H
#define CARVERA_STOCK_CARVE_INTERNAL_H

#include "stock_carve.h"

#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline double sc_clamp(double v, double lo, double hi) {
    if (v < lo) {
        return lo;
    }
    if (v > hi) {
        return hi;
    }
    return v;
}

static inline int sc_clamp_int(int v, int lo, int hi) {
    if (v < lo) {
        return lo;
    }
    if (v > hi) {
        return hi;
    }
    return v;
}

/* Clip the index range [*lo, *hi] to the valid indices [0, n). Returns 0 if nothing is left. */
static inline int sc_clip_range(int *lo, int *hi, int n) {
    if (*lo < 0) {
        *lo = 0;
    }
    if (*hi > n - 1) {
        *hi = n - 1;
    }
    return *lo <= *hi;
}

/* Use `v` as a divisor / cell size without ever dividing by (nearly) zero. */
static inline double sc_nonzero(double v) {
    return v > 1e-12 ? v : 1e-12;
}

/* Remainder in [0, b), matching Python's `%`. */
static inline double sc_py_mod(double a, double b) {
    double m = fmod(a, b);
    if (m < 0.0) {
        m += b;
    }
    return m;
}

/* Rotate (y, z) around the X axis, which is the machine's A axis. `c` and `s` are the cosine and sine of the
 * angle, so callers rotating many points by the same angle only pay for cos/sin once.
 */
static inline void sc_rotate_yz_cs(double y, double z, double c, double s, double *oy, double *oz) {
    *oy = y * c - z * s;
    *oz = y * s + z * c;
}

/* Tool shape lookups (profile.c). The profile is the tool's side view:
 * radius `rs[i]` at height `zs[i]` above the tip.
 */
double sc_sample_radius(const ScProfile *p, double z_rel);      /* radius at a height above the tip */
double sc_sample_z_for_radius(const ScProfile *p, double dist); /* lowest height whose radius reaches `dist` */
double sc_max_radius_in_band(const ScProfile *p, double band_lo, double band_hi);

/* One tool move (tip travelling p0 -> p1), with the values every point test needs precomputed.
 * Build it once with sc_seg_tool_init, then test many points with sc_seg_tool_contains.
 */
typedef struct ScSegTool {
    double p0x, p0y, p0z;
    double p1x, p1y, p1z;
    double dx, dy, dz;      /* p1 - p0 */
    double xy_len_sq;       /* dx^2 + dy^2 */
    double seg_len_sq;      /* dx^2 + dy^2 + dz^2 */
    double flute_lo_z;      /* lowest profile height */
    double flute_hi_z;      /* highest profile height (top of the cutting part) */
    const ScProfile *profile;
} ScSegTool;

void sc_seg_tool_init(
    ScSegTool *seg,
    double p0x,
    double p0y,
    double p0z,
    double p1x,
    double p1y,
    double p1z,
    const ScProfile *profile);

/* True if (x, y, z) is inside the volume swept by the tool during the move. */
int sc_seg_tool_contains(const ScSegTool *seg, double x, double y, double z);

#endif
