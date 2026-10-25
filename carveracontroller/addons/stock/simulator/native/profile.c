/* Tool-shape lookups used by all three carvers.
 *
 * A tool is described by its profile: a list of (height above the tip, radius) samples,
 * with straight lines between them. An end mill is one constant radius, a ball nose has a
 * rounded bottom, and a V-bit is a cone.
 */
#include "stock_carve_internal.h"

#include <math.h>

/* Compute the derived fields: widest radius, cutting length, and whether the tool is a plain cylinder. */
void sc_profile_prepare(ScProfile *p, const double *zs, const double *rs, int n) {
    double max_r;
    double min_r;
    int i;
    p->zs = zs;
    p->rs = rs;
    p->n = n;
    if (n <= 0) {
        p->max_r = 0.0;
        p->flute_z = 0.0;
        p->const_r = -1.0;
        p->mono_r = 1;
        return;
    }
    max_r = rs[0];
    min_r = rs[0];
    p->mono_r = 1;
    for (i = 1; i < n; i++) {
        if (rs[i] > max_r) {
            max_r = rs[i];
        }
        if (rs[i] < min_r) {
            min_r = rs[i];
        }
        if (rs[i] < rs[i - 1] - 1e-12) {
            p->mono_r = 0;
        }
    }
    p->max_r = max_r;
    p->flute_z = zs[n - 1];
    p->const_r = (max_r - min_r < 1e-12) ? max_r : -1.0;
}

/* Tool radius at height `z_rel` above the tip, linearly interpolated between the two surrounding samples.
 * Returns 0 above or below the cutting part.
 */
double sc_sample_radius(const ScProfile *p, double z_rel) {
    int idx;
    double z0, z1, r0, r1, denom, t;
    if (p->n <= 0) {
        return 0.0;
    }
    if (z_rel < p->zs[0] - 1e-9 || z_rel > p->zs[p->n - 1] + 1e-9) {
        return 0.0;
    }
    if (p->n == 1) {
        return p->rs[0];
    }
    /* Last sample at or below z_rel, kept far enough from the end that idx + 1 exists. */
    idx = 0;
    while (idx < p->n && !(p->zs[idx] > z_rel)) {
        idx++;
    }
    idx = sc_clamp_int(idx - 1, 0, p->n - 2);
    z0 = p->zs[idx];
    z1 = p->zs[idx + 1];
    r0 = p->rs[idx];
    r1 = p->rs[idx + 1];
    denom = z1 - z0;
    t = 0.0;
    if (fabs(denom) >= 1e-12) {
        t = (z_rel - z0) / denom;
    }
    return r0 + t * (r1 - r0);
}

/* Inverse of sc_sample_radius for tools that only get wider going up (end mill, ball, cone):
 * binary search for the first sample that reaches `dist`, then interpolate inside that step.
 */
static double sc_sample_z_mono(const ScProfile *p, double dist) {
    int lo;
    int hi;
    int idx;
    double r0, r1, z0, z1, denom, t;
    /* First index with rs[i] >= dist, or n if every sample is narrower. */
    lo = 0;
    hi = p->n;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (p->rs[mid] < dist) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    idx = lo;
    if (idx >= p->n) {
        idx = p->n - 1;
    }
    if (idx == 0) {
        return p->zs[0];
    }
    r0 = p->rs[idx - 1];
    r1 = p->rs[idx];
    z0 = p->zs[idx - 1];
    z1 = p->zs[idx];
    denom = r1 - r0;
    if (fabs(denom) >= 1e-12) {
        t = sc_clamp((dist - r0) / denom, 0.0, 1.0);
        return z0 + t * (z1 - z0);
    }
    return z0 < z1 ? z0 : z1;
}

/* Inverse of sc_sample_radius: the height above the tip where the tool first reaches a radius of `dist`.
 * Returns INFINITY if the tool never gets that wide. A tool that widens and narrows again can reach the
 * same radius at several heights; the lowest one wins, since that is the deepest the tool can cut there.
 */
double sc_sample_z_for_radius(const ScProfile *p, double dist) {
    double z_best;
    int i;
    if (p->n <= 0) {
        return INFINITY;
    }
    if (p->const_r >= 0.0) {
        return dist <= p->const_r + 1e-9 ? p->zs[0] : INFINITY;
    }
    if (dist > p->max_r + 1e-9) {
        return INFINITY;
    }
    if (p->mono_r) {
        return sc_sample_z_mono(p, dist);
    }
    /* General case: check every line between samples, then every sample itself. */
    z_best = INFINITY;
    for (i = 0; i < p->n - 1; i++) {
        double z0 = p->zs[i];
        double z1 = p->zs[i + 1];
        double r0 = p->rs[i];
        double r1 = p->rs[i + 1];
        double cand = INFINITY;
        if (dist <= r0 + 1e-9) {
            cand = z0;
        } else if (fabs(r1 - r0) >= 1e-12 && dist <= r1 + 1e-9) {
            double t = sc_clamp((dist - r0) / (r1 - r0), 0.0, 1.0);
            cand = z0 + t * (z1 - z0);
        }
        if (cand < z_best) {
            z_best = cand;
        }
    }
    for (i = 0; i < p->n; i++) {
        if (dist <= p->rs[i] + 1e-9 && p->zs[i] < z_best) {
            z_best = p->zs[i];
        }
    }
    return z_best;
}

/* Widest tool radius between two heights above the tip. */
double sc_max_radius_in_band(const ScProfile *p, double band_lo, double band_hi) {
    double r;
    double r_hi;
    int i;
    r = sc_sample_radius(p, band_lo);
    r_hi = sc_sample_radius(p, band_hi);
    if (r_hi > r) {
        r = r_hi;
    }
    /* The widest point can also be a profile sample strictly inside the band. */
    for (i = 1; i < p->n - 1; i++) {
        double z = p->zs[i];
        double r_sample;
        if (z < band_lo - 1e-9 || z > band_hi + 1e-9) {
            continue;
        }
        r_sample = sc_sample_radius(p, z);
        if (r_sample > r) {
            r = r_sample;
        }
    }
    return r;
}

void sc_seg_tool_init(
    ScSegTool *seg,
    double p0x,
    double p0y,
    double p0z,
    double p1x,
    double p1y,
    double p1z,
    const ScProfile *profile) {
    seg->p0x = p0x;
    seg->p0y = p0y;
    seg->p0z = p0z;
    seg->p1x = p1x;
    seg->p1y = p1y;
    seg->p1z = p1z;
    seg->dx = p1x - p0x;
    seg->dy = p1y - p0y;
    seg->dz = p1z - p0z;
    seg->xy_len_sq = seg->dx * seg->dx + seg->dy * seg->dy;
    seg->seg_len_sq = seg->xy_len_sq + seg->dz * seg->dz;
    seg->flute_lo_z = profile->n ? profile->zs[0] : 0.0;
    seg->flute_hi_z = profile->n ? profile->zs[profile->n - 1] : 0.0;
    seg->profile = profile;
}

/* Is a point `dist_xy` away from the tool axis inside the tool?
 * `window_ok` says whether the point is at a height the cutting part covers at all.
 * The tool's radius comes from `z_rel` (the point's height above the tip), or, when `use_band` is set,
 * from the widest part of the tool between `band_lo` and `band_hi`.
 */
static int sc_within_tool_radius(
    const ScProfile *profile,
    int window_ok,
    double dist_xy,
    int use_band,
    double z_rel,
    double band_lo,
    double band_hi) {
    double eps = 1e-9;
    double radius;
    if (!window_ok || dist_xy > profile->max_r + eps) {
        return 0;
    }
    if (profile->const_r >= 0.0) {
        return profile->const_r > 0.0;
    }
    if (use_band) {
        radius = sc_max_radius_in_band(profile, band_lo, band_hi);
    } else {
        radius = sc_sample_radius(profile, z_rel);
    }
    return dist_xy <= radius + eps && radius > 0.0;
}

int sc_seg_tool_contains(const ScSegTool *seg, double x, double y, double z) {
    const ScProfile *profile = seg->profile;
    double dz = seg->dz;
    double z0 = seg->flute_lo_z;
    double z1 = seg->flute_hi_z;
    double eps = 1e-9;
    double dist;
    double z_rel;
    int window;

    if (seg->seg_len_sq < 1e-18) {
        /* The tool doesn't move: it just sits at p0. */
        z_rel = z - seg->p0z;
        dist = hypot(x - seg->p0x, y - seg->p0y);
        window = profile->n > 0 && z_rel >= z0 - eps && z_rel <= z1 + eps;
        return sc_within_tool_radius(profile, window, dist, 0, z_rel, 0.0, 0.0);
    }
    if (seg->xy_len_sq < 1e-18) {
        /* Straight plunge: the tip slides along Z, so at this height the tool could be at any radius
         * that the profile has anywhere in the band of heights the point can sit at relative to the tip. */
        double tip_lo = seg->p0z < seg->p0z + dz ? seg->p0z : seg->p0z + dz;
        double tip_hi = seg->p0z > seg->p0z + dz ? seg->p0z : seg->p0z + dz;
        double band_lo = z - tip_hi;
        double band_hi = z - tip_lo;
        if (band_lo < z0) {
            band_lo = z0;
        }
        if (band_hi > z1) {
            band_hi = z1;
        }
        window = band_lo <= band_hi + eps;
        dist = hypot(x - seg->p0x, y - seg->p0y);
        return sc_within_tool_radius(profile, window, dist, 1, 0.0, band_lo, band_hi);
    }
    {
        /* General move. Find the stretch [t_lo, t_hi] of the move (0 = start, 1 = end) during which the
         * point is at a height the cutting part covers. Then the closest tip position to the point,
         * projected onto the XY path and limited to that stretch, decides whether the point is inside. */
        double t_lo, t_hi, t;
        double tip_x, tip_y, tip_z;
        if (fabs(dz) < 1e-18) {
            z_rel = z - seg->p0z;
            window = z_rel >= z0 - eps && z_rel <= z1 + eps;
            t_lo = window ? 0.0 : 1.0;
            t_hi = window ? 1.0 : 0.0;
        } else {
            double t_a = (z - z1 - seg->p0z) / dz;
            double t_b = (z - z0 - seg->p0z) / dz;
            double lo = t_a < t_b ? t_a : t_b;
            double hi = t_a > t_b ? t_a : t_b;
            t_lo = lo > 0.0 ? lo : 0.0;
            t_hi = hi < 1.0 ? hi : 1.0;
            window = t_lo <= t_hi + eps;
        }
        t = ((x - seg->p0x) * seg->dx + (y - seg->p0y) * seg->dy) / seg->xy_len_sq;
        if (t < t_lo) {
            t = t_lo;
        }
        if (t > t_hi) {
            t = t_hi;
        }
        tip_x = seg->p0x + t * seg->dx;
        tip_y = seg->p0y + t * seg->dy;
        tip_z = seg->p0z + t * dz;
        z_rel = z - tip_z;
        dist = hypot(x - tip_x, y - tip_y);
        return sc_within_tool_radius(profile, window, dist, 0, z_rel, 0.0, 0.0);
    }
}
